//  Copyright (c) 2025 Couchbase, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 		http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package zap

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/ajroetker/go-highway/hwy/contrib/varint"
)

// UseStreamVByte controls whether new segments use StreamVByte encoding
// for location data. Set to false to disable StreamVByte encoding.
var UseStreamVByte = true

// UseSeparatedFieldIDs controls whether location data stores field IDs
// in a separate stream from values. When false (default), single-stream format
// is used which has fewer allocations and better merge performance.
// When true, field IDs and values are stored in separate streams.
// Only effective when UseStreamVByte is true.
var UseSeparatedFieldIDs = false

// UseDeltaEncoding controls whether StreamVByte uses delta encoding.
// Delta encoding stores differences between consecutive values, which
// produces smaller values that compress better with StreamVByte.
// NOTE: Currently disabled because location data is interleaved (fieldID, pos,
// start, end, numAP) rather than sorted, so delta encoding can produce larger
// values (negative deltas wrap to large unsigned values).
// Use UseColumnarLocations instead for proper delta encoding.
// Only effective when UseStreamVByte is true.
var UseDeltaEncoding = false

// UseColumnarLocations controls whether location data uses columnar format.
// Columnar format separates each field (fieldID, pos, start, end, numAP, arrayPos)
// into separate columns, enabling SIMD-accelerated delta encoding for positions
// and starts (monotonically increasing), and storing lengths (end-start) directly.
//
// Performance (large positions, 20 locations):
//   - 17% smaller than varint (163 vs 197 bytes)
//   - 25% faster CPU time (188 vs 252 ns)
//   - 23% faster total time on NVMe (296 vs 384 ns)
//
// Only effective when UseStreamVByte is true.
var UseColumnarLocations = true

// StreamVByte chunk format:
//
//	[format byte] [numValues varint] [controlLen varint] [control bytes] [data bytes]
//
// Format byte values:
const (
	ChunkFormatVarint           = 0x00 // Legacy varint encoding
	ChunkFormatStreamVByte      = 0x01 // StreamVByte encoding
	ChunkFormatStreamVByteDelta = 0x02 // StreamVByte with naive delta encoding (disabled)
	ChunkFormatColumnar         = 0x03 // Columnar format with delta encoding for start/end
)

var streamVByteControlByteLengths [256]uint8

func init() {
	for ctrl := range streamVByteControlByteLengths {
		length := 0
		for shift := 0; shift < 8; shift += 2 {
			length += int((byte(ctrl)>>shift)&0x03) + 1
		}
		streamVByteControlByteLengths[ctrl] = uint8(length)
	}
}

// decodeStreamVByteFull decodes exactly len(dst) values from control/data.
// The SIMD kernel needs a full 16-byte load window per group; on groups
// without one its tail paths are unreliable (observed on linux/arm64:
// zero-filled output with decoded/dataConsumed still reporting success), so
// it is only ever given the provably safe group prefix — full-window groups
// writing 4 whole dst lanes. The remainder is decoded scalar here, and if
// the kernel deviates from the plan everything is redone scalar.
func decodeStreamVByteFull(control, data []byte, dst []uint32) error {
	if len(dst) == 0 {
		return nil
	}

	group, dataPos := 0, 0
	for group < len(control) && dataPos+16 <= len(data) && (group+1)*4 <= len(dst) {
		dataPos += int(streamVByteControlByteLengths[control[group]])
		group++
	}
	if group > 0 {
		_, consumed := varint.DecodeStreamVByte32Into(control[:group], data, dst[:group*4])
		if consumed != dataPos {
			group, dataPos = 0, 0
		}
	}

	dstPos := group * 4
	if dstPos >= len(dst) {
		return nil
	}
	for ; group < len(control) && dstPos < len(dst); group++ {
		ctrl := control[group]
		for lane := 0; lane < 4 && dstPos < len(dst); lane++ {
			size := int((ctrl>>(uint(lane)*2))&0x03) + 1
			if dataPos+size > len(data) {
				return fmt.Errorf("StreamVByte: truncated data at value %d of %d", dstPos, len(dst))
			}
			var v uint32
			for b := size - 1; b >= 0; b-- {
				v = v<<8 | uint32(data[dataPos+b])
			}
			dst[dstPos] = v
			dstPos++
			dataPos += size
		}
	}
	if dstPos < len(dst) {
		return fmt.Errorf("StreamVByte: decoded %d of %d values", dstPos, len(dst))
	}
	return nil
}

// SeparatedLocFormatMarker is a 2-byte marker indicating separated field ID encoding.
// The sequence 0xFF 0x00 can't be a valid varint because:
// - 0xFF means continuation (value=127, more bytes follow)
// - 0x00 means value=0 with no continuation
// - Total would be 127, but that should be encoded as single byte 0x7F
// So 0xFF 0x00 is invalid varint, making it a safe marker.
var SeparatedLocFormatMarker = []byte{0xFF, 0x00}

// streamVByteChunkedIntCoder encodes integers using StreamVByte within chunks.
// This is a drop-in replacement for chunkedIntCoder when UseStreamVByte is true.
type streamVByteChunkedIntCoder struct {
	final     []byte
	chunkSize uint64
	chunkBuf  bytes.Buffer
	chunkLens []uint64
	currChunk uint64

	// Current chunk's values (collected before encoding)
	chunkValues []uint32

	buf []byte

	// Reusable buffers for encoding to reduce allocations
	numBuf      []byte   // buffer for varint encoding (header + column lengths)
	controlBuf  []byte   // buffer for StreamVByte control bytes
	dataBuf     []byte   // buffer for StreamVByte data bytes
	deltaBuf    []uint32 // buffer for delta encoding (start deltas)
	endDeltaBuf []uint32 // buffer for delta encoding (end deltas)

	// Columnar encoding buffers (for UseColumnarLocations)
	colCounts    []uint32 // document value counts
	colFieldIDs  []uint32
	colPositions []uint32
	colStarts    []uint32
	colEnds      []uint32
	colNumAPs    []uint32
	colArrayPos  []uint32

	bytesWritten uint64
}

// newStreamVByteChunkedIntCoder returns a new StreamVByte chunk int coder
func newStreamVByteChunkedIntCoder(chunkSize uint64, maxDocNum uint64) *streamVByteChunkedIntCoder {
	total := maxDocNum/chunkSize + 1
	// Estimate final size: ~8 bytes per value (generous to avoid reallocation)
	// This is larger than needed but reduces memory management overhead
	estimatedFinalSize := int(total) * 256
	return &streamVByteChunkedIntCoder{
		chunkSize:    chunkSize,
		chunkLens:    make([]uint64, total),
		final:        make([]byte, 0, estimatedFinalSize),
		chunkValues:  make([]uint32, 0, 512), // larger initial capacity
		numBuf:       make([]byte, binary.MaxVarintLen64*3),
		controlBuf:   make([]byte, 0, 128),  // larger for bigger chunks
		dataBuf:      make([]byte, 0, 2048), // larger for bigger chunks
		deltaBuf:     make([]uint32, 0, 512),
		endDeltaBuf:  make([]uint32, 0, 512),
		colCounts:    make([]uint32, 0, 32),
		colFieldIDs:  make([]uint32, 0, 128),
		colPositions: make([]uint32, 0, 128),
		colStarts:    make([]uint32, 0, 128),
		colEnds:      make([]uint32, 0, 128),
		colNumAPs:    make([]uint32, 0, 128),
		colArrayPos:  make([]uint32, 0, 32),
	}
}

// Reset resets the coder for reuse
func (c *streamVByteChunkedIntCoder) Reset() {
	c.final = c.final[:0]
	c.bytesWritten = 0
	c.chunkBuf.Reset()
	c.currChunk = 0
	c.chunkValues = c.chunkValues[:0]
	for i := range c.chunkLens {
		c.chunkLens[i] = 0
	}
	// Reset columnar buffers
	c.colCounts = c.colCounts[:0]
	c.colFieldIDs = c.colFieldIDs[:0]
	c.colPositions = c.colPositions[:0]
	c.colStarts = c.colStarts[:0]
	c.colEnds = c.colEnds[:0]
	c.colNumAPs = c.colNumAPs[:0]
	c.colArrayPos = c.colArrayPos[:0]
}

// SetChunkSize changes the chunk size
func (c *streamVByteChunkedIntCoder) SetChunkSize(chunkSize uint64, maxDocNum uint64) {
	total := int(maxDocNum/chunkSize + 1)
	c.chunkSize = chunkSize
	if cap(c.chunkLens) < total {
		c.chunkLens = make([]uint64, total)
	} else {
		c.chunkLens = c.chunkLens[:total]
	}
}

func (c *streamVByteChunkedIntCoder) incrementBytesWritten(val uint64) {
	c.bytesWritten += val
}

func (c *streamVByteChunkedIntCoder) getBytesWritten() uint64 {
	return c.bytesWritten
}

// Add encodes the provided integers into the correct chunk
func (c *streamVByteChunkedIntCoder) Add(docNum uint64, vals ...uint64) error {
	chunk := docNum / c.chunkSize
	if chunk != c.currChunk {
		// Starting a new chunk - encode and flush the current one
		c.Close()
		c.chunkValues = c.chunkValues[:0]
		c.currChunk = chunk
	}

	// Collect values for StreamVByte encoding
	for _, val := range vals {
		c.chunkValues = append(c.chunkValues, uint32(val))
	}

	return nil
}

// Add1 encodes a single integer into the correct chunk (non-variadic to avoid slice allocation).
func (c *streamVByteChunkedIntCoder) Add1(docNum uint64, val uint64) error {
	chunk := docNum / c.chunkSize
	if chunk != c.currChunk {
		c.Close()
		c.chunkValues = c.chunkValues[:0]
		c.currChunk = chunk
	}

	c.chunkValues = append(c.chunkValues, uint32(val))
	return nil
}

// AddValues32 appends uint32 values directly without conversion.
// This is faster than Add() when values are already uint32.
func (c *streamVByteChunkedIntCoder) AddValues32(docNum uint64, vals []uint32) error {
	chunk := docNum / c.chunkSize
	if chunk != c.currChunk {
		c.Close()
		c.chunkValues = c.chunkValues[:0]
		c.currChunk = chunk
	}

	c.chunkValues = append(c.chunkValues, vals...)
	return nil
}

// AddBytes adds raw bytes to the current chunk (not StreamVByte encoded)
func (c *streamVByteChunkedIntCoder) AddBytes(docNum uint64, buf []byte) error {
	chunk := docNum / c.chunkSize
	if chunk != c.currChunk {
		c.Close()
		c.chunkValues = c.chunkValues[:0]
		c.currChunk = chunk
	}

	// For raw bytes, we fall back to direct storage
	// This is used in some edge cases
	_, err := c.chunkBuf.Write(buf)
	return err
}

// Close encodes and flushes the current chunk
func (c *streamVByteChunkedIntCoder) Close() {
	if len(c.chunkValues) == 0 && c.chunkBuf.Len() == 0 {
		c.chunkLens[c.currChunk] = 0
		return
	}

	// If we have raw bytes in chunkBuf (from AddBytes), use them directly
	if c.chunkBuf.Len() > 0 && len(c.chunkValues) == 0 {
		encodingBytes := c.chunkBuf.Bytes()
		c.incrementBytesWritten(uint64(len(encodingBytes)))
		c.chunkLens[c.currChunk] = uint64(len(encodingBytes))
		c.final = append(c.final, encodingBytes...)
		c.chunkBuf.Reset()
		c.currChunk = uint64(cap(c.chunkLens))
		return
	}

	// Use columnar encoding if enabled
	if UseColumnarLocations && len(c.chunkValues) > 0 {
		c.closeColumnar()
		return
	}

	// Determine format and values to encode
	format := ChunkFormatStreamVByte
	valuesToEncode := c.chunkValues

	// Apply delta encoding if enabled (naive version, usually not effective)
	if UseDeltaEncoding && len(c.chunkValues) > 0 {
		format = ChunkFormatStreamVByteDelta

		// Ensure delta buffer is large enough
		if cap(c.deltaBuf) < len(c.chunkValues) {
			c.deltaBuf = make([]uint32, len(c.chunkValues))
		} else {
			c.deltaBuf = c.deltaBuf[:len(c.chunkValues)]
		}

		// Delta encode: first value stays as-is, subsequent values are differences
		c.deltaBuf[0] = c.chunkValues[0]
		for i := 1; i < len(c.chunkValues); i++ {
			c.deltaBuf[i] = c.chunkValues[i] - c.chunkValues[i-1]
		}
		valuesToEncode = c.deltaBuf
	}

	// Encode values using StreamVByte (reuse buffers to avoid allocations)
	control, data := varint.EncodeStreamVByte32Into(valuesToEncode, c.controlBuf, c.dataBuf)
	c.controlBuf = control // update buffer reference in case it was grown
	c.dataBuf = data

	// Build chunk directly in final: [format] [numValues] [controlLen] [control] [data]
	// Calculate header sizes
	n1 := binary.PutUvarint(c.numBuf, uint64(len(c.chunkValues)))
	n2 := binary.PutUvarint(c.numBuf[n1:], uint64(len(control)))

	// Total chunk size: 1 (format) + n1 (numValues) + n2 (controlLen) + control + data
	chunkSize := 1 + n1 + n2 + len(control) + len(data)

	// Pre-grow final to avoid multiple reallocations
	startLen := len(c.final)
	if cap(c.final) < startLen+chunkSize {
		newFinal := make([]byte, startLen, startLen+chunkSize+1024)
		copy(newFinal, c.final)
		c.final = newFinal
	}

	// Write directly to final
	c.final = append(c.final, byte(format))
	c.final = append(c.final, c.numBuf[:n1+n2]...)
	c.final = append(c.final, control...)
	c.final = append(c.final, data...)

	c.incrementBytesWritten(uint64(chunkSize))
	c.chunkLens[c.currChunk] = uint64(chunkSize)
	c.chunkBuf.Reset()
	c.currChunk = uint64(cap(c.chunkLens))
}

// closeColumnar encodes the chunk using columnar format with delta encoding
// for monotonically increasing start/end byte offsets.
//
// Location data format in chunkValues:
//
//	[count1] [fieldID, pos, start, end, numAP, (arrayPos...)]* [count2] ...
//
// Columnar output format:
//
//	[format=0x03] [numDocs] [numLocs] [numArrayPos]
//	[counts column] [fieldIDs column] [positions column]
//	[starts column (delta)] [ends column (delta)] [numAPs column] [arrayPos column]
func (c *streamVByteChunkedIntCoder) closeColumnar() {
	// Reset columnar buffers
	c.colCounts = c.colCounts[:0]
	c.colFieldIDs = c.colFieldIDs[:0]
	c.colPositions = c.colPositions[:0]
	c.colStarts = c.colStarts[:0]
	c.colEnds = c.colEnds[:0]
	c.colNumAPs = c.colNumAPs[:0]
	c.colArrayPos = c.colArrayPos[:0]

	// Parse location data into columns.
	// Format: [count] [fieldID, pos, start, end, numAP, (arrayPos...)]* repeated per doc.
	// Count is trusted only up to the current chunk boundary. Older buggy merge
	// paths could write a byte count into StreamVByte chunks; malformed prefixes
	// must not panic background scorch merge workers.
	idx := 0
	for idx < len(c.chunkValues) {
		count := int(c.chunkValues[idx])
		idx++

		endIdx := idx + count
		if endIdx > len(c.chunkValues) {
			endIdx = len(c.chunkValues)
		}

		valuesWritten := 0
		for idx+5 <= endIdx {
			c.colFieldIDs = append(c.colFieldIDs, c.chunkValues[idx])
			c.colPositions = append(c.colPositions, c.chunkValues[idx+1])
			c.colStarts = append(c.colStarts, c.chunkValues[idx+2])
			c.colEnds = append(c.colEnds, c.chunkValues[idx+3])
			numAP := c.chunkValues[idx+4]
			if int(numAP) > endIdx-(idx+5) {
				numAP = uint32(endIdx - (idx + 5))
			}
			c.colNumAPs = append(c.colNumAPs, numAP)
			idx += 5

			// Read array positions
			for j := 0; j < int(numAP); j++ {
				c.colArrayPos = append(c.colArrayPos, c.chunkValues[idx])
				idx++
			}
			valuesWritten += 5 + int(numAP)
		}
		if idx < endIdx {
			idx = endIdx
		}
		c.colCounts = append(c.colCounts, uint32(valuesWritten))
	}

	numDocs := len(c.colCounts)
	numLocs := len(c.colFieldIDs)
	numArrayPos := len(c.colArrayPos)

	// Delta encode starts (monotonically increasing byte offsets)
	if cap(c.deltaBuf) < numLocs {
		c.deltaBuf = make([]uint32, numLocs)
	}
	startDeltas := c.deltaBuf[:numLocs]
	if numLocs > 0 {
		startDeltas[0] = c.colStarts[0]
		for i := 1; i < numLocs; i++ {
			startDeltas[i] = c.colStarts[i] - c.colStarts[i-1]
		}
	}

	// Delta encode ends (monotonically increasing byte offsets)
	if cap(c.endDeltaBuf) < numLocs {
		c.endDeltaBuf = make([]uint32, numLocs)
	}
	endDeltas := c.endDeltaBuf[:numLocs]
	if numLocs > 0 {
		endDeltas[0] = c.colEnds[0]
		for i := 1; i < numLocs; i++ {
			endDeltas[i] = c.colEnds[i] - c.colEnds[i-1]
		}
	}

	// Encode each column with StreamVByte, writing directly to final
	startLen := len(c.final)

	// Write format byte
	c.final = append(c.final, ChunkFormatColumnar)

	// Write header: numDocs, numLocs, numArrayPos
	n := binary.PutUvarint(c.numBuf, uint64(numDocs))
	n += binary.PutUvarint(c.numBuf[n:], uint64(numLocs))
	n += binary.PutUvarint(c.numBuf[n:], uint64(numArrayPos))
	c.final = append(c.final, c.numBuf[:n]...)

	// Helper to encode and append a column directly to final
	writeColumn := func(values []uint32) {
		if len(values) == 0 {
			// Write 0 control length for empty column
			c.final = append(c.final, 0)
			return
		}
		ctrl, data := varint.EncodeStreamVByte32Into(values, c.controlBuf, c.dataBuf)
		c.controlBuf = ctrl
		c.dataBuf = data

		// Append controlLen + control + data
		ln := binary.PutUvarint(c.numBuf, uint64(len(ctrl)))
		c.final = append(c.final, c.numBuf[:ln]...)
		c.final = append(c.final, ctrl...)
		c.final = append(c.final, data...)
	}

	// Write columns in order
	writeColumn(c.colCounts)
	writeColumn(c.colFieldIDs)
	writeColumn(c.colPositions)
	writeColumn(startDeltas)
	writeColumn(endDeltas)
	writeColumn(c.colNumAPs)
	if numArrayPos > 0 {
		writeColumn(c.colArrayPos)
	}

	chunkSize := len(c.final) - startLen
	c.incrementBytesWritten(uint64(chunkSize))
	c.chunkLens[c.currChunk] = uint64(chunkSize)
	c.currChunk = uint64(cap(c.chunkLens))
}

// Write commits all encoded chunks to the provided writer
func (c *streamVByteChunkedIntCoder) Write(w io.Writer) (int, error) {
	bufNeeded := binary.MaxVarintLen64 * (1 + len(c.chunkLens))
	if len(c.buf) < bufNeeded {
		c.buf = make([]byte, bufNeeded)
	}
	buf := c.buf

	// Convert chunk lengths to offsets
	chunkOffsets := modifyLengthsToEndOffsets(c.chunkLens)

	// Write number of chunks and each offset
	n := binary.PutUvarint(buf, uint64(len(chunkOffsets)))
	for _, chunkOffset := range chunkOffsets {
		n += binary.PutUvarint(buf[n:], chunkOffset)
	}

	tw, err := w.Write(buf[:n])
	if err != nil {
		return tw, err
	}

	// Write data
	nw, err := w.Write(c.final)
	tw += nw
	return tw, err
}

// writeAt commits all encoded chunks and returns start offset
func (c *streamVByteChunkedIntCoder) writeAt(w io.Writer) (uint64, int, error) {
	startOffset := uint64(termNotEncoded)
	if len(c.final) <= 0 {
		return startOffset, 0, nil
	}

	if chw := w.(*CountHashWriter); chw != nil {
		startOffset = uint64(chw.Count())
	}

	tw, err := c.Write(w)
	return startOffset, tw, err
}

// FinalSize returns the size of the final encoded data
func (c *streamVByteChunkedIntCoder) FinalSize() int {
	return len(c.final)
}

// =============================================================================
// StreamVByte Chunked Int Decoder
// =============================================================================

// streamVByteChunkedIntDecoder decodes StreamVByte-encoded chunks
type streamVByteChunkedIntDecoder struct {
	startOffset     uint64
	dataStartOffset uint64
	chunkOffsets    []uint64
	curChunkBytes   []byte
	data            []byte

	// Decoded values from current chunk
	values []uint32
	pos    int
	format byte

	// For StreamVByte byte position tracking (enables copying optimization)
	headerLen  int    // Length of format + numValues + controlLen prefix
	controlLen int    // Length of control bytes
	control    []byte // Control bytes (for computing byte positions)

	// Reusable buffers for columnar decoding to reduce allocations
	decCounts    []uint32
	decFieldIDs  []uint32
	decPositions []uint32
	decStarts    []uint32
	decEnds      []uint32
	decNumAPs    []uint32
	decArrayPos  []uint32

	// Fallback varint reader for legacy chunks
	r *memUvarintReader

	bytesRead uint64
}

// newStreamVByteChunkedIntDecoder creates a new decoder
func newStreamVByteChunkedIntDecoder(buf []byte, offset uint64, rv *streamVByteChunkedIntDecoder) *streamVByteChunkedIntDecoder {
	if rv == nil {
		rv = &streamVByteChunkedIntDecoder{startOffset: offset, data: buf}
	} else {
		rv.startOffset = offset
		rv.data = buf
	}

	var n, numChunks uint64
	var read int
	if offset == termNotEncoded {
		numChunks = 0
	} else {
		numChunks, read = binary.Uvarint(buf[offset+n : offset+n+binary.MaxVarintLen64])
	}

	n += uint64(read)
	if cap(rv.chunkOffsets) >= int(numChunks) {
		rv.chunkOffsets = rv.chunkOffsets[:int(numChunks)]
	} else {
		rv.chunkOffsets = make([]uint64, int(numChunks))
	}
	for i := 0; i < int(numChunks); i++ {
		rv.chunkOffsets[i], read = binary.Uvarint(buf[offset+n : offset+n+binary.MaxVarintLen64])
		n += uint64(read)
	}
	rv.bytesRead += n
	rv.dataStartOffset = offset + n
	return rv
}

func (d *streamVByteChunkedIntDecoder) getBytesRead() uint64 {
	return d.bytesRead
}

func (d *streamVByteChunkedIntDecoder) loadChunk(chunk int) error {
	if d.startOffset == termNotEncoded {
		d.values = d.values[:0]
		d.pos = 0
		return nil
	}

	if chunk >= len(d.chunkOffsets) {
		return fmt.Errorf("tried to load chunk that doesn't exist %d/(%d)",
			chunk, len(d.chunkOffsets))
	}

	end, start := d.dataStartOffset, d.dataStartOffset
	s, e := readChunkBoundary(chunk, d.chunkOffsets)
	start += s
	end += e
	d.curChunkBytes = d.data[start:end]
	d.bytesRead += uint64(len(d.curChunkBytes))

	if len(d.curChunkBytes) == 0 {
		d.values = d.values[:0]
		d.pos = 0
		return nil
	}

	// Check format byte
	d.format = d.curChunkBytes[0]

	if d.format == ChunkFormatColumnar {
		// Columnar format with delta encoding for starts/ends
		return d.loadChunkColumnar()
	} else if d.format == ChunkFormatStreamVByte || d.format == ChunkFormatStreamVByteDelta {
		// StreamVByte format (with or without delta encoding)
		offset := 1

		// Read number of values
		numValues, n := binary.Uvarint(d.curChunkBytes[offset:])
		if n <= 0 {
			return fmt.Errorf("invalid StreamVByte chunk: can't read numValues")
		}
		offset += n

		// Read control length
		controlLen, n := binary.Uvarint(d.curChunkBytes[offset:])
		if n <= 0 {
			return fmt.Errorf("invalid StreamVByte chunk: can't read controlLen")
		}
		offset += n

		// Save header info for byte position tracking
		d.headerLen = offset
		d.controlLen = int(controlLen)
		d.control = d.curChunkBytes[offset : offset+int(controlLen)]

		// Extract data bytes
		dataBytes := d.curChunkBytes[offset+int(controlLen):]

		// Decode all values at once using SIMD
		if cap(d.values) >= int(numValues) {
			d.values = d.values[:numValues]
		} else {
			d.values = make([]uint32, numValues)
		}
		if err := decodeStreamVByteFull(d.control, dataBytes, d.values); err != nil {
			return err
		}

		// Apply delta decoding if this chunk was delta encoded
		if d.format == ChunkFormatStreamVByteDelta && len(d.values) > 1 {
			// Delta decode: first value stays, subsequent values are cumulative sums
			for i := 1; i < len(d.values); i++ {
				d.values[i] = d.values[i-1] + d.values[i]
			}
		}

		d.pos = 0
	} else {
		// Legacy varint format - use memUvarintReader
		if d.r == nil {
			d.r = newMemUvarintReader(d.curChunkBytes)
		} else {
			d.r.Reset(d.curChunkBytes)
		}
		d.values = d.values[:0]
		d.pos = 0
	}

	return nil
}

// loadChunkColumnar decodes a columnar format chunk and reconstructs the
// interleaved location data format for compatibility with existing code.
//
// Columnar format:
//
//	[format=0x03] [numDocs] [numLocs] [numArrayPos]
//	[counts column] [fieldIDs column] [positions column]
//	[starts column (delta)] [ends column (delta)] [numAPs column] [arrayPos column]
func (d *streamVByteChunkedIntDecoder) loadChunkColumnar() error {
	offset := 1

	// Read header: numDocs, numLocs, numArrayPos
	numDocs, n := binary.Uvarint(d.curChunkBytes[offset:])
	if n <= 0 {
		return fmt.Errorf("invalid columnar chunk: can't read numDocs")
	}
	offset += n

	numLocs, n := binary.Uvarint(d.curChunkBytes[offset:])
	if n <= 0 {
		return fmt.Errorf("invalid columnar chunk: can't read numLocs")
	}
	offset += n

	numArrayPos, n := binary.Uvarint(d.curChunkBytes[offset:])
	if n <= 0 {
		return fmt.Errorf("invalid columnar chunk: can't read numArrayPos")
	}
	offset += n

	// Helper to read and decode a column into a reusable buffer
	readColumnInto := func(numValues int, buf []uint32) ([]uint32, error) {
		ctrlLen, n := binary.Uvarint(d.curChunkBytes[offset:])
		if n <= 0 {
			return nil, fmt.Errorf("invalid columnar chunk: can't read control length")
		}
		offset += n

		if ctrlLen == 0 {
			return nil, nil // empty column
		}

		ctrl := d.curChunkBytes[offset : offset+int(ctrlLen)]
		offset += int(ctrlLen)

		dataLen := 0
		for _, ctrlByte := range ctrl {
			dataLen += int(streamVByteControlByteLengths[ctrlByte])
		}

		dataBytes := d.curChunkBytes[offset : offset+dataLen]
		offset += dataLen

		if cap(buf) >= numValues {
			buf = buf[:numValues]
		} else {
			buf = make([]uint32, numValues)
		}
		if err := decodeStreamVByteFull(ctrl, dataBytes, buf); err != nil {
			return nil, err
		}
		return buf, nil
	}

	// Read all columns into reusable decoder buffers
	counts, err := readColumnInto(int(numDocs), d.decCounts)
	if err != nil {
		return err
	}
	d.decCounts = counts

	fieldIDs, err := readColumnInto(int(numLocs), d.decFieldIDs)
	if err != nil {
		return err
	}
	d.decFieldIDs = fieldIDs

	positions, err := readColumnInto(int(numLocs), d.decPositions)
	if err != nil {
		return err
	}
	d.decPositions = positions

	startDeltas, err := readColumnInto(int(numLocs), d.decStarts)
	if err != nil {
		return err
	}
	d.decStarts = startDeltas

	endDeltas, err := readColumnInto(int(numLocs), d.decEnds)
	if err != nil {
		return err
	}
	d.decEnds = endDeltas

	numAPs, err := readColumnInto(int(numLocs), d.decNumAPs)
	if err != nil {
		return err
	}
	d.decNumAPs = numAPs

	var arrayPositions []uint32
	if numArrayPos > 0 {
		arrayPositions, err = readColumnInto(int(numArrayPos), d.decArrayPos)
		if err != nil {
			return err
		}
		d.decArrayPos = arrayPositions
	}

	// Delta decode starts (in-place)
	starts := startDeltas
	if numLocs > 0 {
		for i := 1; i < int(numLocs); i++ {
			starts[i] = starts[i-1] + starts[i]
		}
	}

	// Delta decode ends (in-place)
	ends := endDeltas
	if numLocs > 0 {
		for i := 1; i < int(numLocs); i++ {
			ends[i] = ends[i-1] + ends[i]
		}
	}

	// Reconstruct interleaved format: [count] [fieldID, pos, start, end, numAP, (arrayPos...)]* per doc
	// Calculate total size needed
	totalValues := int(numDocs) + int(numLocs)*5 + int(numArrayPos)
	if cap(d.values) >= totalValues {
		d.values = d.values[:totalValues]
	} else {
		d.values = make([]uint32, totalValues)
	}

	// Reconstruct
	valIdx := 0
	locIdx := 0
	apIdx := 0
	for docIdx := 0; docIdx < int(numDocs); docIdx++ {
		count := counts[docIdx]
		d.values[valIdx] = count
		valIdx++

		// Calculate how many locations this document has
		// We need to figure out how many locations are covered by this count
		// The count is the number of VALUES (not locations)
		// Each location = 5 + numAP values
		remaining := int(count)
		for remaining > 0 && locIdx < int(numLocs) {
			d.values[valIdx] = fieldIDs[locIdx]
			d.values[valIdx+1] = positions[locIdx]
			d.values[valIdx+2] = starts[locIdx]
			d.values[valIdx+3] = ends[locIdx]
			d.values[valIdx+4] = numAPs[locIdx]
			valIdx += 5

			numAP := int(numAPs[locIdx])
			for j := 0; j < numAP && apIdx < int(numArrayPos); j++ {
				d.values[valIdx] = arrayPositions[apIdx]
				valIdx++
				apIdx++
			}

			remaining -= 5 + numAP
			locIdx++
		}
	}

	// Trim values slice to actual size written
	d.values = d.values[:valIdx]
	d.pos = 0

	return nil
}

func (d *streamVByteChunkedIntDecoder) reset() {
	d.startOffset = 0
	d.dataStartOffset = 0
	d.chunkOffsets = d.chunkOffsets[:0]
	d.curChunkBytes = d.curChunkBytes[:0]
	d.bytesRead = 0
	d.data = d.data[:0]
	d.values = d.values[:0]
	d.pos = 0
	if d.r != nil {
		d.r.Reset([]byte(nil))
	}
}

func (d *streamVByteChunkedIntDecoder) isNil() bool {
	return d.curChunkBytes == nil || len(d.curChunkBytes) == 0
}

// isStreamVByteFormat returns true if the format is StreamVByte (any variant)
func (d *streamVByteChunkedIntDecoder) isStreamVByteFormat() bool {
	return d.format == ChunkFormatStreamVByte ||
		d.format == ChunkFormatStreamVByteDelta ||
		d.format == ChunkFormatColumnar
}

func (d *streamVByteChunkedIntDecoder) readUvarint() (uint64, error) {
	if d.isStreamVByteFormat() {
		if d.pos >= len(d.values) {
			return 0, io.EOF
		}
		val := d.values[d.pos]
		d.pos++
		return uint64(val), nil
	}
	// Fallback to legacy varint reader
	if d.r == nil {
		return 0, io.EOF
	}
	return d.r.ReadUvarint()
}

// readValues32 reads n values and returns them as a uint32 slice.
// For StreamVByte, this returns a slice of the internal decoded buffer (no copy).
// For varint, this decodes values into the provided buffer.
func (d *streamVByteChunkedIntDecoder) readValues32(n int, buf []uint32) ([]uint32, error) {
	if d.isStreamVByteFormat() {
		if d.pos+n > len(d.values) {
			return nil, io.EOF
		}
		result := d.values[d.pos : d.pos+n]
		d.pos += n
		return result, nil
	}
	// Fallback: decode into buffer
	if cap(buf) < n {
		buf = make([]uint32, n)
	} else {
		buf = buf[:n]
	}
	for i := 0; i < n; i++ {
		val, err := d.r.ReadUvarint()
		if err != nil {
			return nil, err
		}
		buf[i] = uint32(val)
	}
	return buf, nil
}

func (d *streamVByteChunkedIntDecoder) readBytes(start, end int) []byte {
	return d.curChunkBytes[start:end]
}

func (d *streamVByteChunkedIntDecoder) SkipUvarint() {
	if d.isStreamVByteFormat() {
		d.pos++
	} else if d.r != nil {
		d.r.SkipUvarint()
	}
}

func (d *streamVByteChunkedIntDecoder) SkipBytes(count int) {
	if d.isStreamVByteFormat() {
		// For StreamVByte, count is actually a VALUE count (not byte count)
		// because the writer stores value count for StreamVByte format
		d.pos += count
	} else if d.r != nil {
		d.r.SkipBytes(count)
	}
}

func (d *streamVByteChunkedIntDecoder) Len() int {
	if d.isStreamVByteFormat() {
		return len(d.values) - d.pos
	}
	if d.r == nil {
		return 0
	}
	return d.r.Len()
}

func (d *streamVByteChunkedIntDecoder) remainingLen() int {
	if d.isStreamVByteFormat() {
		// Compute byte position by walking control bytes
		return d.bytePositionForValue(d.pos)
	}
	if d.r == nil {
		return len(d.curChunkBytes)
	}
	return len(d.curChunkBytes) - d.r.Len()
}

// bytePositionForValue computes the byte offset in curChunkBytes for value index.
// This enables the byte-copying optimization for StreamVByte.
// Note: O(valueIdx) complexity since it walks all control bytes up to the target.
// Only called via remainingLen() in the legacy byte-copying path (guarded by !UseStreamVByte).
func (d *streamVByteChunkedIntDecoder) bytePositionForValue(valueIdx int) int {
	if !d.isStreamVByteFormat() || len(d.control) == 0 {
		return 0
	}

	// Count control bytes consumed: each control byte handles 4 values
	controlBytesConsumed := valueIdx / 4

	// Count data bytes consumed by walking control bytes
	dataBytesConsumed := 0
	for i := 0; i < valueIdx; i++ {
		controlIdx := i / 4
		if controlIdx >= len(d.control) {
			break
		}
		shift := (i % 4) * 2
		size := int((d.control[controlIdx]>>shift)&0x03) + 1
		dataBytesConsumed += size
	}

	// Total position = header + control bytes consumed + data bytes consumed
	return d.headerLen + controlBytesConsumed + dataBytesConsumed
}
