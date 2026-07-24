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
	"testing"

	"github.com/RoaringBitmap/roaring/v2"
	"github.com/ajroetker/go-highway/hwy/contrib/varint"
	seg "github.com/blevesearch/scorch_segment_api/v2"
)

// TestStreamVByteChunkedIntCoderRoundTrip tests encoding and decoding
func TestStreamVByteChunkedIntCoderRoundTrip(t *testing.T) {
	// Simulate location data: fieldID, pos, start, end, numArrayPos, [arrayPos...]
	testData := [][]uint64{
		{0, 1, 0, 5, 0},            // Simple location, no array positions
		{0, 2, 5, 10, 2, 0, 1},     // Location with 2 array positions
		{1, 3, 10, 20, 3, 0, 1, 2}, // Different field, 3 array positions
		{0, 100, 500, 600, 1, 5},   // Larger offsets
		{2, 1000, 5000, 6000, 0},   // Even larger offsets
	}

	coder := newStreamVByteChunkedIntCoder(1024, 100)

	// Compute total values count for the count prefix
	// Format expected by columnar encoding: [count] [values...]
	var totalValues int
	for _, locData := range testData {
		totalValues += len(locData)
	}

	// Add count prefix first (this is what merge path does with Add1)
	err := coder.Add(0, uint64(totalValues))
	if err != nil {
		t.Fatalf("Add count failed: %v", err)
	}

	// Add all data as if from document 0
	for _, locData := range testData {
		err = coder.Add(0, locData...)
		if err != nil {
			t.Fatalf("Add failed: %v", err)
		}
	}
	coder.Close()

	// Write to buffer
	var buf bytes.Buffer
	_, err = coder.Write(&buf)
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	// Build a buffer with the data at a non-zero offset
	// (offset=0 means "not encoded" in zapx)
	bufBytes := buf.Bytes()
	const testOffset = 8 // Use non-zero offset to avoid termNotEncoded sentinel
	fullBuf := make([]byte, testOffset+len(bufBytes))
	copy(fullBuf[testOffset:], bufBytes)

	// Create decoder with the non-zero offset
	decoder := newStreamVByteChunkedIntDecoder(fullBuf, testOffset, nil)

	err = decoder.loadChunk(0)
	if err != nil {
		t.Fatalf("loadChunk failed: %v", err)
	}

	// Verify format is StreamVByte (any variant including columnar)
	if decoder.format != ChunkFormatStreamVByte && decoder.format != ChunkFormatStreamVByteDelta && decoder.format != ChunkFormatColumnar {
		t.Fatalf("Expected StreamVByte format, got %d", decoder.format)
	}

	// Read the count prefix first
	gotCount, err := decoder.readUvarint()
	if err != nil {
		t.Fatalf("readUvarint for count failed: %v", err)
	}
	if gotCount != uint64(totalValues) {
		t.Errorf("Count mismatch: got %d, want %d", gotCount, totalValues)
	}

	// Read back and verify location data
	for i, locData := range testData {
		for j, expected := range locData {
			got, err := decoder.readUvarint()
			if err != nil {
				t.Fatalf("readUvarint failed at loc %d, field %d: %v", i, j, err)
			}
			if got != expected {
				t.Errorf("Mismatch at loc %d, field %d: got %d, want %d", i, j, got, expected)
			}
		}
	}
}

func TestStreamVByteColumnarLocationCountClampedToChunk(t *testing.T) {
	origUseColumnar := UseColumnarLocations
	UseColumnarLocations = true
	defer func() {
		UseColumnarLocations = origUseColumnar
	}()

	coder := newStreamVByteChunkedIntCoder(1024, 100)

	// Simulate an old malformed merge output where the per-document location
	// prefix was a byte count instead of the number of StreamVByte values.
	if err := coder.Add1(0, 100); err != nil {
		t.Fatalf("Add1 count failed: %v", err)
	}
	if err := coder.Add(0, 0, 1, 10, 20, 0); err != nil {
		t.Fatalf("Add location failed: %v", err)
	}

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Close panicked with oversized location count: %v", r)
			}
		}()
		coder.Close()
	}()

	var buf bytes.Buffer
	if _, err := coder.Write(&buf); err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	const testOffset = 8
	fullBuf := make([]byte, testOffset+buf.Len())
	copy(fullBuf[testOffset:], buf.Bytes())

	decoder := newStreamVByteChunkedIntDecoder(fullBuf, testOffset, nil)
	if err := decoder.loadChunk(0); err != nil {
		t.Fatalf("loadChunk failed: %v", err)
	}

	gotCount, err := decoder.readUvarint()
	if err != nil {
		t.Fatalf("read clamped count failed: %v", err)
	}
	if gotCount != 5 {
		t.Fatalf("decoded location count = %d, want 5", gotCount)
	}

	want := []uint64{0, 1, 10, 20, 0}
	for i, expected := range want {
		got, err := decoder.readUvarint()
		if err != nil {
			t.Fatalf("read location value %d failed: %v", i, err)
		}
		if got != expected {
			t.Fatalf("location value %d = %d, want %d", i, got, expected)
		}
	}
}

// BenchmarkLocationDecode_Varint benchmarks legacy varint decoding
func BenchmarkLocationDecode_Varint(b *testing.B) {
	// Generate typical location data
	numLocs := 100
	locData := generateLocationData(numLocs)

	// Encode using legacy varint
	encoded := encodeLocationsVarint(locData)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		decodeLocationsVarint(encoded, numLocs)
	}
}

// BenchmarkLocationDecode_StreamVByte benchmarks StreamVByte decoding
func BenchmarkLocationDecode_StreamVByte(b *testing.B) {
	// Generate typical location data
	numLocs := 100
	locData := generateLocationData(numLocs)

	// Flatten to uint32 slice
	var values []uint32
	for _, loc := range locData {
		for _, v := range loc {
			values = append(values, uint32(v))
		}
	}

	// Encode using StreamVByte
	control, data := varint.EncodeStreamVByte32(values)

	// Pre-allocate decode buffer
	decoded := make([]uint32, len(values))

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		varint.DecodeStreamVByte32Into(control, data, decoded)
	}
}

// BenchmarkLocationDecode_Comparison runs both for direct comparison
func BenchmarkLocationDecode_Comparison(b *testing.B) {
	numLocs := 100
	locData := generateLocationData(numLocs)

	b.Run("Varint", func(b *testing.B) {
		encoded := encodeLocationsVarint(locData)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			decodeLocationsVarint(encoded, numLocs)
		}
	})

	b.Run("StreamVByte", func(b *testing.B) {
		var values []uint32
		for _, loc := range locData {
			for _, v := range loc {
				values = append(values, uint32(v))
			}
		}
		control, data := varint.EncodeStreamVByte32(values)
		decoded := make([]uint32, len(values))

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			varint.DecodeStreamVByte32Into(control, data, decoded)
		}
	})
}

// generateLocationData generates realistic location data
// Each location: fieldID, pos, start, end, numArrayPos, [arrayPos...]
func generateLocationData(numLocs int) [][]uint64 {
	locData := make([][]uint64, numLocs)
	for i := 0; i < numLocs; i++ {
		// Simulate typical location patterns
		fieldID := uint64(i % 3)      // 0-2 fields
		pos := uint64(i + 1)          // Position 1-N
		start := uint64(i * 10)       // Start offset
		end := uint64(i*10 + 5 + i%5) // End offset
		numArrayPos := i % 4          // 0-3 array positions

		loc := []uint64{fieldID, pos, start, end, uint64(numArrayPos)}
		for j := 0; j < numArrayPos; j++ {
			loc = append(loc, uint64(j))
		}
		locData[i] = loc
	}
	return locData
}

// encodeLocationsVarint encodes locations using legacy varint
func encodeLocationsVarint(locData [][]uint64) []byte {
	var buf bytes.Buffer
	tmp := make([]byte, binary.MaxVarintLen64)

	for _, loc := range locData {
		for _, v := range loc {
			n := binary.PutUvarint(tmp, v)
			buf.Write(tmp[:n])
		}
	}
	return buf.Bytes()
}

// decodeLocationsVarint decodes locations using legacy varint
func decodeLocationsVarint(data []byte, numLocs int) [][]uint64 {
	result := make([][]uint64, 0, numLocs)
	pos := 0

	for i := 0; i < numLocs && pos < len(data); i++ {
		// Read fixed fields
		fieldID, n := binary.Uvarint(data[pos:])
		pos += n
		p, n := binary.Uvarint(data[pos:])
		pos += n
		start, n := binary.Uvarint(data[pos:])
		pos += n
		end, n := binary.Uvarint(data[pos:])
		pos += n
		numArrayPos, n := binary.Uvarint(data[pos:])
		pos += n

		loc := []uint64{fieldID, p, start, end, numArrayPos}
		for j := 0; j < int(numArrayPos); j++ {
			ap, n := binary.Uvarint(data[pos:])
			pos += n
			loc = append(loc, ap)
		}
		result = append(result, loc)
	}
	return result
}

// TestMergeFormatUpgrade verifies that segments created with old location format
// are correctly upgraded to the new columnar delta format during merge.
func TestMergeFormatUpgrade(t *testing.T) {
	// Save original settings
	origUseStreamVByte := UseStreamVByte
	origUseColumnar := UseColumnarLocations
	defer func() {
		UseStreamVByte = origUseStreamVByte
		UseColumnarLocations = origUseColumnar
	}()

	tmpDir := t.TempDir()

	// Step 1: Create segments with OLD format (row-oriented StreamVByte)
	UseStreamVByte = true
	UseColumnarLocations = false

	seg1Path := tmpDir + "/seg1.zap"
	seg2Path := tmpDir + "/seg2.zap"

	// Build first segment
	testSeg1, _, _ := buildTestSegmentMulti()
	err := PersistSegmentBase(testSeg1, seg1Path)
	if err != nil {
		t.Fatalf("persist seg1: %v", err)
	}

	// Build second segment
	testSeg2, _, _ := buildTestSegmentMulti2()
	err = PersistSegmentBase(testSeg2, seg2Path)
	if err != nil {
		t.Fatalf("persist seg2: %v", err)
	}

	// Open segments
	segment1, err := zapPlugin.Open(seg1Path)
	if err != nil {
		t.Fatalf("open seg1: %v", err)
	}
	defer segment1.Close()

	segment2, err := zapPlugin.Open(seg2Path)
	if err != nil {
		t.Fatalf("open seg2: %v", err)
	}
	defer segment2.Close()

	// Step 2: Merge with NEW format enabled (columnar delta)
	UseColumnarLocations = true

	mergedPath := tmpDir + "/merged.zap"
	segsToMerge := []seg.Segment{segment1, segment2}
	drops := []*roaring.Bitmap{nil, nil}

	_, _, err = zapPlugin.Merge(segsToMerge, drops, mergedPath, nil, nil)
	if err != nil {
		t.Fatalf("merge: %v", err)
	}

	// Step 3: Open merged segment and verify location data is readable
	mergedSeg, err := zapPlugin.Open(mergedPath)
	if err != nil {
		t.Fatalf("open merged: %v", err)
	}
	defer mergedSeg.Close()

	merged := mergedSeg.(*Segment)

	// Verify document count
	if merged.Count() != 4 {
		t.Errorf("expected 4 docs, got %d", merged.Count())
	}

	// Verify we can read location data from postings
	// This exercises the decoder with the new format
	dict, err := merged.Dictionary("name")
	if err != nil {
		t.Fatalf("get dictionary: %v", err)
	}

	// Iterate through all terms and verify location data is readable
	dictIter := dict.AutomatonIterator(nil, nil, nil)
	for {
		entry, err := dictIter.Next()
		if err != nil {
			t.Fatalf("dict iter: %v", err)
		}
		if entry == nil {
			break
		}

		// Get postings list with locations
		plist, err := dict.(*Dictionary).postingsList([]byte(entry.Term), nil, nil)
		if err != nil {
			t.Fatalf("postings list for %s: %v", entry.Term, err)
		}

		// Iterate through postings and read locations
		pitr := plist.Iterator(true, true, true, nil)
		for {
			posting, err := pitr.Next()
			if err != nil {
				t.Fatalf("posting iter: %v", err)
			}
			if posting == nil {
				break
			}

			// Get locations - this verifies the decoder works with the new format
			locs := posting.Locations()
			for _, loc := range locs {
				// Verify location data is sensible
				if loc.Start() > loc.End() {
					t.Errorf("invalid location: start %d > end %d", loc.Start(), loc.End())
				}
			}
		}
	}

	t.Logf("Format upgrade test passed: merged segment has %d docs with readable locations", merged.Count())
}

// TestDecodeStreamVByteFullTailCompletion exercises the ≤15-byte data tails
// the SIMD batch kernel leaves undecoded (it stops at the last full 16-byte
// load window). Sizes cover: below one SIMD window, exact group boundaries,
// maximal tails, and multi-window inputs; widths cover 1-4 byte values.
func TestDecodeStreamVByteFullTailCompletion(t *testing.T) {
	widths := map[string]func(i int) uint32{
		"1byte": func(i int) uint32 { return uint32(i % 250) },
		"2byte": func(i int) uint32 { return uint32(260 + i) },
		"4byte": func(i int) uint32 { return uint32(0x01000000 + i*7919) },
		"mixed": func(i int) uint32 { return uint32(i) << (uint(i%4) * 8) },
	}
	for name, gen := range widths {
		for _, n := range []int{1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 100, 1000} {
			values := make([]uint32, n)
			for i := range values {
				values[i] = gen(i)
			}
			control, data := varint.EncodeStreamVByte32Into(values, nil, nil)
			got := make([]uint32, n)
			if err := decodeStreamVByteFull(control, data, got); err != nil {
				t.Fatalf("%s/n=%d: decode error: %v", name, n, err)
			}
			for i := range values {
				if got[i] != values[i] {
					t.Fatalf("%s/n=%d: value %d mismatch: got %d, want %d", name, n, i, got[i], values[i])
				}
			}
		}
	}

	// Truncated data must error, not silently zero-fill.
	values := []uint32{300, 300, 300, 300, 300}
	control, data := varint.EncodeStreamVByte32Into(values, nil, nil)
	got := make([]uint32, len(values))
	// last group = one 2-byte value + three 1-byte zero pads; cut past the
	// padding into the value's bytes so real data is actually missing
	if err := decodeStreamVByteFull(control, data[:len(data)-4], got); err == nil {
		t.Fatal("expected error for truncated data, got nil")
	}
}
