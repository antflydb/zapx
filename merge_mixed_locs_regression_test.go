//  Copyright (c) 2026 Couchbase, Inc.
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
	"path/filepath"
	"strings"
	"testing"

	"github.com/RoaringBitmap/roaring/v2"
	index "github.com/blevesearch/bleve_index_api"
	seg "github.com/blevesearch/scorch_segment_api/v2"
)

// newStubFieldNoVectors returns an indexed field whose token frequencies
// carry no locations, producing hasLocs=false postings.
func newStubFieldNoVectors(name, value string) *stubField {
	freqs := make(index.TokenFrequencies)
	for _, tok := range strings.Split(value, " ") {
		if curr, exists := freqs[tok]; exists {
			curr.SetFrequency(curr.Frequency() + 1)
			continue
		}
		tf := &index.TokenFreq{Term: []byte(tok)}
		tf.SetFrequency(1)
		freqs[tok] = tf
	}
	return &stubField{
		name:          name,
		value:         []byte(value),
		encodedType:   't',
		options:       index.IndexField | index.StoreField,
		analyzedLen:   len(freqs),
		analyzedFreqs: freqs,
	}
}

// TestMergeMixedHasLocsPostings pins the merge-mode stale location count bug:
// a postings list interleaving hasLocs=true and hasLocs=false postings, with
// another hasLocs=true posting after a false one. A hasLocs=false posting must
// not inherit the previous posting's numLocValues; when it did, the merge
// reader consumed location values belonging to later postings and failed with
// "read location values: EOF".
func TestMergeMixedHasLocsPostings(t *testing.T) {
	docA := newStubDocument("a", []*stubField{
		newStubFieldSplitString("_id", nil, "a", true, false, false),
		newStubFieldSplitString("body", nil, "shared shared", true, false, true),
	}, "_all")
	docB := newStubDocument("b", []*stubField{
		newStubFieldSplitString("_id", nil, "b", true, false, false),
		newStubFieldNoVectors("body", "shared"),
	}, "_all")
	docC := newStubDocument("c", []*stubField{
		newStubFieldSplitString("_id", nil, "c", true, false, false),
		newStubFieldSplitString("body", nil, "shared", true, false, true),
	}, "_all")

	sb, _, err := zapPlugin.newWithChunkMode(
		[]index.Document{docA, docB, docC}, DefaultChunkMode, nil)
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	segPath := filepath.Join(dir, "mixed.zap")
	if err := PersistSegmentBase(sb.(*SegmentBase), segPath); err != nil {
		t.Fatal(err)
	}

	segment, err := zapPlugin.Open(segPath)
	if err != nil {
		t.Fatalf("error opening segment: %v", err)
	}
	defer func() {
		if cerr := segment.Close(); cerr != nil {
			t.Fatalf("error closing segment: %v", cerr)
		}
	}()

	mergedPath := filepath.Join(dir, "merged.zap")
	_, _, err = zapPlugin.Merge([]seg.Segment{segment},
		[]*roaring.Bitmap{nil}, mergedPath, nil, nil)
	if err != nil {
		t.Fatalf("merge failed: %v", err)
	}

	merged, err := zapPlugin.Open(mergedPath)
	if err != nil {
		t.Fatalf("error opening merged segment: %v", err)
	}
	defer func() {
		if cerr := merged.Close(); cerr != nil {
			t.Fatalf("error closing merged segment: %v", cerr)
		}
	}()

	dict, err := merged.(*Segment).Dictionary("body")
	if err != nil {
		t.Fatal(err)
	}
	pl, err := dict.PostingsList([]byte("shared"), nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	expectFreqs := []uint64{2, 1, 1}
	expectLocs := []int{2, 0, 1}
	itr := pl.Iterator(true, true, true, nil)
	for i := 0; ; i++ {
		next, err := itr.Next()
		if err != nil {
			t.Fatalf("error iterating merged postings: %v", err)
		}
		if next == nil {
			if i != len(expectFreqs) {
				t.Fatalf("expected %d postings, got %d", len(expectFreqs), i)
			}
			break
		}
		if i >= len(expectFreqs) {
			t.Fatalf("unexpected extra posting for doc %d", next.Number())
		}
		if next.Frequency() != expectFreqs[i] {
			t.Errorf("doc %d: expected freq %d, got %d",
				next.Number(), expectFreqs[i], next.Frequency())
		}
		if len(next.Locations()) != expectLocs[i] {
			t.Errorf("doc %d: expected %d locations, got %d",
				next.Number(), expectLocs[i], len(next.Locations()))
		}
	}
}
