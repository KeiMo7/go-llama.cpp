package llama

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
)

// On-disk snapshot format (all integers little-endian uint32):
//
//	magic "LLSK" | version 1 | nPast | token count | tokens |
//	state length | SHA-256 over (tokens || state bytes) | state bytes
const (
	snapshotMagic   = "LLSK"
	snapshotVersion = uint32(1)
	snapshotSuffix  = ".snap"
	tmpSuffix       = ".tmp"
)

// snapshotPath is the per-id snapshot file inside SnapshotDir.
func (m *SessionManager) snapshotPath(id string) string {
	return filepath.Join(m.cfg.SnapshotDir, id+snapshotSuffix)
}

// encodeTokens renders tokens as little-endian uint32 values.
func encodeTokens(tokens []int32) []byte {
	buf := make([]byte, 4*len(tokens))
	for i, t := range tokens {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(t))
	}
	return buf
}

// writeSnapshot persists the ledger-tail state of a session. The file is
// written to a temporary name and atomically renamed.
func (m *SessionManager) writeSnapshot(s *session, ctx EngineContext) error {
	state, err := ctx.Checkpoint()
	if err != nil {
		return err
	}
	tokens := cloneTokens(s.ledger)
	digest := snapshotDigest(tokens, state)

	var buf []byte
	put := func(b []byte) { buf = append(buf, b...) }
	putU32 := func(v uint32) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], v)
		put(b[:])
	}

	put([]byte(snapshotMagic))
	putU32(snapshotVersion)
	putU32(uint32(len(tokens))) // nPast equals the full ledger length
	putU32(uint32(len(tokens)))
	put(encodeTokens(tokens))
	putU32(uint32(len(state)))
	put(digest[:])
	put(state)

	final := m.snapshotPath(s.id)
	tmp := filepath.Join(m.cfg.SnapshotDir, s.id+tmpSuffix)
	if err := os.WriteFile(tmp, buf, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, final)
}

// snapshotDigest is the SHA-256 over the concatenation of the encoded token
// array and the raw state bytes.
func snapshotDigest(tokens []int32, state []byte) [32]byte {
	h := sha256.New()
	h.Write(encodeTokens(tokens))
	h.Write(state)
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

// tryRestoreSnapshot attempts to load and validate the snapshot for s.
// Results:
//   - (true, nil):  a valid snapshot was loaded and its state restored;
//   - (false, nil): no snapshot exists, or it was corrupt and removed
//     (CorruptSnapshots incremented); the caller should cold-start;
//   - (false, err): a transient error (e.g. capacity) aborted the attempt.
func (m *SessionManager) tryRestoreSnapshot(s *session, report *SyncReport) (bool, error) {
	path := m.snapshotPath(s.id)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		// Unreadable snapshot: treat as corrupt and cold-start.
		m.mu.Lock()
		m.corruptSnapshots++
		m.mu.Unlock()
		m.removeSnapshot(s)
		return false, nil
	}

	tokens, state, ok := parseSnapshot(data)
	if !ok {
		m.mu.Lock()
		m.corruptSnapshots++
		m.mu.Unlock()
		m.removeSnapshot(s)
		return false, nil
	}
	if len(tokens) > m.maxTokens() {
		m.mu.Lock()
		m.corruptSnapshots++
		m.mu.Unlock()
		m.removeSnapshot(s)
		return false, nil
	}

	// Open the context before restoring state into it.
	if err := m.openContext(context.Background(), s); err != nil {
		return false, err
	}
	if err := s.ctx.Restore(state); err != nil || s.ctx.KVTokens() != len(tokens) {
		m.mu.Lock()
		m.corruptSnapshots++
		ctx := s.ctx
		if ctx != nil {
			m.contextBytes -= ctx.Bytes()
			s.ctx = nil
		}
		m.mu.Unlock()
		if ctx != nil {
			_ = ctx.Close()
		}
		m.removeSnapshot(s)
		return false, nil
	}

	m.mu.Lock()
	s.ledger = tokens
	s.state = SessionEvicted // remains evicted until Sync finishes and marks Warm
	m.mu.Unlock()
	return true, nil
}

// parseSnapshot validates the framing and checksum of a snapshot and returns
// the token ledger and state bytes.
func parseSnapshot(data []byte) (tokens []int32, state []byte, ok bool) {
	r := &byteReader{data: data}
	magic := r.bytes(len(snapshotMagic))
	if string(magic) != snapshotMagic {
		return nil, nil, false
	}
	version := r.u32()
	if version != snapshotVersion {
		return nil, nil, false
	}
	nPast := r.u32()
	nTokens := r.u32()
	if r.err || nPast != nTokens {
		return nil, nil, false
	}
	rawTokens := make([]byte, 4*int(nTokens))
	if !r.read(rawTokens) {
		return nil, nil, false
	}
	stateLen := r.u32()
	if r.err {
		return nil, nil, false
	}
	gotDigest := r.bytes(32)
	if r.err {
		return nil, nil, false
	}
	state = r.bytes(int(stateLen))
	if r.err || len(r.data) != r.off {
		return nil, nil, false
	}

	tokens = make([]int32, nTokens)
	for i := range tokens {
		tokens[i] = int32(binary.LittleEndian.Uint32(rawTokens[i*4:]))
	}

	want := snapshotDigest(tokens, state)
	if [32]byte(gotDigest) != want {
		return nil, nil, false
	}
	return tokens, state, true
}

// removeSnapshot deletes the snapshot file best-effort; a removal failure is
// only counted as a snapshot failure and never blocks the caller.
func (m *SessionManager) removeSnapshot(s *session) {
	if m.cfg.SnapshotDir == "" {
		return
	}
	if err := os.Remove(m.snapshotPath(s.id)); err != nil && !errors.Is(err, os.ErrNotExist) {
		m.mu.Lock()
		m.snapshotFailures++
		m.mu.Unlock()
	}
}

// byteReader is a minimal little-endian reader with sticky errors.
type byteReader struct {
	data []byte
	off  int
	err  bool
}

func (r *byteReader) read(dst []byte) bool {
	if r.err || r.off+len(dst) > len(r.data) {
		r.err = true
		return false
	}
	copy(dst, r.data[r.off:r.off+len(dst)])
	r.off += len(dst)
	return true
}

func (r *byteReader) u32() uint32 {
	var b [4]byte
	if !r.read(b[:]) {
		return 0
	}
	return binary.LittleEndian.Uint32(b[:])
}

func (r *byteReader) bytes(n int) []byte {
	if r.err || n < 0 || r.off+n > len(r.data) {
		r.err = true
		return nil
	}
	out := make([]byte, n)
	copy(out, r.data[r.off:r.off+n])
	r.off += n
	return out
}
