package llama_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llama "github.com/go-skynet/go-llama.cpp"
)

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

type evalCall struct {
	nTokens int
	nPast   int
}

type fakeContext struct {
	mu sync.Mutex

	kvTokens   int
	bytes      int64
	kvFail     bool
	restoreErr bool

	// evalFailAt is the global token position (nPast+offset within the call)
	// at which Eval must fail, when enabled.
	evalFailAt int
	evalFailOn bool
	evalHook   func()
	closeErr   error

	// inFlight tracks concurrent Eval calls on THIS context; overlaps is
	// incremented if a second Eval enters while one is already running.
	inFlight int32
	overlaps int32

	evalHistory []evalCall
	closed      int32
}

func (c *fakeContext) Eval(tokens []int32, nPast int) error {
	c.mu.Lock()
	hook := c.evalHook
	c.mu.Unlock()
	if hook != nil {
		hook()
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.inFlight > 0 {
		c.overlaps++
	}
	c.inFlight++
	defer func() { c.inFlight-- }()

	if c.evalFailOn {
		for i := range tokens {
			if nPast+i == c.evalFailAt {
				return fmt.Errorf("injected eval failure at %d", nPast+i)
			}
		}
	}
	c.evalHistory = append(c.evalHistory, evalCall{nTokens: len(tokens), nPast: nPast})
	c.kvTokens = nPast + len(tokens)
	return nil
}

func (c *fakeContext) Checkpoint() ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return []byte(fmt.Sprintf("state@%d", c.kvTokens)), nil
}

func (c *fakeContext) Restore(data []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.restoreErr {
		return errors.New("injected restore failure")
	}
	var nPast int
	if _, err := fmt.Sscanf(string(data), "state@%d", &nPast); err != nil {
		return err
	}
	c.kvTokens = nPast
	return nil
}

func (c *fakeContext) KVTokens() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := c.kvTokens
	if c.kvFail {
		n++
	}
	return n
}

func (c *fakeContext) Bytes() int64 { return c.bytes }

func (c *fakeContext) Close() error {
	atomic.AddInt32(&c.closed, 1)
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closeErr
}

type fakeEngine struct {
	mu sync.Mutex

	modelBytes    int64
	contextBytes  int64
	openErr       error
	nextContext   *fakeContext
	contexts      []*fakeContext
	opens         int32
	closed        int32
	tokenizeErr   error
	tokenizeCalls int32
	tokenizeHook  func()
	openHook      func()
}

func (e *fakeEngine) Tokenize(text string) ([]int32, error) {
	atomic.AddInt32(&e.tokenizeCalls, 1)
	if e.tokenizeHook != nil {
		e.tokenizeHook()
	}
	if e.tokenizeErr != nil {
		return nil, e.tokenizeErr
	}
	// Deterministic, prefix-stable mapping: one token per rune.
	tokens := make([]int32, 0, len(text))
	for _, r := range text {
		tokens = append(tokens, int32(r))
	}
	return tokens, nil
}

func (e *fakeEngine) Open(cfg llama.EngineConfig) (llama.EngineContext, error) {
	e.mu.Lock()
	hook := e.openHook
	e.mu.Unlock()
	if hook != nil {
		hook()
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	atomic.AddInt32(&e.opens, 1)
	if e.openErr != nil {
		return nil, e.openErr
	}
	var ctx *fakeContext
	if e.nextContext != nil {
		ctx = e.nextContext
		e.nextContext = nil
	} else {
		ctx = &fakeContext{bytes: e.contextBytes}
	}
	e.contexts = append(e.contexts, ctx)
	return ctx, nil
}

func (e *fakeEngine) Bytes() int64 { return e.modelBytes }

func (e *fakeEngine) Close() error {
	atomic.AddInt32(&e.closed, 1)
	return nil
}

func newManager(t *testing.T, eng *fakeEngine, cfg llama.SessionConfig) *llama.SessionManager {
	t.Helper()
	base := llama.SessionConfig{
		ContextSize:     64,
		MaxLive:         4,
		VRAMBudgetBytes: 1 << 30,
		CheckpointEvery: 8,
		MaxCheckpoints:  4,
		Now:             func() time.Time { return time.Unix(1700000000, 0) },
	}
	if cfg.ContextSize != 0 {
		base.ContextSize = cfg.ContextSize
	}
	if cfg.MaxLive != 0 {
		base.MaxLive = cfg.MaxLive
	}
	if cfg.VRAMBudgetBytes != 0 {
		base.VRAMBudgetBytes = cfg.VRAMBudgetBytes
	}
	if cfg.IdleTTL != 0 {
		base.IdleTTL = cfg.IdleTTL
	}
	if cfg.SnapshotDir != "" {
		base.SnapshotDir = cfg.SnapshotDir
	}
	if cfg.CheckpointEvery != 0 {
		base.CheckpointEvery = cfg.CheckpointEvery
	}
	if cfg.MaxCheckpoints != 0 {
		base.MaxCheckpoints = cfg.MaxCheckpoints
	}
	if cfg.Now != nil {
		base.Now = cfg.Now
	}
	m, err := llama.NewSessionManager(base, eng)
	if err != nil {
		t.Fatalf("NewSessionManager: %v", err)
	}
	return m
}

func textOf(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = rune('a' + (i % 26))
	}
	return string(b)
}

// ---------------------------------------------------------------------------
// Configuration validation
// ---------------------------------------------------------------------------

func TestNewSessionManagerInvalidConfig(t *testing.T) {
	eng := &fakeEngine{}
	cases := []llama.SessionConfig{
		{ContextSize: 7, MaxLive: 1, VRAMBudgetBytes: 100, CheckpointEvery: 1, MaxCheckpoints: 1},
		{ContextSize: 8, MaxLive: 0, VRAMBudgetBytes: 100, CheckpointEvery: 1, MaxCheckpoints: 1},
		{ContextSize: 8, MaxLive: 1, VRAMBudgetBytes: 0, CheckpointEvery: 1, MaxCheckpoints: 1},
		{ContextSize: 8, MaxLive: 1, VRAMBudgetBytes: 100, CheckpointEvery: 0, MaxCheckpoints: 1},
		{ContextSize: 8, MaxLive: 1, VRAMBudgetBytes: 100, CheckpointEvery: 1, MaxCheckpoints: 0},
	}
	for i, cfg := range cases {
		if _, err := llama.NewSessionManager(cfg, eng); !errors.Is(err, llama.ErrInvalidConfig) {
			t.Fatalf("case %d: want ErrInvalidConfig, got %v", i, err)
		}
	}
	if _, err := llama.NewSessionManager(llama.SessionConfig{ContextSize: 8, MaxLive: 1, VRAMBudgetBytes: 100, CheckpointEvery: 1, MaxCheckpoints: 1}, nil); !errors.Is(err, llama.ErrInvalidConfig) {
		t.Fatalf("nil engine: want ErrInvalidConfig, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// Cold start and pure append reuse
// ---------------------------------------------------------------------------

func TestSyncColdStartAndReuse(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{})
	defer m.Close()

	ctx := context.Background()
	rep, err := m.Sync(ctx, "s1", textOf(10))
	if err != nil {
		t.Fatalf("first Sync: %v", err)
	}
	if !rep.ColdStart || rep.TotalTokens != 10 || rep.EvaluatedTokens != 10 ||
		rep.ReusedTokens != 0 || rep.Restored {
		t.Fatalf("unexpected cold report: %+v", rep)
	}

	// Identical text: nothing to evaluate.
	rep, err = m.Sync(ctx, "s1", textOf(10))
	if err != nil {
		t.Fatalf("second Sync: %v", err)
	}
	if rep.ColdStart || rep.EvaluatedTokens != 0 || rep.ReusedTokens != 10 ||
		rep.RewoundTokens != 0 {
		t.Fatalf("unexpected reuse report: %+v", rep)
	}
	fc := eng.contexts[0]
	fc.mu.Lock()
	hist := append([]evalCall(nil), fc.evalHistory...)
	fc.mu.Unlock()
	// CheckpointEvery=8 splits the 10-token cold eval at the boundary.
	if len(hist) != 2 || hist[0].nPast != 0 || hist[0].nTokens != 8 ||
		hist[1].nPast != 8 || hist[1].nTokens != 2 {
		t.Fatalf("expected cold evals 8@0 + 2@8, got %+v", hist)
	}

	// Append: only the suffix is evaluated at nPast=10.
	rep, err = m.Sync(ctx, "s1", textOf(15))
	if err != nil {
		t.Fatalf("append Sync: %v", err)
	}
	if rep.EvaluatedTokens != 5 || rep.ReusedTokens != 10 || rep.TotalTokens != 15 {
		t.Fatalf("unexpected append report: %+v", rep)
	}
	fc.mu.Lock()
	hist = append([]evalCall(nil), fc.evalHistory...)
	fc.mu.Unlock()
	last := hist[len(hist)-1]
	if last.nPast != 10 || last.nTokens != 5 {
		t.Fatalf("append eval mismatch: %+v", last)
	}

	stats := m.Stats()
	if stats.ColdStarts != 1 || stats.ReusedTokens != 20 || stats.EvaluatedTokens != 15 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	if stats.Live != 1 || stats.Idle != 1 || stats.Busy != 0 {
		t.Fatalf("unexpected live stats: %+v", stats)
	}
	if !stats.PersistDisabled {
		t.Fatalf("PersistDisabled should be true without SnapshotDir")
	}
}

func TestSyncEmptyAndTokenizeErrors(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{})
	defer m.Close()

	if _, err := m.Sync(context.Background(), "s1", ""); !errors.Is(err, llama.ErrEmptyText) {
		t.Fatalf("empty text: want ErrEmptyText, got %v", err)
	}
	if _, ok := m.State("s1"); ok {
		t.Fatalf("empty Sync must not create session state")
	}

	eng.tokenizeErr = errors.New("boom")
	if _, err := m.Sync(context.Background(), "s1", "abc"); err == nil {
		t.Fatalf("tokenize error should propagate")
	}
	eng.tokenizeErr = nil
	rep, err := m.Sync(context.Background(), "s1", "abc")
	if err != nil {
		t.Fatalf("recovered Sync: %v", err)
	}
	if !rep.ColdStart {
		t.Fatalf("failed tokenize must leave session cold")
	}
}

func TestSyncContextTooLong(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{ContextSize: 16}) // max 12 tokens
	defer m.Close()

	_, err := m.Sync(context.Background(), "s1", textOf(13))
	if !errors.Is(err, llama.ErrContextTooLong) {
		t.Fatalf("want ErrContextTooLong, got %v", err)
	}
	if atomic.LoadInt32(&eng.opens) != 0 {
		t.Fatalf("oversized Sync must not open a context")
	}
	// Ledger untouched: a subsequent valid sync is still cold.
	rep, err := m.Sync(context.Background(), "s1", textOf(10))
	if err != nil || !rep.ColdStart {
		t.Fatalf("ledger must be unchanged after oversize, got %+v %v", rep, err)
	}
}

// ---------------------------------------------------------------------------
// Forking rewrite with checkpoints
// ---------------------------------------------------------------------------

func TestSyncForkWithCheckpoints(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{CheckpointEvery: 8, MaxCheckpoints: 4})
	defer m.Close()

	ctx := context.Background()
	base := textOf(40)
	if _, err := m.Sync(ctx, "s1", base); err != nil {
		t.Fatalf("base sync: %v", err)
	}

	// Fork at rune position 23 (LCP p=23 -> rewound = 40-23 = 17).
	forked := []rune(base)
	forked[23] = 'Z'
	forked = append(forked[:24], []rune(textOf(6))...) // 24 + 6 = 30 tokens
	rep, err := m.Sync(ctx, "s1", string(forked))
	if err != nil {
		t.Fatalf("fork sync: %v", err)
	}
	if rep.ReusedTokens != 23 || rep.RewoundTokens != 17 || rep.TotalTokens != 30 {
		t.Fatalf("unexpected fork report: %+v", rep)
	}
	// Nearest checkpoint not later than 23 is nPast=16.
	if rep.ReplayedTokens > 8 || rep.ReplayedTokens != 23-16 {
		t.Fatalf("replayed tokens mismatch: %+v", rep)
	}
	if rep.CheckpointHits != 1 || rep.EvaluatedTokens != 30-23 {
		t.Fatalf("checkpoint accounting mismatch: %+v", rep)
	}
	fc := eng.contexts[0]
	fc.mu.Lock()
	kv := fc.kvTokens
	fc.mu.Unlock()
	if kv != rep.TotalTokens {
		t.Fatalf("KVTokens %d != TotalTokens %d", kv, rep.TotalTokens)
	}
}

func TestSyncForkBeforeAllCheckpoints(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{CheckpointEvery: 8, MaxCheckpoints: 4})
	defer m.Close()

	ctx := context.Background()
	base := textOf(40)
	if _, err := m.Sync(ctx, "s1", base); err != nil {
		t.Fatalf("base sync: %v", err)
	}
	// With MaxCheckpoints=4 only nPast in {16,24,32,40} survive. Fork at 10.
	forked := []rune(base)
	forked[10] = 'Z'
	rep, err := m.Sync(ctx, "s1", string(forked))
	if err != nil {
		t.Fatalf("fork sync: %v", err)
	}
	if !rep.ColdStart || rep.EvaluatedTokens != 40 || rep.ReplayedTokens != 0 {
		t.Fatalf("expected cold restart, got %+v", rep)
	}
}

func TestSyncShortening(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{})
	defer m.Close()

	ctx := context.Background()
	if _, err := m.Sync(ctx, "s1", textOf(20)); err != nil {
		t.Fatalf("base sync: %v", err)
	}
	// Shorten to 12 tokens; nearest checkpoint <=12 is nPast=8.
	rep, err := m.Sync(ctx, "s1", textOf(12))
	if err != nil {
		t.Fatalf("shorten sync: %v", err)
	}
	if rep.TotalTokens != 12 || rep.ReusedTokens != 12 || rep.RewoundTokens != 8 ||
		rep.ReplayedTokens != 4 || rep.EvaluatedTokens != 0 {
		t.Fatalf("unexpected shorten report: %+v", rep)
	}
	if rep.CheckpointHits != 1 {
		t.Fatalf("expected one checkpoint hit, got %+v", rep)
	}
}

// ---------------------------------------------------------------------------
// Eviction: MaxLive, VRAM budget, idle TTL
// ---------------------------------------------------------------------------

func TestEvictionMaxLiveAndBudget(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	var now int64 = 1000
	clock := func() time.Time { return time.Unix(atomic.LoadInt64(&now), 0) }
	m := newManager(t, eng, llama.SessionConfig{
		MaxLive:         2,
		VRAMBudgetBytes: 300,
		Now:             clock,
	})
	defer m.Close()

	ctx := context.Background()
	for i := 0; i < 3; i++ {
		atomic.StoreInt64(&now, int64(1000+i))
		if _, err := m.Sync(ctx, fmt.Sprintf("s%d", i), textOf(4)); err != nil {
			t.Fatalf("sync %d: %v", i, err)
		}
	}
	stats := m.Stats()
	if stats.Live != 2 || stats.ContextBytes > 300 {
		t.Fatalf("capacity stats mismatch: %+v", stats)
	}
	if state, _ := m.State("s0"); state != llama.SessionEvicted {
		t.Fatalf("s0 should be evicted, got %v", state)
	}
	if got := atomic.LoadInt32(&eng.contexts[0].closed); got != 1 {
		t.Fatalf("evicted context closed %d times, want 1", got)
	}
}

func TestEvictionIdleTTL(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	var now int64 = 1000
	clock := func() time.Time { return time.Unix(atomic.LoadInt64(&now), 0) }
	m := newManager(t, eng, llama.SessionConfig{IdleTTL: 10 * time.Second, Now: clock})
	defer m.Close()

	ctx := context.Background()
	if _, err := m.Sync(ctx, "s1", textOf(4)); err != nil {
		t.Fatalf("sync: %v", err)
	}
	atomic.StoreInt64(&now, 1020)
	stats := m.Stats() // triggers idle sweep
	if stats.Live != 0 {
		t.Fatalf("idle session should be reaped, stats=%+v", stats)
	}
	if state, _ := m.State("s1"); state != llama.SessionEvicted {
		t.Fatalf("state = %v, want SessionEvicted", state)
	}
}

func TestNoCapacityContextExpiry(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	// One slot. The first session holds it while busy inside Eval, so the
	// second session can neither evict it nor open a context.
	m := newManager(t, eng, llama.SessionConfig{MaxLive: 1, VRAMBudgetBytes: 1 << 30})
	defer m.Close()

	release := make(chan struct{})
	entered := make(chan struct{})
	slow := &fakeContext{
		bytes: 120,
		evalHook: func() {
			select {
			case <-entered:
			default:
				close(entered)
			}
			<-release
		},
	}
	eng.nextContext = slow

	go func() {
		_, _ = m.Sync(context.Background(), "a", textOf(4))
	}()
	<-entered

	// a is Busy (not evictable) and the only slot is taken: c must wait and
	// honor its deadline.
	cctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	_, err := m.Sync(cctx, "c", textOf(4))
	if !errors.Is(err, llama.ErrNoCapacity) {
		t.Fatalf("want ErrNoCapacity on ctx expiry, got %v", err)
	}

	close(release)
	// Let the slow Sync finish and the manager become quiescent for Close.
	deadline := time.Now().Add(2 * time.Second)
	for m.Stats().Busy != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
}

// ---------------------------------------------------------------------------
// Snapshot persistence and corruption
// ---------------------------------------------------------------------------

func TestSnapshotRestoreAndCorruption(t *testing.T) {
	dir := t.TempDir()
	eng := &fakeEngine{contextBytes: 120}
	var now int64 = 1000
	m := newManager(t, eng, llama.SessionConfig{
		SnapshotDir: dir,
		IdleTTL:     time.Second,
		Now:         func() time.Time { return time.Unix(atomic.LoadInt64(&now), 0) },
	})

	ctx := context.Background()
	if _, err := m.Sync(ctx, "s1", textOf(10)); err != nil {
		t.Fatalf("sync: %v", err)
	}
	snap := filepath.Join(dir, "s1.snap")
	if _, err := os.Stat(snap); err == nil {
		t.Fatalf("snapshot must only be written on reclamation, not after Sync")
	}

	// Evict via idle TTL through Stats; this is the reclamation path that
	// writes the snapshot.
	atomic.StoreInt64(&now, 1020)
	_ = m.Stats()
	if state, _ := m.State("s1"); state != llama.SessionEvicted {
		t.Fatalf("want evicted, got %v", state)
	}
	if _, err := os.Stat(snap); err != nil {
		t.Fatalf("snapshot not written on eviction: %v", err)
	}

	// Re-sync: restore from disk, zero new evaluations.
	atomic.StoreInt64(&now, 1030)
	rep, err := m.Sync(ctx, "s1", textOf(10))
	if err != nil {
		t.Fatalf("restore sync: %v", err)
	}
	if !rep.Restored || rep.EvaluatedTokens != 0 || rep.ReusedTokens != 10 {
		t.Fatalf("unexpected restore report: %+v", rep)
	}
	beforeCorrupt := m.Stats().CorruptSnapshots

	// Evict again and corrupt the snapshot.
	atomic.StoreInt64(&now, 1050)
	_ = m.Stats()
	data, err := os.ReadFile(snap)
	if err != nil {
		t.Fatalf("read snapshot: %v", err)
	}
	data[0] = 'X'
	if err := os.WriteFile(snap, data, 0o600); err != nil {
		t.Fatalf("corrupt snapshot: %v", err)
	}

	atomic.StoreInt64(&now, 1060)
	rep, err = m.Sync(ctx, "s1", textOf(10))
	if err != nil {
		t.Fatalf("corrupt-snapshot sync must succeed, got %v", err)
	}
	if !rep.ColdStart {
		t.Fatalf("corrupt snapshot must force cold start: %+v", rep)
	}
	stats := m.Stats()
	if stats.CorruptSnapshots != beforeCorrupt+1 {
		t.Fatalf("CorruptSnapshots = %d, want %d", stats.CorruptSnapshots, beforeCorrupt+1)
	}
	if _, err := os.Stat(snap); !os.IsNotExist(err) {
		t.Fatalf("corrupt snapshot should be removed, stat err=%v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

func TestSnapshotWriteFailureStillEvicts(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "missing") // does not exist
	eng := &fakeEngine{contextBytes: 120}
	var now int64 = 1000
	m := newManager(t, eng, llama.SessionConfig{
		SnapshotDir: dir,
		IdleTTL:     time.Second,
		Now:         func() time.Time { return time.Unix(atomic.LoadInt64(&now), 0) },
	})
	defer m.Close()

	if _, err := m.Sync(context.Background(), "s1", textOf(4)); err != nil {
		t.Fatalf("sync: %v", err)
	}
	atomic.StoreInt64(&now, 1020)
	stats := m.Stats()
	if stats.Live != 0 || stats.SnapshotFailures == 0 {
		t.Fatalf("eviction should complete despite snapshot failure: %+v", stats)
	}
	if state, _ := m.State("s1"); state != llama.SessionEvicted {
		t.Fatalf("state = %v", state)
	}
	// Next sync cold-starts.
	atomic.StoreInt64(&now, 1030)
	rep, err := m.Sync(context.Background(), "s1", textOf(4))
	if err != nil || !rep.ColdStart {
		t.Fatalf("want cold start, got %+v %v", rep, err)
	}
}

func TestInvalidSessionID(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{SnapshotDir: t.TempDir()})
	defer m.Close()

	for _, id := range []string{"", "a/b", `a\b`, "..", "a..b"} {
		if _, err := m.Sync(context.Background(), id, "abc"); !errors.Is(err, llama.ErrInvalidSessionID) {
			t.Fatalf("id %q: want ErrInvalidSessionID, got %v", id, err)
		}
		if err := m.CloseSession(id); !errors.Is(err, llama.ErrInvalidSessionID) {
			t.Fatalf("CloseSession %q: want ErrInvalidSessionID, got %v", id, err)
		}
	}
}

// ---------------------------------------------------------------------------
// Eval failure handling
// ---------------------------------------------------------------------------

func TestEvalFailureResetsToCold(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{})
	defer m.Close()

	// First context fails on global token position 2; the replacement works.
	failCtx := &fakeContext{bytes: 120, evalFailOn: true, evalFailAt: 2}
	eng.nextContext = failCtx

	_, err := m.Sync(context.Background(), "s1", textOf(6))
	if err == nil {
		t.Fatalf("injected eval failure should surface")
	}
	stats := m.Stats()
	if stats.EvalFailures != 1 {
		t.Fatalf("EvalFailures = %d, want 1", stats.EvalFailures)
	}
	if state, _ := m.State("s1"); state != llama.SessionCold {
		t.Fatalf("state = %v, want SessionCold", state)
	}
	if atomic.LoadInt32(&failCtx.closed) != 1 {
		t.Fatalf("failed context must be closed once")
	}

	// Next Sync opens a fresh context and evaluates from zero.
	rep, err := m.Sync(context.Background(), "s1", textOf(6))
	if err != nil || !rep.ColdStart || rep.EvaluatedTokens != 6 {
		t.Fatalf("cold restart mismatch: %+v %v", rep, err)
	}
	okCtx := eng.contexts[len(eng.contexts)-1]
	okCtx.mu.Lock()
	hist := append([]evalCall(nil), okCtx.evalHistory...)
	okCtx.mu.Unlock()
	if len(hist) != 1 || hist[0].nPast != 0 {
		t.Fatalf("fresh context should eval from 0, got %+v", hist)
	}
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

func TestConcurrentSyncs(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}

	// Wrap Open so every produced context is collected for overlap checks.
	tracker := &trackingEngine{inner: eng}

	mt, err := llama.NewSessionManager(llama.SessionConfig{
		ContextSize:     64,
		MaxLive:         16,
		VRAMBudgetBytes: 1 << 30,
		CheckpointEvery: 8,
		MaxCheckpoints:  4,
	}, tracker)
	if err != nil {
		t.Fatalf("manager: %v", err)
	}
	defer mt.Close()

	var wg sync.WaitGroup
	// 8 goroutines hammering the same session.
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				if _, err := mt.Sync(context.Background(), "shared", textOf(4+(i%8))); err != nil {
					t.Errorf("shared sync: %v", err)
					return
				}
			}
		}()
	}
	// 8 goroutines on distinct sessions.
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				if _, err := mt.Sync(context.Background(), fmt.Sprintf("p%d", idx), textOf(4)); err != nil {
					t.Errorf("private sync: %v", err)
					return
				}
			}
		}(g)
	}
	wg.Wait()

	for i, fc := range tracker.contexts() {
		fc.mu.Lock()
		overlaps := fc.overlaps
		fc.mu.Unlock()
		if overlaps != 0 {
			t.Fatalf("context %d saw %d overlapping Eval calls", i, overlaps)
		}
	}
}

type trackingEngine struct {
	mu     sync.Mutex
	inner  *fakeEngine
	opened []*fakeContext
}

func (e *trackingEngine) contexts() []*fakeContext {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]*fakeContext, len(e.opened))
	copy(out, e.opened)
	return out
}

func (e *trackingEngine) Tokenize(text string) ([]int32, error) {
	return e.inner.Tokenize(text)
}

func (e *trackingEngine) Open(cfg llama.EngineConfig) (llama.EngineContext, error) {
	ctx, err := e.inner.Open(cfg)
	if err != nil {
		return nil, err
	}
	fc := ctx.(*fakeContext)
	e.mu.Lock()
	e.opened = append(e.opened, fc)
	e.mu.Unlock()
	return fc, nil
}

func (e *trackingEngine) Bytes() int64 { return e.inner.Bytes() }
func (e *trackingEngine) Close() error { return e.inner.Close() }

// ---------------------------------------------------------------------------
// Close semantics
// ---------------------------------------------------------------------------

func TestCloseWaitsForInFlightSync(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	slow := &fakeContext{bytes: 120}
	release := make(chan struct{})
	entered := make(chan struct{})
	slow.evalHook = func() {
		close(entered)
		<-release
	}
	eng.nextContext = slow

	m := newManager(t, eng, llama.SessionConfig{})
	go func() {
		_, _ = m.Sync(context.Background(), "s1", textOf(4))
	}()
	<-entered

	done := make(chan struct{})
	go func() {
		_ = m.Close()
		close(done)
	}()
	select {
	case <-done:
		t.Fatalf("Close returned before in-flight Sync finished")
	case <-time.After(30 * time.Millisecond):
	}
	if _, err := m.Sync(context.Background(), "s2", "abc"); !errors.Is(err, llama.ErrManagerClosed) {
		t.Fatalf("Sync after Close: want ErrManagerClosed, got %v", err)
	}
	close(release)
	<-done
	if err := m.Close(); err != nil {
		t.Fatalf("second Close must return nil, got %v", err)
	}
	if got := atomic.LoadInt32(&eng.closed); got != 1 {
		t.Fatalf("engine Close called %d times, want 1", got)
	}
	if atomic.LoadInt32(&slow.closed) != 1 {
		t.Fatalf("context Close called %d times, want 1", slow.closed)
	}
}

func TestCloseSessionIdempotentAndWaits(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{})
	defer m.Close()

	if _, err := m.Sync(context.Background(), "s1", textOf(4)); err != nil {
		t.Fatalf("sync: %v", err)
	}
	if err := m.CloseSession("s1"); err != nil {
		t.Fatalf("CloseSession: %v", err)
	}
	if err := m.CloseSession("s1"); err != nil {
		t.Fatalf("CloseSession must be idempotent: %v", err)
	}
	if state, _ := m.State("s1"); state != llama.SessionClosed {
		t.Fatalf("state = %v", state)
	}
	if atomic.LoadInt32(&eng.contexts[0].closed) != 1 {
		t.Fatalf("context must close once")
	}
}

func TestSessionBusyOnContextExpiry(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{})
	defer m.Close()

	// Hold the session gate in another goroutine by blocking inside Tokenize.
	release := make(chan struct{})
	entered := make(chan struct{})
	eng.tokenizeHook = func() {
		select {
		case <-entered:
		default:
			close(entered)
		}
		<-release
	}
	go func() { _, _ = m.Sync(context.Background(), "s1", textOf(4)) }()
	<-entered

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	_, err := m.Sync(ctx, "s1", textOf(4))
	if !errors.Is(err, llama.ErrSessionBusy) {
		t.Fatalf("want ErrSessionBusy, got %v", err)
	}
	close(release)
}

// ---------------------------------------------------------------------------
// Additional degradation branches
// ---------------------------------------------------------------------------

func TestVRAMBudgetRejectsContext(t *testing.T) {
	// One context of 120 bytes already exceeds a 100-byte budget after idle
	// eviction has nothing to reclaim.
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{MaxLive: 4, VRAMBudgetBytes: 100})
	defer m.Close()

	_, err := m.Sync(context.Background(), "s1", textOf(4))
	if !errors.Is(err, llama.ErrVRAMBudget) {
		t.Fatalf("want ErrVRAMBudget, got %v", err)
	}
	// The oversized context is closed and never counted as live.
	stats := m.Stats()
	if stats.Live != 0 || stats.ContextBytes != 0 {
		t.Fatalf("budget overflow must leave no live context: %+v", stats)
	}
	if got := atomic.LoadInt32(&eng.contexts[0].closed); got != 1 {
		t.Fatalf("rejected context closed %d times, want 1", got)
	}
}

func TestCorruptCheckpointFallsBackToCold(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{CheckpointEvery: 8, MaxCheckpoints: 4})
	defer m.Close()

	ctx := context.Background()
	base := textOf(40)
	if _, err := m.Sync(ctx, "s1", base); err != nil {
		t.Fatalf("base: %v", err)
	}
	// Make every Restore fail and every KV count disagree; fork at position 23.
	fc := eng.contexts[0]
	fc.mu.Lock()
	fc.restoreErr = true
	fc.mu.Unlock()

	forked := []rune(base)
	forked[23] = 'Z'
	rep, err := m.Sync(ctx, "s1", string(forked))
	if err != nil {
		t.Fatalf("sync must still succeed: %v", err)
	}
	if !rep.ColdStart {
		t.Fatalf("all-corrupt checkpoints must force cold start: %+v", rep)
	}
	if m.Stats().CorruptSnapshots == 0 {
		t.Fatalf("CorruptSnapshots must be incremented")
	}
}

func TestCheckpointKVMismatchRejected(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{CheckpointEvery: 8, MaxCheckpoints: 4})
	defer m.Close()

	ctx := context.Background()
	base := textOf(40)
	if _, err := m.Sync(ctx, "s1", base); err != nil {
		t.Fatalf("base: %v", err)
	}
	// KVTokens reports one more than the restored nPast: every checkpoint is
	// considered corrupt, forcing a cold restart.
	fc := eng.contexts[0]
	fc.mu.Lock()
	fc.kvFail = true
	fc.mu.Unlock()

	forked := []rune(base)
	forked[17] = 'Z'
	rep, err := m.Sync(ctx, "s1", string(forked))
	if err != nil {
		t.Fatalf("sync must still succeed: %v", err)
	}
	if !rep.ColdStart || m.Stats().CorruptSnapshots == 0 {
		t.Fatalf("KV mismatch must corrupt checkpoints and cold-start: %+v", rep)
	}
}

func TestSessionStateString(t *testing.T) {
	cases := map[llama.SessionState]string{
		llama.SessionCold:    "SessionCold",
		llama.SessionWarm:    "SessionWarm",
		llama.SessionBusy:    "SessionBusy",
		llama.SessionEvicted: "SessionEvicted",
		llama.SessionClosed:  "SessionClosed",
	}
	for state, want := range cases {
		if got := state.String(); got != want {
			t.Fatalf("%v.String() = %q, want %q", state, got, want)
		}
	}
	if got := llama.SessionState(99).String(); got != "SessionStateUnknown" {
		t.Fatalf("unknown state string = %q", got)
	}
}

func TestStateReturnsFalseForUnknownID(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{})
	defer m.Close()
	if state, ok := m.State("nope"); ok || state != llama.SessionCold {
		t.Fatalf("unknown id: state=%v ok=%v", state, ok)
	}
}

func TestSnapshotVersionMismatchColdStarts(t *testing.T) {
	dir := t.TempDir()
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{SnapshotDir: dir})
	defer m.Close()

	// Hand-craft a snapshot with a bad version byte.
	path := filepath.Join(dir, "s1.snap")
	bad := []byte("LLSK")
	bad = append(bad, 2, 0, 0, 0) // version 2
	bad = append(bad, make([]byte, 40)...)
	if err := os.WriteFile(path, bad, 0o600); err != nil {
		t.Fatal(err)
	}
	rep, err := m.Sync(context.Background(), "s1", textOf(4))
	if err != nil {
		t.Fatalf("bad snapshot must not fail Sync: %v", err)
	}
	if !rep.ColdStart || m.Stats().CorruptSnapshots == 0 {
		t.Fatalf("version mismatch must cold-start and count corruption: %+v", rep)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("bad snapshot should be removed")
	}
}

func TestCloseSessionUnknownIDIsNoop(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{})
	defer m.Close()
	if err := m.CloseSession("ghost"); err != nil {
		t.Fatalf("unknown id should be a no-op, got %v", err)
	}
}

func TestCloseAggregatesContextErrors(t *testing.T) {
	eng := &fakeEngine{contextBytes: 120}
	m := newManager(t, eng, llama.SessionConfig{MaxLive: 4})

	closeErr := errors.New("context close boom")
	for _, id := range []string{"s1", "s2"} {
		if _, err := m.Sync(context.Background(), id, textOf(4)); err != nil {
			t.Fatalf("sync %s: %v", id, err)
		}
	}
	for _, fc := range eng.contexts {
		fc.mu.Lock()
		fc.closeErr = closeErr
		fc.mu.Unlock()
	}

	err := m.Close()
	if err == nil || !errors.Is(err, closeErr) {
		t.Fatalf("Close must aggregate context errors, got %v", err)
	}
	// Every context is still closed exactly once despite the errors.
	for i, fc := range eng.contexts {
		if got := atomic.LoadInt32(&fc.closed); got != 1 {
			t.Fatalf("context %d closed %d times, want 1", i, got)
		}
	}
	// Repeated Close returns nil and does not double-close the engine.
	if err := m.Close(); err != nil {
		t.Fatalf("second Close must return nil, got %v", err)
	}
	if got := atomic.LoadInt32(&eng.closed); got != 1 {
		t.Fatalf("engine closed %d times, want 1", got)
	}
}

// TestRealModelSmoke runs only when TEST_MODEL is set and exercises the
// cgo SharedEngine end to end: two syncs on the same session, with the second
// evaluating fewer tokens than the first, and no live contexts after Close.
func TestRealModelSmoke(t *testing.T) {
	modelPath := os.Getenv("TEST_MODEL")
	if modelPath == "" {
		t.Skip("test skipped - only makes sense if the TEST_MODEL environment variable is set.")
	}

	eng, err := llama.NewSharedEngine(modelPath)
	if err != nil {
		t.Fatalf("NewSharedEngine: %v", err)
	}
	cfg := llama.SessionConfig{
		ContextSize:     256,
		MaxLive:         2,
		VRAMBudgetBytes: 1 << 34,
		CheckpointEvery: 16,
		MaxCheckpoints:  4,
	}
	m, err := llama.NewSessionManager(cfg, eng)
	if err != nil {
		t.Fatalf("NewSessionManager: %v", err)
	}

	text := "The only winning move is not to play. How about a nice game of chess?"
	first, err := m.Sync(context.Background(), "real", text)
	if err != nil {
		t.Fatalf("first Sync: %v", err)
	}
	if !first.ColdStart || first.EvaluatedTokens == 0 {
		t.Fatalf("first sync should be a cold start, got %+v", first)
	}

	second, err := m.Sync(context.Background(), "real", text)
	if err != nil {
		t.Fatalf("second Sync: %v", err)
	}
	if second.EvaluatedTokens >= first.EvaluatedTokens {
		t.Fatalf("second eval %d must be smaller than first %d", second.EvaluatedTokens, first.EvaluatedTokens)
	}

	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := m.Stats(); got.Live != 0 || got.ContextBytes != 0 {
		t.Fatalf("no live contexts expected after Close, got %+v", got)
	}
}
