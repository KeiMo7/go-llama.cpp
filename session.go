package llama

import (
	"context"
	"crypto/sha256"
	"errors"
	"sync"
	"time"
)

// Sentinel errors reported by the session manager. Callers should use
// errors.Is rather than matching messages.
var (
	ErrInvalidConfig    = errors.New("llama: invalid session manager configuration")
	ErrEmptyText        = errors.New("llama: empty text")
	ErrSessionBusy      = errors.New("llama: session is busy")
	ErrContextTooLong   = errors.New("llama: token sequence exceeds context size")
	ErrNoCapacity       = errors.New("llama: no capacity available")
	ErrManagerClosed    = errors.New("llama: session manager is closed")
	ErrVRAMBudget       = errors.New("llama: VRAM budget exceeded")
	ErrInvalidSessionID = errors.New("llama: invalid session id")
)

// SessionState is the lifecycle state of a managed session.
type SessionState int

const (
	// SessionCold has never been (or can no longer be) evaluated.
	SessionCold SessionState = iota
	// SessionWarm holds a live context and is not currently syncing.
	SessionWarm
	// SessionBusy holds a live context and a Sync is in progress.
	SessionBusy
	// SessionEvicted had its context reclaimed; its ledger and disk
	// snapshot are retained so the session can be restored.
	SessionEvicted
	// SessionClosed is terminal.
	SessionClosed
)

// String returns the constant name for the session state.
func (s SessionState) String() string {
	switch s {
	case SessionCold:
		return "SessionCold"
	case SessionWarm:
		return "SessionWarm"
	case SessionBusy:
		return "SessionBusy"
	case SessionEvicted:
		return "SessionEvicted"
	case SessionClosed:
		return "SessionClosed"
	default:
		return "SessionStateUnknown"
	}
}

// Engine abstracts the model shared between sessions.
type Engine interface {
	// Tokenize tokenizes text with the vocabulary of the shared model.
	Tokenize(text string) ([]int32, error)
	// Open creates a new context on top of the shared model.
	Open(EngineConfig) (EngineContext, error)
	// Bytes reports the size in bytes of the shared model weights.
	Bytes() int64
	// Close releases the shared model handle.
	Close() error
}

// EngineContext is one inference context backed by the shared model.
type EngineContext interface {
	// Eval evaluates tokens starting at KV position nPast.
	Eval(tokens []int32, nPast int) error
	// Checkpoint returns an opaque snapshot of the full context state.
	Checkpoint() ([]byte, error)
	// Restore reinstates a context state previously produced by Checkpoint.
	Restore(data []byte) error
	// KVTokens returns the number of tokens currently held in the KV cache.
	KVTokens() int
	// Bytes reports the context footprint in bytes.
	Bytes() int64
	// Close releases the context.
	Close() error
}

// EngineConfig configures contexts opened through Engine.Open.
type EngineConfig struct {
	ContextSize int
	BatchSize   int
	Threads     int
	F16KV       bool
}

// SessionConfig configures a SessionManager.
type SessionConfig struct {
	ContextSize     int
	MaxLive         int
	VRAMBudgetBytes int64
	IdleTTL         time.Duration
	SnapshotDir     string
	CheckpointEvery int
	MaxCheckpoints  int
	Now             func() time.Time
}

// SyncReport describes the work performed by one Sync call.
type SyncReport struct {
	SessionID       string
	ColdStart       bool
	Restored        bool
	ReusedTokens    int
	RewoundTokens   int
	ReplayedTokens  int
	EvaluatedTokens int
	TotalTokens     int
	CheckpointHits  int
	VRAMBytes       int64
}

// SessionStats exposes manager-wide counters and gauges.
type SessionStats struct {
	Live             int
	Busy             int
	Idle             int
	Evicted          int
	Closed           int
	ColdStarts       int
	Restores         int
	SnapshotFailures int
	CorruptSnapshots int
	EvalFailures     int
	ReusedTokens     int64
	EvaluatedTokens  int64
	ModelBytes       int64
	ContextBytes     int64
	PersistDisabled  bool
}

// defaultThreads mirrors the thread count of DefaultOptions.
const defaultThreads = 4

// checkpoint is an in-memory rollback point. It records the KV position
// (nPast), a digest of the token prefix at that position and the raw
// context state bytes.
type checkpoint struct {
	nPast  int
	digest [sha256.Size]byte
	state  []byte
}

// session is the per-id bookkeeping owned by the manager.
type session struct {
	id          string
	state       SessionState
	ledger      []int32
	checkpoints []*checkpoint
	ctx         EngineContext
	lastUse     time.Time

	// gate has capacity one: holding an element serializes Sync calls for
	// this session. Close/CloseSession also take it before teardown.
	gate chan struct{}
}

// SessionManager keeps one model open and multiplexes sessions over the
// contexts it opens on that model.
type SessionManager struct {
	cfg       SessionConfig
	engineCfg EngineConfig
	eng       Engine

	mu       sync.Mutex
	sessions map[string]*session
	closed   bool

	// contextBytes is the sum of Bytes() over every live context.
	contextBytes int64

	// notify is replaced with a fresh closed channel on every release so
	// capacity waiters can block on it without losing wakeups under
	// contention (closed channels deliver to every waiter).
	notify chan struct{}

	wg sync.WaitGroup

	evicted          int
	closedCount      int
	coldStarts       int
	restores         int
	snapshotFailures int
	corruptSnapshots int
	evalFailures     int
	reusedTokens     int64
	evaluatedTokens  int64
}

// NewSessionManager validates the configuration and returns a new manager.
func NewSessionManager(cfg SessionConfig, eng Engine) (*SessionManager, error) {
	if eng == nil {
		return nil, ErrInvalidConfig
	}
	if cfg.ContextSize < 8 {
		return nil, ErrInvalidConfig
	}
	if cfg.MaxLive < 1 || cfg.CheckpointEvery < 1 || cfg.MaxCheckpoints < 1 {
		return nil, ErrInvalidConfig
	}
	if cfg.VRAMBudgetBytes <= 0 {
		return nil, ErrInvalidConfig
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}

	return &SessionManager{
		cfg: cfg,
		engineCfg: EngineConfig{
			ContextSize: cfg.ContextSize,
			BatchSize:   cfg.ContextSize,
			Threads:     defaultThreads,
			F16KV:       true,
		},
		eng:      eng,
		sessions: map[string]*session{},
		notify:   make(chan struct{}, 1),
	}, nil
}

// maxTokens is the same n_ctx-4 headroom used by Predict.
func (m *SessionManager) maxTokens() int {
	return m.cfg.ContextSize - 4
}

// signal wakes every capacity waiter by closing the current notify channel
// and installing a fresh one. Callers must hold m.mu.
func (m *SessionManager) signal() {
	close(m.notify)
	m.notify = make(chan struct{})
}

// validateID rejects empty ids, path separators and ".." so an id can never
// escape the snapshot directory.
func validateID(id string) error {
	if id == "" || id == ".." || containsSeparator(id) || containsDotDot(id) {
		return ErrInvalidSessionID
	}
	return nil
}

func containsSeparator(id string) bool {
	for _, r := range id {
		if r == '/' || r == '\\' {
			return true
		}
	}
	return false
}

func containsDotDot(id string) bool {
	for i := 0; i+2 <= len(id); i++ {
		if id[i] == '.' && id[i+1] == '.' {
			return true
		}
	}
	return false
}

// commonPrefix returns the element-wise common prefix length, bounded by the
// shorter of the two slices.
func commonPrefix(a, b []int32) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	p := 0
	for p < n && a[p] == b[p] {
		p++
	}
	return p
}

// digestTokens returns the SHA-256 of the little-endian encoding of tokens.
func digestTokens(tokens []int32) [sha256.Size]byte {
	return sha256.Sum256(encodeTokens(tokens))
}

// Sync synchronizes the session identified by id with text, reusing as much
// of the previously evaluated KV prefix as possible.
func (m *SessionManager) Sync(ctx context.Context, id, text string) (SyncReport, error) {
	if err := validateID(id); err != nil {
		return SyncReport{SessionID: id}, err
	}
	if text == "" {
		return SyncReport{SessionID: id}, ErrEmptyText
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return SyncReport{SessionID: id}, ErrManagerClosed
	}
	s, ok := m.sessions[id]
	if !ok {
		s = &session{
			id:   id,
			gate: make(chan struct{}, 1),
		}
		m.sessions[id] = s
	}
	if s.state == SessionClosed {
		m.mu.Unlock()
		return SyncReport{SessionID: id}, ErrManagerClosed
	}
	m.mu.Unlock()

	// Serialize syncs targeting the same session.
	select {
	case s.gate <- struct{}{}:
	case <-ctx.Done():
		return SyncReport{SessionID: id}, ErrSessionBusy
	}
	defer func() { <-s.gate }()

	m.mu.Lock()
	if m.closed || s.state == SessionClosed {
		m.mu.Unlock()
		return SyncReport{SessionID: id}, ErrManagerClosed
	}
	prevState := s.state
	s.state = SessionBusy
	m.mu.Unlock()

	m.wg.Add(1)
	defer m.wg.Done()

	tokens, err := m.eng.Tokenize(text)
	if err != nil {
		m.revertState(s, prevState)
		return SyncReport{SessionID: id}, err
	}
	if len(tokens) == 0 {
		m.revertState(s, prevState)
		return SyncReport{SessionID: id}, ErrEmptyText
	}
	if len(tokens) > m.maxTokens() {
		m.revertState(s, prevState)
		return SyncReport{SessionID: id}, ErrContextTooLong
	}

	report, failed, err := m.runSync(ctx, s, tokens)
	if err != nil {
		if !failed {
			m.revertState(s, prevState)
		}
		return report, err
	}

	m.mu.Lock()
	// A concurrent Close/CloseSession may have moved the session to Closed;
	// never resurrect it into Warm.
	if !m.closed && s.state != SessionClosed {
		s.state = SessionWarm
		s.lastUse = m.cfg.Now()
	}
	report.SessionID = id
	report.TotalTokens = len(s.ledger)
	report.VRAMBytes = m.contextBytes
	if report.Restored {
		m.restores++
	}
	m.reusedTokens += int64(report.ReusedTokens)
	m.evaluatedTokens += int64(report.EvaluatedTokens)
	m.mu.Unlock()
	return report, nil
}

// revertState restores the pre-Sync state after a side-effect-free failure.
func (m *SessionManager) revertState(s *session, prev SessionState) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.closed && s.state == SessionBusy {
		s.state = prev
	}
}

// runSync plans and executes the evaluation. The boolean result is true when
// the failure already transitioned the session (eval failure path).
func (m *SessionManager) runSync(ctx context.Context, s *session, tokens []int32) (SyncReport, bool, error) {
	report := SyncReport{}

	// Reclaim idle sessions before spending capacity.
	if err := m.reapIdle(); err != nil {
		return report, false, err
	}

	cold := false
	if s.ctx == nil {
		// Fix the reuse prefix against the ledger as it stands before any
		// context work: a cold start replaces an empty/evicted ledger and
		// must therefore report zero reused tokens.
		p := commonPrefix(s.ledger, tokens)
		report.ReusedTokens = p
		report.RewoundTokens = len(s.ledger) - p
		restored, err := m.ensureContext(ctx, s, tokens, &report)
		if err != nil {
			return report, false, err
		}
		report.Restored = restored
		if !restored {
			// A cold start evaluated the full sequence: the plan is complete.
			report.ReusedTokens = 0
			report.RewoundTokens = 0
			cold = true
		}
	}

	if cold {
		return report, false, nil
	}

	// Measure the prefix against the ledger currently in place. For a
	// restored snapshot this is the on-disk ledger; for a live session it is
	// the in-memory ledger.
	p := commonPrefix(s.ledger, tokens)
	report.ReusedTokens = p
	report.RewoundTokens = len(s.ledger) - p

	if p == len(s.ledger) {
		// Pure append (or identical text): only the suffix needs evaluating.
		err := m.evaluateSuffix(s, tokens, p, &report)
		return report, err != nil, err
	}

	if !m.hasCheckpointAtOrBefore(s, p) {
		// Divergence (or shortening) beyond what an in-memory checkpoint can
		// roll back to: rebuild the context from scratch.
		err := m.coldRestart(s, tokens, &report)
		return report, err != nil, err
	}

	err := m.rewindAndReplay(s, tokens, p, &report)
	return report, err != nil, err
}

// hasCheckpointAtOrBefore reports whether an in-memory checkpoint exists at
// or before position p.
func (m *SessionManager) hasCheckpointAtOrBefore(s *session, p int) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, cp := range s.checkpoints {
		if cp.nPast <= p {
			return true
		}
	}
	return false
}

// ensureContext opens a context for the session, restoring from the disk
// snapshot when one is available and valid. It reports whether a snapshot
// was restored.
func (m *SessionManager) ensureContext(ctx context.Context, s *session, tokens []int32, report *SyncReport) (bool, error) {
	if m.cfg.SnapshotDir != "" {
		restored, err := m.tryRestoreSnapshot(s, report)
		if err != nil {
			return false, err
		}
		if restored {
			return true, nil
		}
		// No usable snapshot: fall through to a cold start.
	}

	if err := m.openContext(ctx, s); err != nil {
		return false, err
	}
	if err := m.evaluateFromScratch(s, tokens, report); err != nil {
		return false, err
	}
	return false, nil
}

// openContext waits for capacity, opens a context and enforces the VRAM
// budget, evicting idle sessions LRU-first when necessary.
func (m *SessionManager) openContext(ctx context.Context, s *session) error {
	if err := m.waitForSlot(ctx, s); err != nil {
		return err
	}

	engineCtx, err := m.eng.Open(m.engineCfg)
	if err != nil {
		return err
	}

	m.mu.Lock()
	s.ctx = engineCtx
	m.contextBytes += engineCtx.Bytes()
	// Evict idle sessions (never the one we just opened) until the budget
	// fits again.
	for m.contextBytes > m.cfg.VRAMBudgetBytes {
		victim := m.lruVictimLocked(s)
		if victim == nil {
			break
		}
		ev := m.detachLocked(victim)
		m.mu.Unlock()
		m.finishEviction(ev)
		m.mu.Lock()
	}
	overBudget := m.contextBytes > m.cfg.VRAMBudgetBytes
	m.mu.Unlock()

	if overBudget {
		// Close the freshly opened context; already reclaimed sessions stay
		// reclaimed.
		m.mu.Lock()
		if s.ctx == engineCtx {
			s.ctx = nil
			m.contextBytes -= engineCtx.Bytes()
		}
		m.signal()
		m.mu.Unlock()
		_ = engineCtx.Close()
		return ErrVRAMBudget
	}
	return nil
}

// waitForSlot blocks while MaxLive contexts are live, evicting idle sessions
// first and otherwise waiting for a release or ctx expiry.
func (m *SessionManager) waitForSlot(ctx context.Context, s *session) error {
	for {
		m.mu.Lock()
		if m.closed || s.state == SessionClosed {
			m.mu.Unlock()
			return ErrManagerClosed
		}
		if m.liveCountLocked() < m.cfg.MaxLive {
			m.mu.Unlock()
			return nil
		}
		if victim := m.lruVictimLocked(s); victim != nil {
			ev := m.detachLocked(victim)
			m.mu.Unlock()
			m.finishEviction(ev)
			continue
		}
		ch := m.notify
		m.mu.Unlock()

		select {
		case <-ch:
		case <-ctx.Done():
			return ErrNoCapacity
		}
	}
}

// liveCountLocked counts sessions with live contexts. Caller holds m.mu.
func (m *SessionManager) liveCountLocked() int {
	n := 0
	for _, sess := range m.sessions {
		if sess.ctx != nil {
			n++
		}
	}
	return n
}

// lruVictimLocked returns the idle session with the oldest lastUse time,
// excluding keep (typically the session acquiring capacity), or nil.
func (m *SessionManager) lruVictimLocked(keep *session) *session {
	var victim *session
	for _, sess := range m.sessions {
		if sess == keep || sess.ctx == nil || sess.state != SessionWarm {
			continue
		}
		if victim == nil || sess.lastUse.Before(victim.lastUse) {
			victim = sess
		}
	}
	return victim
}

// detachLocked marks a victim as reclaimed and removes its context from the
// live accounting without touching the engine. Caller holds m.mu.
func (m *SessionManager) detachLocked(victim *session) eviction {
	ctx := victim.ctx
	if ctx != nil {
		m.contextBytes -= ctx.Bytes()
	}
	victim.ctx = nil
	victim.checkpoints = nil
	victim.state = SessionEvicted
	m.evicted++
	m.signal()
	return eviction{session: victim, ctx: ctx}
}

// eviction bundles a detached session with its context for off-lock work.
type eviction struct {
	session *session
	ctx     EngineContext
}

// finishEviction writes the snapshot and closes the context outside the lock.
func (m *SessionManager) finishEviction(ev eviction) {
	if ev.ctx == nil {
		return
	}
	m.teardownContext(ev.session, ev.ctx)
}

// teardownContext snapshots then closes a detached context. Snapshot failures
// are counted but never surfaced; the close always runs.
func (m *SessionManager) teardownContext(victim *session, ctx EngineContext) {
	if m.cfg.SnapshotDir != "" && len(victim.ledger) > 0 {
		if err := m.writeSnapshot(victim, ctx); err != nil {
			m.mu.Lock()
			m.snapshotFailures++
			m.mu.Unlock()
		}
	}
	_ = ctx.Close()
}

// reapIdle reclaims warm sessions idle for longer than IdleTTL. A non-positive
// IdleTTL disables idle reclamation.
func (m *SessionManager) reapIdle() error {
	if m.cfg.IdleTTL <= 0 {
		return nil
	}
	now := m.cfg.Now()
	var victims []eviction
	m.mu.Lock()
	for _, sess := range m.sessions {
		if sess.ctx != nil && sess.state == SessionWarm &&
			now.Sub(sess.lastUse) > m.cfg.IdleTTL {
			victims = append(victims, m.detachLocked(sess))
		}
	}
	m.mu.Unlock()

	for _, ev := range victims {
		m.finishEviction(ev)
	}
	return nil
}

// evaluateFromScratch runs a full cold evaluation and replaces the ledger.
func (m *SessionManager) evaluateFromScratch(s *session, tokens []int32, report *SyncReport) error {
	m.mu.Lock()
	m.coldStarts++
	m.mu.Unlock()
	report.ColdStart = true

	newLedger := cloneTokens(tokens)
	if err := m.evalWithCheckpoints(s, s.ctx, newLedger, 0); err != nil {
		m.handleEvalFailure(s)
		return err
	}
	m.mu.Lock()
	s.ledger = newLedger
	m.mu.Unlock()
	report.EvaluatedTokens = len(newLedger)
	return nil
}

// coldRestart closes the current context, opens a fresh one and evaluates
// the whole token sequence from position zero.
func (m *SessionManager) coldRestart(s *session, tokens []int32, report *SyncReport) error {
	m.mu.Lock()
	ctx := s.ctx
	if ctx != nil {
		m.contextBytes -= ctx.Bytes()
		s.ctx = nil
	}
	s.checkpoints = nil
	m.coldStarts++
	m.signal()
	m.mu.Unlock()
	if ctx != nil {
		_ = ctx.Close()
	}
	report.ColdStart = true

	if err := m.openContext(context.Background(), s); err != nil {
		return err
	}
	newLedger := cloneTokens(tokens)
	if err := m.evalWithCheckpoints(s, s.ctx, newLedger, 0); err != nil {
		m.handleEvalFailure(s)
		return err
	}
	m.mu.Lock()
	s.ledger = newLedger
	m.mu.Unlock()
	report.EvaluatedTokens = len(newLedger)
	return nil
}

// evaluateSuffix evaluates the tokens following the reused prefix.
func (m *SessionManager) evaluateSuffix(s *session, tokens []int32, p int, report *SyncReport) error {
	suffix := tokens[p:]
	if len(suffix) == 0 {
		return nil
	}
	newLedger := make([]int32, 0, len(tokens))
	newLedger = append(newLedger, s.ledger...)
	newLedger = append(newLedger, suffix...)

	if err := m.evalWithCheckpoints(s, s.ctx, newLedger, p); err != nil {
		m.handleEvalFailure(s)
		return err
	}
	m.mu.Lock()
	s.ledger = newLedger
	m.mu.Unlock()
	report.EvaluatedTokens = len(suffix)
	return nil
}

// rewindAndReplay restores the newest checkpoint not later than p, replays
// the prefix tokens up to p and evaluates the divergent suffix.
func (m *SessionManager) rewindAndReplay(s *session, tokens []int32, p int, report *SyncReport) error {
	cp, err := m.restoreLatestCheckpoint(s, p)
	if err != nil {
		// Every checkpoint is unusable: rebuild from scratch.
		m.mu.Lock()
		m.corruptSnapshots++
		m.mu.Unlock()
		return m.coldRestart(s, tokens, report)
	}
	report.CheckpointHits = 1
	report.ReplayedTokens = p - cp.nPast

	// Replay the shared prefix from the checkpoint up to p, without taking
	// new checkpoints.
	if p > cp.nPast {
		if err := plainEvalBatches(s.ctx, tokens[cp.nPast:p], cp.nPast, m.engineCfg.BatchSize); err != nil {
			m.handleEvalFailure(s)
			return err
		}
	}

	newLedger := cloneTokens(tokens)
	if err := m.evalWithCheckpoints(s, s.ctx, newLedger, p); err != nil {
		m.handleEvalFailure(s)
		return err
	}
	m.mu.Lock()
	s.ledger = newLedger
	m.mu.Unlock()
	report.EvaluatedTokens = len(tokens) - p
	return nil
}

// restoreLatestCheckpoint walks checkpoints newest-to-oldest, validates the
// digest and post-restore KV count, and returns the first usable one.
func (m *SessionManager) restoreLatestCheckpoint(s *session, p int) (*checkpoint, error) {
	m.mu.Lock()
	ctx := s.ctx
	cps := make([]*checkpoint, len(s.checkpoints))
	copy(cps, s.checkpoints)
	ledger := cloneTokens(s.ledger)
	m.mu.Unlock()

	for i := len(cps) - 1; i >= 0; i-- {
		cp := cps[i]
		if cp.nPast > p || cp.nPast > len(ledger) {
			continue
		}
		if digestTokens(ledger[:cp.nPast]) != cp.digest {
			continue
		}
		if err := ctx.Restore(cp.state); err != nil {
			continue
		}
		if ctx.KVTokens() != cp.nPast {
			continue
		}
		return cp, nil
	}
	return nil, errors.New("no usable checkpoint")
}

// evalWithCheckpoints evaluates target[startPast:] against the context,
// splitting the stream at CheckpointEvery boundaries so an in-memory
// checkpoint is captured whenever the ledger reaches a multiple of
// CheckpointEvery.
func (m *SessionManager) evalWithCheckpoints(s *session, ctx EngineContext, target []int32, startPast int) error {
	every := m.cfg.CheckpointEvery
	batch := m.engineCfg.BatchSize
	past := startPast
	idx := startPast
	for idx < len(target) {
		nextBoundary := ((past / every) + 1) * every
		chunk := nextBoundary - past
		if batch > 0 && chunk > batch {
			chunk = batch
		}
		if remaining := len(target) - idx; chunk > remaining {
			chunk = remaining
		}
		if chunk <= 0 {
			return errors.New("internal error: non-positive eval chunk")
		}
		if err := ctx.Eval(target[idx:idx+chunk], past); err != nil {
			return err
		}
		past += chunk
		idx += chunk
		if past%every == 0 {
			if err := m.captureCheckpoint(s, ctx, target, past); err != nil {
				return err
			}
		}
	}
	return nil
}

// captureCheckpoint stores the context state at KV position nPast, dropping
// the oldest checkpoint once MaxCheckpoints is exceeded.
func (m *SessionManager) captureCheckpoint(s *session, ctx EngineContext, target []int32, nPast int) error {
	state, err := ctx.Checkpoint()
	if err != nil {
		return err
	}
	cp := &checkpoint{
		nPast:  nPast,
		digest: digestTokens(target[:nPast]),
		state:  cloneBytes(state),
	}
	m.mu.Lock()
	s.checkpoints = append(s.checkpoints, cp)
	if len(s.checkpoints) > m.cfg.MaxCheckpoints {
		s.checkpoints = s.checkpoints[len(s.checkpoints)-m.cfg.MaxCheckpoints:]
	}
	m.mu.Unlock()
	return nil
}

// plainEvalBatches evaluates tokens in batches without capturing checkpoints.
func plainEvalBatches(ctx EngineContext, tokens []int32, startPast, batch int) error {
	if len(tokens) == 0 {
		return nil
	}
	past := startPast
	idx := 0
	for idx < len(tokens) {
		chunk := len(tokens) - idx
		if batch > 0 && chunk > batch {
			chunk = batch
		}
		if err := ctx.Eval(tokens[idx:idx+chunk], past); err != nil {
			return err
		}
		past += chunk
		idx += chunk
	}
	return nil
}

// handleEvalFailure follows the dirty-KV policy: close the context, clear the
// ledger and checkpoints, return the session to Cold and count the failure.
// A stale snapshot is removed best-effort so the next Sync is cold.
func (m *SessionManager) handleEvalFailure(s *session) {
	m.mu.Lock()
	ctx := s.ctx
	if ctx != nil {
		m.contextBytes -= ctx.Bytes()
	}
	s.ctx = nil
	s.ledger = nil
	s.checkpoints = nil
	s.state = SessionCold
	m.evalFailures++
	m.signal()
	m.mu.Unlock()
	if ctx != nil {
		_ = ctx.Close()
	}
	m.removeSnapshot(s)
}

// CloseSession closes the session identified by id. It is idempotent and
// waits for an in-flight Sync to finish before tearing the context down.
func (m *SessionManager) CloseSession(id string) error {
	if err := validateID(id); err != nil {
		return err
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return ErrManagerClosed
	}
	s, ok := m.sessions[id]
	if !ok || s.state == SessionClosed {
		m.mu.Unlock()
		return nil
	}
	gate := s.gate
	m.mu.Unlock()

	// Reject new Syncs and wait for any in-flight one before teardown.
	gate <- struct{}{}
	m.mu.Lock()
	ev := m.detachForCloseLocked(s)
	m.mu.Unlock()
	<-gate

	if ev.ctx != nil {
		m.teardownContext(ev.session, ev.ctx)
	}
	return nil
}

// detachForCloseLocked moves a session to Closed and detaches its context.
func (m *SessionManager) detachForCloseLocked(sess *session) eviction {
	ctx := sess.ctx
	if ctx != nil {
		m.contextBytes -= ctx.Bytes()
	}
	sess.ctx = nil
	sess.checkpoints = nil
	sess.state = SessionClosed
	m.closedCount++
	m.signal()
	return eviction{session: sess, ctx: ctx}
}

// State returns the current state of a session.
func (m *SessionManager) State(id string) (SessionState, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sessions[id]
	if !ok {
		return SessionCold, false
	}
	return s.state, true
}

// Stats reclaims idle sessions and returns a snapshot of the counters.
func (m *SessionManager) Stats() SessionStats {
	_ = m.reapIdle()

	m.mu.Lock()
	defer m.mu.Unlock()

	stats := SessionStats{
		Evicted:          m.evicted,
		Closed:           m.closedCount,
		ColdStarts:       m.coldStarts,
		Restores:         m.restores,
		SnapshotFailures: m.snapshotFailures,
		CorruptSnapshots: m.corruptSnapshots,
		EvalFailures:     m.evalFailures,
		ReusedTokens:     m.reusedTokens,
		EvaluatedTokens:  m.evaluatedTokens,
		ModelBytes:       m.eng.Bytes(),
		ContextBytes:     m.contextBytes,
		PersistDisabled:  m.cfg.SnapshotDir == "",
	}
	for _, sess := range m.sessions {
		switch sess.state {
		case SessionBusy:
			stats.Busy++
		case SessionWarm:
			stats.Idle++
		}
	}
	stats.Live = stats.Busy + stats.Idle
	return stats
}

// Close stops accepting new calls, waits for in-flight Syncs, reclaims every
// context through the snapshot/close path and finally closes the engine.
func (m *SessionManager) Close() error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	for _, sess := range m.sessions {
		if sess.state != SessionClosed {
			sess.state = SessionClosed
			m.closedCount++
		}
	}
	m.signal()
	m.mu.Unlock()

	// Wait for in-flight Syncs while their contexts stay attached and valid,
	// so none of them observes a nil s.ctx mid-evaluation.
	m.wg.Wait()

	type pending struct {
		sess *session
		ctx  EngineContext
	}
	var list []pending
	m.mu.Lock()
	for _, sess := range m.sessions {
		if sess.ctx != nil {
			list = append(list, pending{sess, sess.ctx})
			m.contextBytes -= sess.ctx.Bytes()
			sess.ctx = nil
			sess.checkpoints = nil
		}
	}
	m.mu.Unlock()

	var errs []error
	for _, p := range list {
		if m.cfg.SnapshotDir != "" && len(p.sess.ledger) > 0 {
			if err := m.writeSnapshot(p.sess, p.ctx); err != nil {
				m.mu.Lock()
				m.snapshotFailures++
				m.mu.Unlock()
			}
		}
		if err := p.ctx.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if err := m.eng.Close(); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func cloneTokens(in []int32) []int32 {
	out := make([]int32, len(in))
	copy(out, in)
	return out
}

func cloneBytes(in []byte) []byte {
	out := make([]byte, len(in))
	copy(out, in)
	return out
}
