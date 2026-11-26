package llama

// #cgo CXXFLAGS: -I${SRCDIR}/llama.cpp/common -I${SRCDIR}/llama.cpp
// #cgo LDFLAGS: -L${SRCDIR}/ -lbinding -lm -lstdc++
// #cgo darwin LDFLAGS: -framework Accelerate
// #cgo darwin CXXFLAGS: -std=c++11
// #include "binding.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// SharedEngine holds one model handle and creates contexts on top of it so
// many sessions can share a single copy of the weights.
type SharedEngine struct {
	mu     sync.Mutex
	ptr    unsafe.Pointer
	closed bool
	addBOS bool
}

// NewSharedEngine loads the model at modelPath and returns a SharedEngine.
// The model stays loaded until Close; contexts created through Open share it.
func NewSharedEngine(modelPath string, opts ...ModelOption) (*SharedEngine, error) {
	mo := NewModelOptions(opts...)

	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	cLora := C.CString(mo.LoraAdapter)
	defer C.free(unsafe.Pointer(cLora))
	cLoraBase := C.CString(mo.LoraBase)
	defer C.free(unsafe.Pointer(cLoraBase))

	ptr := C.llama_binding_load_shared_model(
		cPath,
		C.bool(mo.MMap),
		C.bool(mo.MLock),
		cLora,
		cLoraBase,
	)
	if ptr == nil {
		return nil, fmt.Errorf("failed loading model")
	}

	// Match TokenizeString: SPM vocabularies get a leading BOS token,
	// BPE vocabularies do not.
	addBOS := C.llama_binding_model_vocab_type(ptr) == 0

	return &SharedEngine{ptr: ptr, addBOS: addBOS}, nil
}

// Tokenize tokenizes text against the shared model vocabulary. The result
// matches LLama.TokenizeString for the same model.
func (e *SharedEngine) Tokenize(text string) ([]int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, ErrManagerClosed
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// llama_tokenize returns the required size as a negative count when the
	// buffer is too small; start with the model context size and grow.
	capacity := 512
	for {
		buf := make([]C.int, capacity)
		n := C.llama_binding_tokenize(
			e.ptr,
			cText,
			(*C.int)(unsafe.Pointer(&buf[0])),
			C.int(capacity),
			C.bool(e.addBOS),
		)
		if n < 0 {
			required := int(-n)
			if required <= capacity {
				required = capacity * 2
			}
			capacity = required
			continue
		}
		out := make([]int32, int(n))
		for i := range out {
			out[i] = int32(buf[i])
		}
		return out, nil
	}
}

// Open creates a new context on the shared model.
func (e *SharedEngine) Open(cfg EngineConfig) (EngineContext, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, ErrManagerClosed
	}

	ctx := C.llama_binding_new_context(
		e.ptr,
		C.int(cfg.ContextSize),
		C.int(cfg.BatchSize),
		C.int(cfg.Threads),
		C.bool(cfg.F16KV),
	)
	if ctx == nil {
		return nil, fmt.Errorf("failed creating context")
	}
	return &sharedContext{ptr: ctx, threads: cfg.Threads}, nil
}

// Bytes reports the size of the shared model weights in bytes.
func (e *SharedEngine) Bytes() int64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return 0
	}
	return int64(C.llama_binding_model_bytes(e.ptr))
}

// Close releases the shared model. Contexts created from the engine must be
// closed first.
func (e *SharedEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	if e.ptr != nil {
		C.llama_binding_free_shared_model(e.ptr)
		e.ptr = nil
	}
	return nil
}

// sharedContext is one llama_context created on a SharedEngine model.
type sharedContext struct {
	mu      sync.Mutex
	ptr     unsafe.Pointer
	closed  bool
	threads int
}

// Eval evaluates tokens starting at KV position nPast.
func (c *sharedContext) Eval(tokens []int32, nPast int) error {
	if len(tokens) == 0 {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("context is closed")
	}
	ret := C.llama_binding_eval(
		c.ptr,
		(*C.int)(unsafe.Pointer(&tokens[0])),
		C.int(len(tokens)),
		C.int(nPast),
		C.int(c.threads),
	)
	if ret != 0 {
		return fmt.Errorf("inference failed")
	}
	return nil
}

// Checkpoint returns the raw llama context state (rng, logits, KV cache).
func (c *sharedContext) Checkpoint() ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, fmt.Errorf("context is closed")
	}
	size := int(C.llama_binding_state_size(c.ptr))
	if size <= 0 {
		return nil, fmt.Errorf("invalid context state size")
	}
	buf := make([]byte, size)
	n := C.llama_binding_copy_state(c.ptr, (*C.uint8_t)(unsafe.Pointer(&buf[0])))
	if int(n) != size {
		return nil, fmt.Errorf("short context state copy")
	}
	return buf, nil
}

// Restore reinstates a context state produced by Checkpoint.
func (c *sharedContext) Restore(data []byte) error {
	if len(data) == 0 {
		return fmt.Errorf("empty checkpoint state")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("context is closed")
	}
	n := C.llama_binding_set_state(c.ptr, (*C.uint8_t)(unsafe.Pointer(&data[0])))
	if int(n) != len(data) {
		return fmt.Errorf("short context state restore")
	}
	return nil
}

// KVTokens returns the number of tokens in the context KV cache.
func (c *sharedContext) KVTokens() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0
	}
	return int(C.llama_binding_kv_token_count(c.ptr))
}

// Bytes reports the context footprint in bytes.
func (c *sharedContext) Bytes() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0
	}
	return int64(C.llama_binding_context_bytes(c.ptr))
}

// Close releases the context exactly once.
func (c *sharedContext) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	if c.ptr != nil {
		C.llama_binding_free_context(c.ptr)
		c.ptr = nil
	}
	return nil
}
