#ifdef __cplusplus
#include <vector>
#include <string>
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

extern unsigned char tokenCallback(void *, char *);

int load_state(void *ctx, char *statefile, char*modes);

int eval(void* params_ptr, void *ctx, char*text);

void save_state(void *ctx, char *dst, char*modes);

void* load_model(const char *fname, 
                 int n_ctx, 
                 int n_seed, 
                 bool memory_f16, 
                 bool mlock, 
                 bool embeddings, 
                 bool mmap, 
                 bool low_vram, 
                 int n_gpu, 
                 int n_batch, 
                 const char *maingpu, 
                 const char *tensorsplit, 
                 bool numa, 
                 float rope_freq_base, 
                 float rope_freq_scale,
                 bool mul_mat_q, const char *lora, const char *lora_base, bool perplexity
                 );

int get_embeddings(void* params_ptr, void* state_pr, float * res_embeddings);

int get_token_embeddings(void* params_ptr, void* state_pr,  int *tokens, int tokenSize, float * res_embeddings);

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty, 
                            int repeat_last_n, bool ignore_eos, bool memory_f16, 
                            int n_batch, int n_keep, const char** antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *logit_bias, const char *session_file, bool prompt_cache_all, bool mlock, bool mmap, const char *maingpu, const char *tensorsplit , 
                            bool prompt_cache_ro, const char *grammar, float rope_freq_base, float rope_freq_scale, float negative_prompt_scale, const char* negative_prompt,
                            int n_draft);

int speculative_sampling(void* params_ptr, void* target_model, void* draft_model, char* result, bool debug);

void llama_free_params(void* params_ptr);

void llama_binding_free_model(void* state);

int llama_tokenize_string(void* params_ptr, void* state_pr, int* result);

int llama_predict(void* params_ptr, void* state_pr, char* result, bool debug);

// Shared model binding: one model handle, many contexts (used by the
// session manager). Appended without changing any existing declaration.
void* llama_binding_load_shared_model(const char* fname, bool mmap, bool mlock,
                                      const char* lora, const char* lora_base);
void  llama_binding_free_shared_model(void* model_ptr);
int64_t llama_binding_model_bytes(void* model_ptr);
int   llama_binding_model_vocab_type(void* model_ptr);

void* llama_binding_new_context(void* model_ptr, int n_ctx, int n_batch,
                                int n_threads, bool f16_kv);
void  llama_binding_free_context(void* ctx_ptr);
int   llama_binding_eval(void* ctx_ptr, const int* tokens, int n_tokens,
                         int n_past, int n_threads);
int   llama_binding_tokenize(void* model_ptr, const char* text,
                             int* tokens, int n_max_tokens, bool add_bos);
size_t llama_binding_state_size(void* ctx_ptr);
size_t llama_binding_copy_state(void* ctx_ptr, uint8_t* dst);
size_t llama_binding_set_state(void* ctx_ptr, const uint8_t* src);int   llama_binding_kv_token_count(void* ctx_ptr);
int64_t llama_binding_context_bytes(void* ctx_ptr);

#ifdef __cplusplus
}




std::vector<std::string> create_vector(const char** strings, int count);
void delete_vector(std::vector<std::string>* vec);
#endif
