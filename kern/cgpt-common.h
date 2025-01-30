#ifndef CGPT_COMMON
#define CGPT_COMMON

struct gpt_params {
  int32_t seed;
  int32_t n_threads;
  int32_t n_predict;
  int32_t n_parallel;
  int32_t n_batch;
  int32_t n_ctx;
  int32_t n_gpu_layers;

  bool ignore_eos;

  int32_t top_k;
  float   top_p;
  float   temp;
  int32_t repeat_last_n;
  float   repeat_penalty;

  const char *model;
  char *prompt;
  char *token_test;

  bool    interactive;
  int32_t interactive_port;
};

static inline void gpt_params_default(struct gpt_params *p)
{
  p->seed = -1; /* RNG SEED */
  p->n_threads = 1;
  p->n_predict = 200; /* New tokens to predict */
  p->n_parallel = 1; /* Number of parallel streams. */
  p->n_batch = 32; /* batch size for prompt processing. */
  p->n_ctx = 2048; /* context size (this is the KV cache max size) */
  p->n_gpu_layers = 0; /* Numer of layers to offload to the GPU */

  p->ignore_eos = false; /* Ignore EOS token when generating text */

  /* Sampling parameters. */
  p->top_k = 40;
  p->top_p = 0.9f;
  p->temp = 0.9f;
  p->repeat_last_n = 64;
  p->repeat_penalty = 1.00f;

  p->model = "MODELFILE";
  p->prompt = "";
  p->token_test = "";

  p->interactive = false;
  p->interactive_port = -1;
}

void tokenize_words(struct vocab *v, char *text, int32_t **tokens, int *token_count);

int32_t
gpt_sample_top_k_top_p (const float *logits,
                        int vocab_size,
                        int top_k,
                        float top_p,
                        float temp,
                        unsigned long *rng_state);

#endif
