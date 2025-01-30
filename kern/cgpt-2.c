/*
  The following is a C interpretation of GGML example/gpt-2

  Written by Gianluca Guida <glguida@gmail.com>

  Keeping original license.
*/

/*
MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


#include <math.h>
#include "ggml.h"

#include "util.h"
#include "cgpt-common.h"

// default hparams (GPT-2 117M)
struct gpt2_hparams {
  int32_t n_vocab;
  int32_t n_ctx;
  int32_t n_embd;
  int32_t n_head;
  int32_t n_layer;
  int32_t ftype;
  float   eps;
};

struct gpt2_layer {
    // normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    // attention
    struct ggml_tensor * c_attn_attn_w;
    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    // mlp
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gpt2_model {
    struct gpt2_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte;     // position embedding
    struct ggml_tensor * wpe;     //    token embedding
    struct ggml_tensor * lm_head; // language model head

    struct gpt2_layer *layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx_w;
    struct hashmap tensors;
};

bool gpt2_model_load(void *buf, size_t size, struct gpt2_model *model, struct vocab *v)
{
  struct mapped_file f;

  mfile_init(&f, buf, size);

  /* Verify Magic. */
  {
    uint32_t magic;
    mfile_read(&f, (char *)&magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) {
      fprintf(stderr, "%s: invalid model file (bad magic)\n", __func__);
      return false;
    }
  }

  /* Load HParams. */
  {
    struct gpt2_hparams *hparams = &model->hparams;

    mfile_read(&f, (char *) &hparams->n_vocab, sizeof(hparams->n_vocab));
    mfile_read(&f, (char *) &hparams->n_ctx,   sizeof(hparams->n_ctx));
    mfile_read(&f, (char *) &hparams->n_embd,  sizeof(hparams->n_embd));
    mfile_read(&f, (char *) &hparams->n_head,  sizeof(hparams->n_head));
    mfile_read(&f, (char *) &hparams->n_layer, sizeof(hparams->n_layer));
    mfile_read(&f, (char *) &hparams->ftype,   sizeof(hparams->ftype));

    const int32_t qntvr = hparams->ftype / GGML_QNT_VERSION_FACTOR;

    printf("%s: n_vocab = %d\n", __func__, hparams->n_vocab);
    printf("%s: n_ctx   = %d\n", __func__, hparams->n_ctx);
    printf("%s: n_embd  = %d\n", __func__, hparams->n_embd);
    printf("%s: n_head  = %d\n", __func__, hparams->n_head);
    printf("%s: n_layer = %d\n", __func__, hparams->n_layer);
    printf("%s: ftype   = %d\n", __func__, hparams->ftype);
    printf("%s: qntvr   = %d\n", __func__, qntvr);
    
    hparams->ftype %= GGML_QNT_VERSION_FACTOR;
  }

  /* Load Vocab. */
  {
    int32_t n_vocab = 0;
    mfile_read(&f, (char *)&n_vocab, sizeof(n_vocab));

    if (n_vocab != model->hparams.n_vocab)
      {
	fprintf(stderr, "%s: invalid model file (bad vocab size %d != %d)\n",
		__func__, n_vocab, model->hparams.n_vocab);
	return false;
      }
    vocab_init(v, n_vocab);

    for (int i = 0; i < n_vocab; i++)
      {
	uint32_t len;
	mfile_read(&f, (char *) &len, sizeof(len));
	vocab_add(v, i, len, (char *)mfile_curptr(&f));
	mfile_skip(&f, len);
      }
  }

  /*
    for the big tensors, we have the option to store the data in
     16-bit floats or quantized in order to save memory and also to
     speed up the computation
  */
  enum ggml_type wtype = ggml_ftype_to_ggml_type(model->hparams.ftype);
  if (wtype == GGML_TYPE_COUNT)
    {
      fprintf(stderr, "%s: invalid model file (bad ftype value %d)\n",
	      __func__, model->hparams.ftype);
      return false;
    }

  struct gpt2_hparams  hparams = model->hparams;
  size_t ctx_size = 0;
  {
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;

    ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd); // ln_f_g
    ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd); // ln_f_b

    ctx_size += ggml_row_size(wtype,         n_vocab*n_embd); // wte
    ctx_size += ggml_row_size(GGML_TYPE_F32,   n_ctx*n_embd); // wpe
    ctx_size += ggml_row_size(wtype,         n_vocab*n_embd); // lm_head

    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_1_g
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_1_b

    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_2_g
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_2_b

    ctx_size += n_layer*(ggml_row_size(wtype,         3*n_embd*n_embd)); // c_attn_attn_w
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 3*n_embd));        // c_attn_attn_b

    ctx_size += n_layer*(ggml_row_size(wtype,         n_embd*n_embd));   // c_attn_proj_w
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd));          // c_attn_proj_b

    ctx_size += n_layer*(ggml_row_size(wtype,         4*n_embd*n_embd)); // c_mlp_fc_w
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 4*n_embd));        // c_mlp_fc_b

    ctx_size += n_layer*(ggml_row_size(wtype,         4*n_embd*n_embd)); // c_mlp_proj_w
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 4*n_embd));        // c_mlp_proj_b

    ctx_size += n_ctx*n_layer*ggml_row_size(GGML_TYPE_F32, n_embd); // memory_k
    ctx_size += n_ctx*n_layer*ggml_row_size(GGML_TYPE_F32, n_embd); // memory_v

    ctx_size += (6 + 12*n_layer)*512; // object overhead

    printf("%s: ggml tensor size = %d bytes\n", __func__, (int) sizeof(struct ggml_tensor));
    printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
  }

  /* create the ggml context */
  {
    struct ggml_init_params params = {
      /*.mem_size   =*/ ctx_size,
      /*.mem_buffer =*/ NULL,
      /*.no_alloc   =*/ false,
    };

    model->ctx_w = ggml_init(params);
    if (!model->ctx_w) {
      fprintf(stderr, "%s: ggml_init() failed\n", __func__);
      return false;
    }
  }

  struct ggml_context *ctx = model->ctx_w;

  /* prepare memory for the weights */
#define TENSOR_ADD(_s, _t) do {				  \
    hashmap_add(&model->tensors, (_s), strlen((_s)), _t); \
  } while(0)

  {
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;

    model->layers = malloc(n_layer * sizeof(struct gpt2_layer));

    model->ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    model->ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    model->wte     = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);
    model->wpe     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ctx);
    model->lm_head = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);

    /* map by name */
    hashmap_init(&model->tensors);
    TENSOR_ADD("model/ln_f/g", model->ln_f_g);
    TENSOR_ADD("model/ln_f/b", model->ln_f_b);
    TENSOR_ADD("model/wte", model->wte);
    TENSOR_ADD("model/wpe", model->wpe);
    TENSOR_ADD("model/lm_head", model->lm_head);

    for (unsigned i = 0; i < n_layer; ++i) {
      struct gpt2_layer *layer = model->layers + i;
      layer->ln_1_g        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
      layer->ln_1_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

      layer->ln_2_g        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
      layer->ln_2_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

      layer->c_attn_attn_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, 3*n_embd);
      layer->c_attn_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_embd);

      layer->c_attn_proj_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, n_embd);
      layer->c_attn_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

      layer->c_mlp_fc_w    = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
      layer->c_mlp_fc_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_embd);

      layer->c_mlp_proj_w  = ggml_new_tensor_2d(ctx, wtype,         4*n_embd, n_embd);
      layer->c_mlp_proj_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);


      // map by name
      TENSOR_ADD(asprintf("model/h%d/ln_1/g", i), layer->ln_1_g);
      TENSOR_ADD(asprintf("model/h%d/ln_1/b", i), layer->ln_1_b);
      TENSOR_ADD(asprintf("model/h%d/ln_2/g", i), layer->ln_2_g);
      TENSOR_ADD(asprintf("model/h%d/ln_2/b", i), layer->ln_2_b);

      TENSOR_ADD(asprintf("model/h%d/attn/c_attn/w", i), layer->c_attn_attn_w);
      TENSOR_ADD(asprintf("model/h%d/attn/c_attn/b", i), layer->c_attn_attn_b);

      TENSOR_ADD(asprintf("model/h%d/attn/c_proj/w", i), layer->c_attn_proj_w);
      TENSOR_ADD(asprintf("model/h%d/attn/c_proj/b", i), layer->c_attn_proj_b);

      TENSOR_ADD(asprintf("model/h%d/mlp/c_fc/w", i), layer->c_mlp_fc_w);
      TENSOR_ADD(asprintf("model/h%d/mlp/c_fc/b", i), layer->c_mlp_fc_b);

      TENSOR_ADD(asprintf("model/h%d/mlp/c_proj/w", i), layer->c_mlp_proj_w);
      TENSOR_ADD(asprintf("model/h%d/mlp/c_proj/b", i), layer->c_mlp_proj_b);
    }
  }

  /* key + value memory */
  {
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;

    const int n_mem      = n_layer*n_ctx;
    const int n_elements = n_embd*n_mem;

    model->memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    model->memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

    const size_t memory_size = ggml_nbytes(model->memory_k)
      + ggml_nbytes(model->memory_v);

    printf("%s: memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
  }

  /* load weights */
  {
    size_t total_size = 0;
    bool has_lm_head = false;

    while (true) {
      int32_t n_dims;
      int32_t length;
      int32_t ttype;

      mfile_read(&f, (char *)&n_dims, sizeof(n_dims));
      mfile_read(&f, (char *)&length, sizeof(length));
      mfile_read(&f, (char *)&ttype, sizeof(ttype));

      if (length == 0 || mfile_eof(&f))
	break;

      int32_t nelements;
      int32_t ne[2];
      nelements = 1;
      ne[0] = 1;
      ne[1] = 1;
      for (int i = 0; i < n_dims; ++i) {
	mfile_read(&f, (char *)(ne + i), sizeof(ne[i]));
	nelements *= ne[i];
      }

      char *name = malloc(length);
      mfile_read(&f, name, length);

      struct ggml_tensor *tensor;
      if (!hashmap_get(&model->tensors, name, length, (void *)&tensor))
	{
	  fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name);
	  return false;
	}

      if (ggml_nelements(tensor) != nelements)
	{
	  fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name);
	  return false;
	}

      if ((tensor->ne[0] != ne[0]) || (tensor->ne[1] != ne[1]))
	{
	  fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
		  __func__, name, (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
	  return false;
	}

      //      printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name, ne[0], ne[1], ggml_type_name((enum ggml_type)ttype), ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));

      const size_t bpe = ggml_type_size((enum ggml_type)(ttype));

      if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
	{
	  fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
		  __func__, name, ggml_nbytes(tensor), nelements*bpe);
	  return false;
	}

      mfile_read(&f, (char *)(tensor->data), ggml_nbytes(tensor));

      // GPT-2 models share the WTE tensor as the LM head
      if (!memcmp(name,"model/wte", strlen("model/wte")) && has_lm_head == false)
	{
	  memcpy(model->lm_head->data, tensor->data, ggml_nbytes(tensor));
	}

      if (!memcmp(name,"model/lm_head", strlen("model/lm_head")))
	{
	  has_lm_head = true;
	}

      free(name);
      total_size += ggml_nbytes(tensor);
    }

    printf("%s: model size  = %8.2f (%ld) MB\n", __func__, total_size/1024.0/1024.0, total_size>>20);
  }

  return true;
}

bool gpt2_eval(const struct gpt2_model *model,
	       const int n_threads,
	       const int n_past,
	       int32_t *embd_inp,
	       int embd_inp_count,
	       struct fvec *embd_w,
	       size_t *mem_per_token)
{
  const int N = embd_inp_count;
  struct gpt2_hparams hparams = model->hparams;

  const int n_embd  = hparams.n_embd;
  const int n_layer = hparams.n_layer;
  const int n_ctx   = hparams.n_ctx;
  const int n_head  = hparams.n_head;
  const int n_vocab = hparams.n_vocab;

  static size_t buf_size = 256u*1024*1024;
  static void * buf = NULL;

  if (!buf)
    {
      buf = malloc(buf_size);
      memset(buf, 0, buf_size);
    }

  if ((*mem_per_token) > 0 && (*mem_per_token)*N > buf_size)
    {
      const size_t buf_size_new = 1.1*((*mem_per_token)*N); // add 10% to account for ggml object overhead
      buf_size = buf_size_new;
      buf = realloc(buf, buf_size);
      if (buf == NULL) {
	fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
	return false;
      }
    }

  struct ggml_init_params params = {
    .mem_size = buf_size,
    .mem_buffer = buf,
    .no_alloc = false,
  };

  struct ggml_context * ctx0 = ggml_init(params);
  struct ggml_cgraph * gf = ggml_new_graph(ctx0);

  struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
  memcpy(embd->data, embd_inp, N * sizeof(*embd_inp));

  struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
  for (int i = 0; i < N; ++i) {
    ((int32_t *) position->data)[i] = n_past + i;
  }

  // wte + wpe
  struct ggml_tensor * inpL =
    ggml_add(ctx0,
	     ggml_get_rows(ctx0, model->wte, embd),
	     ggml_get_rows(ctx0, model->wpe, position));

  for (int il = 0; il < n_layer; ++il) {
    struct ggml_tensor * cur;

    // norm
    {
      // [ 768, N]
      cur = ggml_norm(ctx0, inpL, hparams.eps);

      // cur = ln_1_g*cur + ln_1_b
      // [ 768, N]
      cur = ggml_add(ctx0,
		     ggml_mul(ctx0,
			      ggml_repeat(ctx0, model->layers[il].ln_1_g, cur),
			      cur),
		     ggml_repeat(ctx0, model->layers[il].ln_1_b, cur));
    }

    // attn
    // [2304, 768] - model.layers[il].c_attn_attn_w
    // [2304,   1] - model.layers[il].c_attn_attn_b
    // [ 768,   N] - cur (in)
    // [2304,   N] - cur (out)
    //
    // cur = attn_w*cur + attn_b
    // [2304, N]
    {
      cur = ggml_mul_mat(ctx0,
			 model->layers[il].c_attn_attn_w,
			 cur);

      cur = ggml_add(ctx0,
		     ggml_repeat(ctx0, model->layers[il].c_attn_attn_b, cur),
		     cur);
    }

    // self-attention
    {
      struct ggml_tensor * Qcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
      struct ggml_tensor * Kcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd);
      struct ggml_tensor * Vcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

      // store key and value to memory
      if (N >= 1) {
	struct ggml_tensor * k = ggml_view_1d(ctx0, model->memory_k, N*n_embd, (ggml_element_size(model->memory_k)*n_embd)*(il*n_ctx + n_past));
	struct ggml_tensor * v = ggml_view_1d(ctx0, model->memory_v, N*n_embd, (ggml_element_size(model->memory_v)*n_embd)*(il*n_ctx + n_past));

	ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
	ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
      }

      // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
      // [64, N, 12]
      struct ggml_tensor * Q =
	ggml_permute(ctx0,
		     ggml_cpy(ctx0,
			      Qcur,
			      ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
		     0, 2, 1, 3);

      // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
      // [64, n_past + N, 12]
      struct ggml_tensor * K =
	ggml_permute(ctx0,
		     ggml_reshape_3d(ctx0,
				     ggml_view_1d(ctx0, model->memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model->memory_k)*n_embd),
				     n_embd/n_head, n_head, n_past + N),
		     0, 2, 1, 3);

      // GG: flash attention
      //struct ggml_tensor * V =
      //    ggml_cpy(ctx0,
      //            ggml_permute(ctx0,
      //                ggml_reshape_3d(ctx0,
      //                    ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
      //                    n_embd/n_head, n_head, n_past + N),
      //                1, 2, 0, 3),
      //            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

      //struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, true);

      // K * Q
      // [n_past + N, N, 12]
      struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

      // KQ_scaled = KQ / sqrt(n_embd/n_head)
      // [n_past + N, N, 12]
      struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0, KQ, 1.0f/sqrt((float)n_embd/n_head));

      // KQ_masked = mask_past(KQ_scaled)
      // [n_past + N, N, 12]
      struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

      // KQ = soft_max(KQ_masked)
      // [n_past + N, N, 12]
      struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

      // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
      // [n_past + N, 64, 12]
      struct ggml_tensor * V_trans =
	ggml_cpy(ctx0,
		 ggml_permute(ctx0,
			      ggml_reshape_3d(ctx0,
					      ggml_view_1d(ctx0, model->memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model->memory_v)*n_embd),
					      n_embd/n_head, n_head, n_past + N),
			      1, 2, 0, 3),
		 ggml_new_tensor_3d(ctx0, model->memory_v->type, n_past + N, n_embd/n_head, n_head));

      // KQV = transpose(V) * KQ_soft_max
      // [64, N, 12]
      struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      // [64, 12, N]
      struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

      // cur = KQV_merged.contiguous().view(n_embd, N)
      // [768, N]
      cur = ggml_cpy(ctx0,
		     KQV_merged,
		     ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
    }

    // projection
    // [ 768, 768] - model.layers[il].c_attn_proj_w
    // [ 768,   1] - model.layers[il].c_attn_proj_b
    // [ 768,   N] - cur (in)
    // [ 768,   N] - cur (out)
    //
    // cur = proj_w*cur + proj_b
    // [768, N]
    {
      cur = ggml_mul_mat(ctx0,
			 model->layers[il].c_attn_proj_w,
			 cur);

      cur = ggml_add(ctx0,
		     ggml_repeat(ctx0, model->layers[il].c_attn_proj_b, cur),
		     cur);


    }

    // add the input
    cur = ggml_add(ctx0, cur, inpL);

    struct ggml_tensor * inpFF = cur;

    // feed-forward network
    {
      // norm
      {
	cur = ggml_norm(ctx0, inpFF, hparams.eps);

	// cur = ln_2_g*cur + ln_2_b
	// [ 768, N]
	cur = ggml_add(ctx0,
		       ggml_mul(ctx0,
				ggml_repeat(ctx0, model->layers[il].ln_2_g, cur),
				cur),
		       ggml_repeat(ctx0, model->layers[il].ln_2_b, cur));
      }

      // fully connected
      // [3072, 768] - model.layers[il].c_mlp_fc_w
      // [3072,   1] - model.layers[il].c_mlp_fc_b
      // [ 768,   N] - cur (in)
      // [3072,   N] - cur (out)
      //
      // cur = fc_w*cur + fc_b
      // [3072, N]
      cur = ggml_mul_mat(ctx0,
			 model->layers[il].c_mlp_fc_w,
			 cur);

      cur = ggml_add(ctx0,
		     ggml_repeat(ctx0, model->layers[il].c_mlp_fc_b, cur),
		     cur);

      // GELU activation
      // [3072, N]
      cur = ggml_gelu(ctx0, cur);

      // projection
      // [ 768, 3072] - model.layers[il].c_mlp_proj_w
      // [ 768,    1] - model.layers[il].c_mlp_proj_b
      // [3072,    N] - cur (in)
      // [ 768,    N] - cur (out)
      //
      // cur = proj_w*cur + proj_b
      // [768, N]
      cur = ggml_mul_mat(ctx0,
			 model->layers[il].c_mlp_proj_w,
			 cur);

      cur = ggml_add(ctx0,
		     ggml_repeat(ctx0, model->layers[il].c_mlp_proj_b, cur),
		     cur);
    }

    // input for next layer
    inpL = ggml_add(ctx0, cur, inpFF);
  }

  // norm
  {
    // [ 768, N]
    inpL = ggml_norm(ctx0, inpL, hparams.eps);

    // inpL = ln_f_g*inpL + ln_f_b
    // [ 768, N]
    inpL = ggml_add(ctx0,
		    ggml_mul(ctx0,
			     ggml_repeat(ctx0, model->ln_f_g, inpL),
			     inpL),
		    ggml_repeat(ctx0, model->ln_f_b, inpL));
  }

  // inpL = WTE * inpL
  // [ 768, 50257] - model.lm_head
  // [ 768, N]     - inpL
  inpL = ggml_mul_mat(ctx0, model->lm_head, inpL);

  // logits -> probs
  //  inpL = ggml_soft_max_inplace(ctx0, inpL);

  // run the computation
  ggml_build_forward_expand(gf, inpL);
  ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

  //if (n_past%100 == 0) {
  //  ggml_graph_print   (gf);
  //  ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
  //}

  //    embd_w.resize(n_vocab*N);
  //    memcpy(embd_w._data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

  // return result just for the last token
  fvec_copy_array(embd_w, (float *)ggml_get_data(inpL)+(n_vocab*(N-1)), n_vocab);

  if (*mem_per_token == 0) {
    *mem_per_token = ggml_used_mem(ctx0)/N;
  }
  ggml_free(ctx0);

  return true;
}

#include <nux/nux.h>

void _gpt2_init(void *unused)
{
  (void)unused;

  ggml_time_init();

  const int64_t t_main_start_us = ggml_time_us();

  struct gpt_params params;
  gpt_params_default(&params);

  params.prompt = "hello";
  params.n_threads = cpu_num() - 1;

  int64_t t_load_us = 0;

  struct vocab vocab;
  struct gpt2_model model;

  // Default values.
  model.hparams.n_vocab = 50257;
  model.hparams.n_ctx = 1024;
  model.hparams.n_embd = 768;
  model.hparams.n_head = 12;
  model.hparams.n_layer = 12;
  model.hparams.ftype = 1;
  model.hparams.eps = 1e-5f;

  // load the model
  {
    const int64_t t_start_us = ggml_time_us();

    if (!gpt2_model_load((void *)0xffff828000000000L, 251222425, &model, &vocab)) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model);
      return;
    }
    t_load_us = ggml_time_us() - t_start_us;
  }

  int n_past = 0;

  int64_t t_sample_us  = 0;
  int64_t t_predict_us = 0;

  int32_t *embd_inp = NULL;
  int embd_inp_count = 0;

  tokenize_words(&vocab, params.prompt, &embd_inp, &embd_inp_count);

#define MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
  params.n_predict = MIN(params.n_predict, model.hparams.n_ctx - embd_inp_count);
  printf("%s: prompt: '%s'\n", __func__, params.prompt);
  printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__,
	 embd_inp_count);
  for (int i = 0; i < MIN(8, embd_inp_count); i++)
    {
      printf("%d ", embd_inp[i]);
    }
  printf("\n\n");

  int32_t vect[4] = { 0, 1, 2, 3};
  struct fvec logits;
  size_t mem_per_token;

  fvec_init (&logits);

  gpt2_eval(&model, params.n_threads, 0, vect, 4, &logits, &mem_per_token);

  int32_t *embd = NULL;
  int embd_count = 0;

  for (size_t i = embd_count; i < embd_inp_count + params.n_predict; i++) {
    // predict
    if (embd_count > 0) {

      const int64_t t_start_us = ggml_time_us();

      if (!gpt2_eval(&model, params.n_threads, n_past, embd, embd_count, &logits, &mem_per_token)) {
	printf("Failed to predict\n");
	return;
      }
      t_predict_us += ggml_time_us() - t_start_us;
	    
    }

    n_past += embd_count;
    free(embd);
    embd = NULL;
    embd_count = 0;

    if (i >= embd_inp_count) {
      // sample next token
      const int   top_k = params.top_k;
      const float top_p = params.top_p;
      const float temp  = params.temp;

      const int n_vocab = model.hparams.n_vocab;

      int32_t id = 0;

      {
	static unsigned long rng = 0;
	const int64_t t_start_sample_us = ggml_time_us();
	id = gpt_sample_top_k_top_p(fvec_data(&logits) + fvec_size(&logits) - n_vocab, n_vocab, top_k, top_p, temp, &rng);
	t_sample_us += ggml_time_us() - t_start_sample_us;
      }
      embd = realloc(embd, (embd_count + 1) * sizeof(*embd));
      embd[embd_count++] = id;
    } else {
      // if here, it means we are still processing the input prompt
      for (size_t k = i; k < embd_inp_count; k++) {
	embd = realloc(embd, (embd_count + 1) * sizeof(int32_t));
	embd[embd_count++] = embd_inp[k];
	if ((int32_t)embd_count >= params.n_batch) {
	  break;
	}
      }
      i += embd_count - 1;
    }

    // display text
    for (int i = 0; i < embd_count; i++)
      {
	int32_t id = embd[i];
	vocab_print(&vocab, id);
    }

    // end of text token
    if (embd[embd_count] == 50256) {
      break;
    }

  }

  // report timing
  {
    const int64_t t_main_end_us = ggml_time_us();

    printf("\n\n");
    printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
    printf("%s:     load time = %ld us\n", __func__, t_load_us);
    printf("%s:   sample time = %ld us\n", __func__, t_sample_us);
    printf("%s:  predict time = %ld us / %ld us per token\n", __func__, t_predict_us, t_predict_us/n_past);
    printf("%s:    total time = %ld us\n", __func__, (t_main_end_us - t_main_start_us));
  }

  ggml_free(model.ctx_w);

}
