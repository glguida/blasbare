/*
  NUX: A kernel Library.
  Copyright (C) 2019 Gianluca Guida, glguida@tlbflush.org

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  See COPYING file for the full license.

  SPDX-License-Identifier:	GPL2.0+
*/

#include <assert.h>
#include <stdio.h>
#include <nux/nux.h>

#include <nux/hal.h>

uctxt_t u_init;
struct hal_umap umap;

#include "ggml.h"
#include <math.h>

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;

    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context * ctx;
};

// initialize the tensors of the model in this case two matrices 2x2
void load_model(struct simple_model *model, float * a, float * b, int rows_A, int cols_A, int rows_B, int cols_B) {
    size_t ctx_size = 0;
    {
        ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32); // tensor a
        ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32); // tensor b
        ctx_size += 2 * ggml_tensor_overhead(), // tensors
        ctx_size += ggml_graph_overhead(); // compute graph
        ctx_size += 1024; // some overhead
    }

    struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };

    // create context
    model->ctx = ggml_init(params);

    // create tensors
    model->a = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, cols_A, rows_A);
    model->b = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, cols_B, rows_B);

    memcpy(model->a->data, a, ggml_nbytes(model->a));
    memcpy(model->b->data, b, ggml_nbytes(model->b));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(const struct simple_model *model) {
    struct ggml_cgraph  * gf = ggml_new_graph(model->ctx);

    // result = a*b^T
    struct ggml_tensor * result = ggml_mul_mat(model->ctx, model->a, model->b);

    ggml_build_forward_expand(gf, result);
    return gf;
}

// compute with backend
struct ggml_tensor * compute(const struct simple_model *model) {
    struct ggml_cgraph * gf = build_graph(model);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    printf("Computing!\n");
    ggml_graph_compute_with_ctx(model->ctx, gf, n_threads);
    printf("Done!\n");

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}

struct simple_model model;

int
main (int argc, char *argv[])
{

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;

    float matrix_A[4 * 2] = {
        2.0, 8.0,
        5.0, 1.0,
        4.0, 2.0,
        8.0, 6.0
    };

    const int rows_B = 3, cols_B = 2;
    /* Transpose([
        10, 9, 5,
        5, 9, 4
    ]) 2 rows, 3 cols */
    float matrix_B[3 * 2] = {
        10.0, 5.0,
        9.0, 9.0,
        5.0, 4.0
    };
  
  /*
    Start AVX and SSE. (INTEL SPECIFIC)
  */

  printf("Enabling SSE/AVX...");
  asm volatile (
		"mov %%cr0, %%rax\n"
		"and $~0x8, %%rax\n" /* Clear TS in case. */
		"and $~0x4, %%rax\n" /* Clear EM in case. */
		"mov %%rax, %%cr0\n"

		"mov %%cr0, %%rax\n"
		"or $0x2, %%rax\n"  /* Set MP */
		"mov %%rax, %%cr0\n"

		"mov %%cr4, %%rax\n"
		"or $0x200, %%rax\n"   /* Set OSFXSR */
		"or $0x400, %%rax\n"   /* Set OSXMMEXCPT */
		"or $0x40000, %%rax\n" /* Set OSXSAVE */
		"mov %%rax, %%cr4\n"

		"xor %%ecx, %%ecx\n" /* XCR0 */
		"mov $0x7, %%eax\n"  /* Enable x87, SSE, AVX */
		"xor %%edx, %%edx\n" /* Upper bits empty. */
		"xsetbv\n"
		::: "rax", "rcx", "rdx");
  printf("done\n");

  load_model(&model, matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B);
  struct ggml_tensor * result = compute(&model);
  printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);

  return EXIT_IDLE;
}

int
main_ap (void)
{
  printf ("%d: %" PRIx64 "\n", cpu_id (), timer_gettime ());
  return EXIT_IDLE;
}

uctxt_t *
entry_sysc (uctxt_t * u,
	    unsigned long a1, unsigned long a2, unsigned long a3,
	    unsigned long a4, unsigned long a5, unsigned long a6,
	    unsigned long a7)
{
  switch (a1)
    {
    case 0:
      info("SYSC%ld test passed.", a1);
      break;
    case 1:
      assert(a2 == 1);
      info("SYSC%ld test passed.", a1);
      break;
    case 2:
      assert(a2 == 1);
      assert(a3 == 2);
      info("SYSC%ld test passed.", a1);
      break;
    case 3:
      assert(a2 == 1);
      assert(a3 == 2);
      assert(a4 == 3);
      info("SYSC%ld test passed.", a1);
      break;
    case 4:
      assert(a2 == 1);
      assert(a3 == 2);
      assert(a4 == 3);
      assert(a5 == 4);
      info("SYSC%ld test passed.", a1);
      break;
    case 5:
      assert(a2 == 1);
      assert(a3 == 2);
      assert(a4 == 3);
      assert(a5 == 4);
      assert(a6 == 5);
      info("SYSC%ld test passed.", a1);
      break;
    case 6:
      assert(a2 == 1);
      assert(a3 == 2);
      assert(a4 == 3);
      assert(a5 == 4);
      assert(a6 == 5);
      assert(a7 == 6);
      info("SYSC%ld test passed.", a1);
      break;
    case 4096:
      putchar (a2);
      break;
    case 4097:
      info ("User exited with error code: %ld", a2);
      hal_umap_load (NULL);
      hal_umap_free (&umap);
      return UCTXT_IDLE;

    default:
      info ("Received unknown syscall %ld %ld %ld %ld %ld %ld %ld\n",
	    a1, a2, a3, a4, a5, a6, a7);
      break;
    }
  return u;
}

uctxt_t *
entry_ipi (uctxt_t * uctxt)
{
  info ("IPI!");
  return &u_init;
}

uctxt_t *
entry_alarm (uctxt_t * uctxt)
{
  timer_alarm (1 * 1000 * 1000 * 1000);
  info ("TMR: %" PRIu64 " us", timer_gettime ());
  uctxt_print (uctxt);
  return uctxt;
}

uctxt_t *
entry_ex (uctxt_t * uctxt, unsigned ex)
{
  info ("Exception %d", ex);
  uctxt_print (uctxt);
  return UCTXT_IDLE;
}

uctxt_t *
entry_pf (uctxt_t * uctxt, vaddr_t va, hal_pfinfo_t pfi)
{
  info ("CPU #%d Pagefault at %08lx (%d)", cpu_id (), va, pfi);
  uctxt_print (uctxt);
  return UCTXT_IDLE;
}

uctxt_t *
entry_irq (uctxt_t * uctxt, unsigned irq, bool lvl)
{
  info ("IRQ %d", irq);
  return uctxt;

}
