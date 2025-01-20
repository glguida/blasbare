#include <math.h>
#include <string.h>
#include "ggml.h"

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

    int n_threads = 2; // number of threads to perform some operations with multi-threading

    ggml_graph_compute_with_ctx(model->ctx, gf, n_threads);

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}

struct simple_model model;

void
start_simple (void)
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

  load_model(&model, matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B);
  struct ggml_tensor * result = compute(&model);
  const float  *data = (const float *)ggml_get_data(result);
  int cols = result->ne[0];
  int rows = result->ne[1];
  printf("mul mat (%d x %d) (%d %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1], rows, cols);
  for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
	{
	  float v = data[i * cols + j];
	  printf ("%+2d.%02d ", (int)v, (int)((v - (int)v) * 100));
	}
      printf("\n");
    }
}

void _simple_init (void *arg)
{
  (void)arg;
  start_simple();
}


