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

/*
  This is is a GGML test.
*/

#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>

int test0_main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * t1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 10);
    struct ggml_tensor * t2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_I16, 10, 20);
    struct ggml_tensor * t3 = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, 10, 20, 30);

    GGML_ASSERT(ggml_n_dims(t1) == 1);
    GGML_ASSERT(t1->ne[0]  == 10);
    GGML_ASSERT(t1->nb[1]  == 10*sizeof(float));

    GGML_ASSERT(ggml_n_dims(t2) == 2);
    GGML_ASSERT(t2->ne[0]  == 10);
    GGML_ASSERT(t2->ne[1]  == 20);
    GGML_ASSERT(t2->nb[1]  == 10*sizeof(int16_t));
    GGML_ASSERT(t2->nb[2]  == 10*20*sizeof(int16_t));

    GGML_ASSERT(ggml_n_dims(t3) == 3);
    GGML_ASSERT(t3->ne[0]  == 10);
    GGML_ASSERT(t3->ne[1]  == 20);
    GGML_ASSERT(t3->ne[2]  == 30);
    GGML_ASSERT(t3->nb[1]  == 10*sizeof(int32_t));
    GGML_ASSERT(t3->nb[2]  == 10*20*sizeof(int32_t));
    GGML_ASSERT(t3->nb[3]  == 10*20*30*sizeof(int32_t));

    ggml_print_objects(ctx0);

    ggml_free(ctx0);

    return 0;
}

void
_test0_init(void *unused)
{
  (void)unused;

  test0_main(0, NULL);
}
