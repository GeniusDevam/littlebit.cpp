#ifndef LITTLEBIT_H
#define LITTLEBIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lb_layer lb_layer;

typedef enum { LB_AUTO, LB_SCALAR, LB_AVX2, LB_AVX512, LB_LUT, LB_BATCHED } lb_kernel;

typedef enum { LB_OK, LB_INVALID, LB_NOMEM, LB_UNSUPPORTED } lb_status;

typedef struct {
    const uint32_t *u;
    const uint32_t *v;
    const float *h;
    const float *l;
    const float *g;
} lb_path;

// Copies all arrays. Packed words have zero row padding. Pass NULL for no residual or bias.
lb_layer *lb_layer_create(size_t in, size_t out, size_t rank, const lb_path *primary,
                          const lb_path *residual, const float *bias);
void lb_layer_free(lb_layer *layer);
// x and y are non-overlapping token-major float arrays. Returns a status.
lb_status lb_layer_forward(const lb_layer *layer, const float *x, size_t tokens, float *y);
lb_status lb_layer_forward_ex(const lb_layer *layer, const float *x, size_t tokens, float *y,
                              lb_kernel kernel, unsigned threads);
// The defaults for lb_layer_forward are auto and one thread. Zero selects the default thread count.
lb_status lb_set_kernel(lb_kernel kernel);
lb_status lb_set_threads(unsigned threads);
int lb_kernel_available(lb_kernel kernel);
// Pass buffers of the original packed sizes to recover the input layout.
lb_status lb_layer_export(const lb_layer *layer, int residual, uint32_t *u, uint32_t *v);

#ifdef __cplusplus
}
#endif

#endif
