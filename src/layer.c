#define _POSIX_C_SOURCE 200809L

#include "internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    lb_matrix u;
    lb_matrix v;
    float *h;
    float *l;
    float *g;
} lb_weights;

struct lb_layer {
    size_t in;
    size_t out;
    size_t rank;
    lb_weights paths[2];
    size_t count;
    float *bias;
};

static atomic_int default_kernel = LB_AUTO;
static atomic_uint default_threads = 1;

static float *floats(size_t width, size_t tokens) {
    if (!width || tokens > SIZE_MAX / sizeof(float) / width) {
        return NULL;
    }
    return malloc(width * tokens * sizeof(float));
}

static float *copy(const float *x, size_t size) {
    float *result = x ? floats(size, 1) : NULL;
    if (result) {
        memcpy(result, x, size * sizeof(float));
    }
    return result;
}

static int make_path(lb_weights *w, const lb_path *p, size_t in, size_t out, size_t rank) {
    w->g = copy(p->g, in);
    w->l = copy(p->l, rank);
    w->h = copy(p->h, out);
    return w->g && w->l && w->h && lb_matrix_init(&w->v, rank, in, p->v) &&
           lb_matrix_init(&w->u, out, rank, p->u);
}

lb_layer *lb_layer_create(size_t in, size_t out, size_t rank, const lb_path *primary,
                          const lb_path *residual, const float *bias) {
    if (!in || !out || !rank || in > INT32_MAX || out > INT32_MAX || rank > INT32_MAX || !primary) {
        return NULL;
    }
    lb_layer *layer = calloc(1, sizeof(*layer));
    if (!layer) {
        return NULL;
    }
    layer->in = in;
    layer->out = out;
    layer->rank = rank;
    layer->count = residual ? 2 : 1;
    layer->bias = copy(bias, out);
    if ((bias && !layer->bias) || !make_path(layer->paths, primary, in, out, rank) ||
        (residual && !make_path(layer->paths + 1, residual, in, out, rank))) {
        lb_layer_free(layer);
        return NULL;
    }
    return layer;
}

void lb_layer_free(lb_layer *layer) {
    if (!layer) {
        return;
    }
    for (size_t p = 0; p < layer->count; ++p) {
        lb_weights *w = layer->paths + p;
        free(w->u.data);
        free(w->v.data);
        free(w->h);
        free(w->l);
        free(w->g);
    }
    free(layer->bias);
    free(layer);
}

lb_status lb_layer_export(const lb_layer *layer, int residual, uint32_t *u, uint32_t *v) {
    if (!layer || residual < 0 || (size_t)residual >= layer->count || !u || !v) {
        return LB_INVALID;
    }
    lb_matrix_export(&layer->paths[residual].u, u);
    lb_matrix_export(&layer->paths[residual].v, v);
    return LB_OK;
}

int lb_kernel_available(lb_kernel kernel) {
    switch (kernel) {
        case LB_AUTO:
        case LB_SCALAR:
        case LB_LUT:
        case LB_BATCHED:
            return 1;
#if LB_X86
        case LB_AVX2:
            return __builtin_cpu_supports("avx2") != 0;
        case LB_AVX512:
            return __builtin_cpu_supports("avx512f") != 0;
#endif
        default:
            return 0;
    }
}

lb_status lb_set_kernel(lb_kernel kernel) {
    if (!lb_kernel_available(kernel)) {
        return LB_UNSUPPORTED;
    }
    atomic_store(&default_kernel, kernel);
    return LB_OK;
}

lb_status lb_set_threads(unsigned threads) {
    if (threads > LB_MAX_THREADS) {
        return LB_INVALID;
    }
    atomic_store(&default_threads, threads ? threads : 1);
    return LB_OK;
}

static lb_status matmul(lb_job *job, lb_kernel kernel, unsigned threads) {
    switch (kernel) {
        case LB_SCALAR:
            lb_parallel(job, lb_scalar, threads);
            break;
#if LB_X86
        case LB_AVX2:
            lb_parallel(job, lb_avx2, threads);
            break;
        case LB_AVX512:
            lb_parallel(job, lb_avx512, threads);
            break;
#endif
        case LB_LUT: {
            const size_t cols = job->matrix->cols;
            const size_t stride = ((cols + 3) / 4) * 16;
            float *table = floats(stride, job->tokens);
            if (!table) {
                return LB_NOMEM;
            }
            for (size_t t = 0; t < job->tokens; ++t) {
                lb_make_lut(job->x + t * cols, cols, table + t * stride);
            }
            job->lut = table;
            lb_parallel(job, lb_lut, threads);
            job->lut = NULL;
            free(table);
            break;
        }
        case LB_BATCHED:
            lb_parallel(job, lb_batched, threads);
            break;
        default:
            return LB_UNSUPPORTED;
    }
    return LB_OK;
}

typedef struct {
    size_t rows, cols, tokens;
    unsigned threads;
    lb_kernel kernel;
} lb_choice;

static lb_choice choices[128];
static size_t next_choice;
static pthread_mutex_t tune_lock = PTHREAD_MUTEX_INITIALIZER;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static lb_kernel tune(lb_job *job, unsigned threads) {
    const size_t rows = job->matrix->rows;
    const size_t cols = job->matrix->cols;
    const size_t tokens = job->tokens;
    pthread_mutex_lock(&tune_lock);
    for (size_t i = 0; i < sizeof(choices) / sizeof(choices[0]); ++i) {
        const lb_choice *choice = choices + i;
        if (choice->rows == rows && choice->cols == cols && choice->tokens == tokens &&
            choice->threads == threads) {
            const lb_kernel selected = choice->kernel;
            pthread_mutex_unlock(&tune_lock);
            return selected;
        }
    }
    lb_kernel fastest = LB_BATCHED;
    double best = -1;
    for (lb_kernel k = LB_AVX2; k <= LB_BATCHED; ++k) {
        if (!lb_kernel_available(k) || (k == LB_BATCHED && tokens == 1 && best >= 0)) {
            continue;
        }
        // Warm up the worker pool and time the full stage, including lookup-table construction.
        if (matmul(job, k, threads) != LB_OK) {
            continue;
        }
        const double start = now();
        if (matmul(job, k, threads) != LB_OK) {
            continue;
        }
        const double elapsed = now() - start;
        if (best < 0 || elapsed < best) {
            best = elapsed;
            fastest = k;
        }
    }
    choices[next_choice] = (lb_choice){rows, cols, tokens, threads, fastest};
    next_choice = (next_choice + 1) % (sizeof(choices) / sizeof(choices[0]));
    pthread_mutex_unlock(&tune_lock);
    return fastest;
}

static lb_status stage(lb_job *job, lb_kernel kernel, unsigned threads) {
    if (kernel == LB_AUTO) {
        if ((double)job->matrix->rows * job->matrix->cols * job->tokens < 65536) {
            threads = 1;
        }
        kernel = tune(job, threads);
        const lb_status status = matmul(job, kernel, threads);
        if (kernel == LB_LUT && status == LB_NOMEM) {
            return matmul(job, LB_BATCHED, threads);
        }
        return status;
    }
    return matmul(job, kernel, threads);
}

lb_status lb_layer_forward_ex(const lb_layer *layer, const float *x, size_t tokens, float *y,
                              lb_kernel kernel, unsigned threads) {
    if (!layer || !x || !y || threads > LB_MAX_THREADS) {
        return LB_INVALID;
    }
    if (!lb_kernel_available(kernel)) {
        return LB_UNSUPPORTED;
    }
    if (!tokens) {
        return LB_OK;
    }
    if (!threads) {
        threads = atomic_load(&default_threads);
    }
    const size_t in = layer->in, out = layer->out, rank = layer->rank;
    if (tokens > SIZE_MAX / sizeof(float) / out) {
        return LB_INVALID;
    }
    float *scaled = floats(in, tokens);
    float *latent = floats(rank, tokens);
    float *residual = layer->count == 2 ? floats(out, tokens) : NULL;
    lb_status status = LB_NOMEM;
    if (!scaled || !latent || (layer->count == 2 && !residual)) {
        goto end;
    }

    for (size_t p = 0; p < layer->count; ++p) {
        const lb_weights *w = layer->paths + p;
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t i = 0; i < in; ++i) {
                scaled[t * in + i] = x[t * in + i] * w->g[i];
            }
        }
        lb_job first = {&w->v, scaled, NULL, latent, tokens};
        status = stage(&first, kernel, threads);
        if (status != LB_OK) {
            goto end;
        }
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t i = 0; i < rank; ++i) {
                latent[t * rank + i] *= w->l[i];
            }
        }
        float *target = p ? residual : y;
        lb_job second = {&w->u, latent, NULL, target, tokens};
        status = stage(&second, kernel, threads);
        if (status != LB_OK) {
            goto end;
        }
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t i = 0; i < out; ++i) {
                const size_t j = t * out + i;
                target[j] *= w->h[i];
                if (p) {
                    y[j] += target[j];
                }
            }
        }
    }
    if (layer->bias) {
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t i = 0; i < out; ++i) {
                y[t * out + i] += layer->bias[i];
            }
        }
    }
end:
    free(scaled);
    free(latent);
    free(residual);
    return status;
}

lb_status lb_layer_forward(const lb_layer *layer, const float *x, size_t tokens, float *y) {
    return lb_layer_forward_ex(layer, x, tokens, y, atomic_load(&default_kernel), 0);
}
