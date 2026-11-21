#include "littlebit.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr)                                                                                \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);                             \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static uint32_t state = 41;

static uint32_t random_word(void) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static uint32_t *bits(size_t rows, size_t cols) {
    size_t stride = (cols + 31) / 32;
    uint32_t *words = malloc(rows * stride * sizeof(*words));
    CHECK(words);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t w = 0; w < stride; ++w) {
            words[r * stride + w] = random_word();
        }
        if (cols % 32) {
            words[(r + 1) * stride - 1] &= (UINT32_C(1) << (cols % 32)) - 1;
        }
    }
    return words;
}

static float *values(size_t n, float scale) {
    float *x = malloc(n * sizeof(*x));
    CHECK(x);
    for (size_t i = 0; i < n; ++i) {
        x[i] = ((int)(random_word() % 2049) - 1024) * (scale / 1024);
    }
    return x;
}

static lb_path path(size_t in, size_t out, size_t rank) {
    return (lb_path){bits(out, rank), bits(rank, in), values(out, 1), values(rank, 0.05f),
                     values(in, 0.5f)};
}

static void free_path(lb_path p) {
    free((void *)p.u);
    free((void *)p.v);
    free((void *)p.h);
    free((void *)p.l);
    free((void *)p.g);
}

static void test_packing(void) {
    const uint32_t u[] = {0, 1};
    const uint32_t v[] = {UINT32_C(0x80000001), 1};
    float g[33], x[33] = {0};
    for (size_t i = 0; i < 33; ++i) {
        g[i] = 1;
    }
    x[0] = 1;
    x[1] = 2;
    x[31] = 4;
    x[32] = 8;
    const float h[] = {2, -3}, l[] = {1}, bias[] = {1, 10};
    lb_path p = {u, v, h, l, g};
    lb_layer *layer = lb_layer_create(33, 2, 1, &p, &p, bias);
    CHECK(layer);
    uint32_t recovered_u[2], recovered_v[2];
    for (int residual = 0; residual <= 1; ++residual) {
        CHECK(lb_layer_export(layer, residual, recovered_u, recovered_v) == LB_OK);
        CHECK(memcmp(u, recovered_u, sizeof(u)) == 0);
        CHECK(memcmp(v, recovered_v, sizeof(v)) == 0);
    }
    for (lb_kernel k = LB_AUTO; k <= LB_BATCHED; ++k) {
        float y[2];
        if (lb_kernel_available(k)) {
            CHECK(lb_set_kernel(k) == LB_OK);
            CHECK(lb_set_threads(2) == LB_OK);
            CHECK(lb_layer_forward(layer, x, 1, y) == LB_OK);
            CHECK(y[0] == -43 && y[1] == -56);
        } else {
            CHECK(lb_layer_forward_ex(layer, x, 1, y, k, 1) == LB_UNSUPPORTED);
        }
    }
    CHECK(lb_set_threads(33) == LB_INVALID);
    CHECK(lb_set_threads(0) == LB_OK);
    CHECK(lb_set_kernel(LB_AUTO) == LB_OK);
    CHECK(lb_layer_export(layer, 2, recovered_u, recovered_v) == LB_INVALID);
    lb_layer_free(layer);

    const uint32_t bad_padding[] = {0, 2};
    p.v = bad_padding;
    CHECK(!lb_layer_create(33, 2, 1, &p, NULL, NULL));
    CHECK(!lb_layer_create(0, 2, 1, &p, NULL, NULL));
}

static void test_shape(size_t in, size_t out, size_t rank, size_t tokens, int residual, int bias) {
    lb_path p = path(in, out, rank);
    lb_path r = path(in, out, rank);
    float *b = bias ? values(out, 1) : NULL;
    lb_layer *layer = lb_layer_create(in, out, rank, &p, residual ? &r : NULL, b);
    CHECK(layer);
    free_path(p);
    free_path(r);
    free(b);

    float *x = values(in * tokens, 1);
    float *reference = values(out * tokens, 0);
    float *got = values(out * tokens, 0);
    CHECK(lb_layer_forward_ex(layer, x, tokens, reference, LB_SCALAR, 1) == LB_OK);
    const unsigned thread_counts[] = {1, 2, 4, 8};
    for (lb_kernel k = LB_AUTO; k <= LB_BATCHED; ++k) {
        if (!lb_kernel_available(k)) {
            continue;
        }
        for (size_t t = 0; t < sizeof(thread_counts) / sizeof(thread_counts[0]); ++t) {
            CHECK(lb_layer_forward_ex(layer, x, tokens, got, k, thread_counts[t]) == LB_OK);
            for (size_t i = 0; i < out * tokens; ++i) {
                const float expected = reference[i];
                if (!(fabsf(got[i] - expected) <= 3e-5f * (1 + fabsf(expected)))) {
                    fprintf(stderr, "kernel %d shape %zu,%zu,%zu,%zu threads %u got %g want %g\n",
                            k, in, out, rank, tokens, thread_counts[t], got[i], expected);
                    exit(1);
                }
            }
        }
    }
    lb_layer_free(layer);
    free(got);
    free(reference);
    free(x);
}

static void test_thread_counts(void) {
    const size_t in = 35, out = 139, rank = 129, tokens = 3;
    lb_path p = path(in, out, rank);
    lb_layer *layer = lb_layer_create(in, out, rank, &p, NULL, NULL);
    CHECK(layer);
    free_path(p);

    float *x = values(in * tokens, 1);
    float expected[3 * 139], actual[3 * 139];
    for (lb_kernel k = LB_SCALAR; k <= LB_BATCHED; ++k) {
        if (!lb_kernel_available(k)) {
            continue;
        }
        CHECK(lb_layer_forward_ex(layer, x, tokens, expected, k, 1) == LB_OK);
        for (unsigned threads = 2; threads <= 32; ++threads) {
            CHECK(lb_layer_forward_ex(layer, x, tokens, actual, k, threads) == LB_OK);
            CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
        }
    }
    free(x);
    lb_layer_free(layer);
}

int main(void) {
    test_packing();
    test_thread_counts();
    const size_t ranks[] = {8, 24, 64, 128, 1025};
    const size_t tokens[] = {1, 3, 16, 64};
    for (size_t r = 0; r < sizeof(ranks) / sizeof(ranks[0]); ++r) {
        for (size_t n = 0; n < sizeof(tokens) / sizeof(tokens[0]); ++n) {
            for (int variant = 0; variant < 4; ++variant) {
                test_shape(variant & 1 ? 33 : 64, variant & 2 ? 35 : 32, ranks[r], tokens[n],
                           variant & 1, variant & 2);
            }
        }
    }
    printf("AVX2: %s, AVX-512F: %s\n", lb_kernel_available(LB_AVX2) ? "tested" : "unsupported",
           lb_kernel_available(LB_AVX512) ? "tested" : "unsupported");
    return 0;
}
