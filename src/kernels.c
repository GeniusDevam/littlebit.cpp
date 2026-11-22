#include "internal.h"

#include <stdlib.h>

#if LB_X86
#include <immintrin.h>
#endif

int lb_matrix_init(lb_matrix *m, size_t rows, size_t cols, const uint32_t *words) {
    const size_t bytes = (cols + 7) / 8;
    const size_t stride = (cols + 31) / 32;
    const size_t tiles = (rows + LB_ROWS - 1) / LB_ROWS;
    if (!words || tiles > SIZE_MAX / LB_ROWS / bytes) {
        return 0;
    }
    uint8_t *data = calloc(tiles * bytes, LB_ROWS);
    if (!data) {
        return 0;
    }
    const unsigned tail = cols % 32;
    for (size_t row = 0; row < rows; ++row) {
        const uint32_t *source = words + row * stride;
        if (tail && (source[stride - 1] >> tail)) {
            free(data);
            return 0;
        }
        for (size_t b = 0; b < bytes; ++b) {
            data[((row / LB_ROWS) * bytes + b) * LB_ROWS + row % LB_ROWS] =
                (uint8_t)(source[b / 4] >> (8 * (b % 4)));
        }
    }
    *m = (lb_matrix){rows, cols, bytes, data};
    return 1;
}

void lb_matrix_export(const lb_matrix *m, uint32_t *words) {
    const size_t stride = (m->cols + 31) / 32;
    for (size_t row = 0; row < m->rows; ++row) {
        uint32_t *target = words + row * stride;
        for (size_t word = 0; word < stride; ++word) {
            target[word] = 0;
        }
        for (size_t b = 0; b < m->bytes; ++b) {
            uint32_t byte = m->data[((row / LB_ROWS) * m->bytes + b) * LB_ROWS + row % LB_ROWS];
            target[b / 4] |= byte << (8 * (b % 4));
        }
    }
}

static size_t tile_rows(const lb_matrix *m, size_t tile) {
    const size_t left = m->rows - tile * LB_ROWS;
    return left < LB_ROWS ? left : LB_ROWS;
}

static float signed_value(float x, unsigned bits, unsigned bit) {
    return ((bits >> bit) & 1) ? -x : x;
}

void lb_scalar(const lb_job *job, size_t tile) {
    const lb_matrix *m = job->matrix;
    const uint8_t *bits = m->data + tile * m->bytes * LB_ROWS;
    for (size_t t = 0; t < job->tokens; ++t) {
        const float *x = job->x + t * m->cols;
        float *y = job->y + t * m->rows + tile * LB_ROWS;
        for (size_t r = 0; r < tile_rows(m, tile); ++r) {
            double sum = 0;
            for (size_t c = 0; c < m->cols; ++c) {
                sum += signed_value(x[c], bits[(c / 8) * LB_ROWS + r], c % 8);
            }
            y[r] = (float)sum;
        }
    }
}

void lb_make_lut(const float *x, size_t cols, float *lut) {
    for (size_t c = 0; c < cols; c += 4, lut += 16) {
        const float a = x[c];
        const float b = c + 1 < cols ? x[c + 1] : 0;
        const float d = c + 2 < cols ? x[c + 2] : 0;
        const float e = c + 3 < cols ? x[c + 3] : 0;
        for (unsigned i = 0; i < 16; ++i) {
            lut[i] = (signed_value(a, i, 0) + signed_value(b, i, 1)) +
                     (signed_value(d, i, 2) + signed_value(e, i, 3));
        }
    }
}

void lb_lut(const lb_job *job, size_t tile) {
    const lb_matrix *m = job->matrix;
    const uint8_t *bits = m->data + tile * m->bytes * LB_ROWS;
    const size_t groups = (m->cols + 3) / 4;
    for (size_t t = 0; t < job->tokens; ++t) {
        const float *lut = job->lut + t * groups * 16;
        float sum[LB_ROWS] = {0};
        for (size_t c = 0; c < groups; ++c) {
            for (size_t r = 0; r < LB_ROWS; ++r) {
                const unsigned mask = bits[(c / 2) * LB_ROWS + r] >> (4 * (c % 2));
                sum[r] += lut[c * 16 + (mask & 15)];
            }
        }
        float *y = job->y + t * m->rows + tile * LB_ROWS;
        for (size_t r = 0; r < tile_rows(m, tile); ++r) {
            y[r] = sum[r];
        }
    }
}

void lb_batched(const lb_job *job, size_t tile) {
    enum { COLS = 64, TOKENS = 8 };
    const lb_matrix *m = job->matrix;
    const uint8_t *bits = m->data + tile * m->bytes * LB_ROWS;
    const size_t rows = tile_rows(m, tile);
    for (size_t t0 = 0; t0 < job->tokens; t0 += TOKENS) {
        const size_t tokens = job->tokens - t0 < TOKENS ? job->tokens - t0 : TOKENS;
        float sum[TOKENS][LB_ROWS] = {{0}};
        for (size_t c0 = 0; c0 < m->cols; c0 += COLS) {
            const size_t cols = m->cols - c0 < COLS ? m->cols - c0 : COLS;
            int8_t signs[LB_ROWS][COLS];
            for (size_t c = 0; c < cols; ++c) {
                for (size_t r = 0; r < rows; ++r) {
                    const unsigned byte = bits[((c0 + c) / 8) * LB_ROWS + r];
                    signs[r][c] = 1 - 2 * (int)((byte >> (c % 8)) & 1);
                }
            }
            for (size_t t = 0; t < tokens; ++t) {
                const float *x = job->x + (t0 + t) * m->cols + c0;
                for (size_t r = 0; r < rows; ++r) {
                    float part = 0;
                    for (size_t c = 0; c < cols; ++c) {
                        part += x[c] * signs[r][c];
                    }
                    sum[t][r] += part;
                }
            }
        }
        for (size_t t = 0; t < tokens; ++t) {
            float *y = job->y + (t0 + t) * m->rows + tile * LB_ROWS;
            for (size_t r = 0; r < rows; ++r) {
                y[r] = sum[t][r];
            }
        }
    }
}

#if LB_X86
__attribute__((target("avx2"))) static inline void avx2_tokens(const lb_job *job, size_t tile,
                                                               size_t first, size_t tokens) {
    const lb_matrix *m = job->matrix;
    const uint8_t *bits = m->data + tile * m->bytes * LB_ROWS;
    const __m256i shifts = _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24);
    const __m256i sign = _mm256_set1_epi32(INT32_MIN);
    const float *x0 = job->x + first * m->cols;
    __m256 sum[2][LB_ROWS];
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t r = 0; r < LB_ROWS; ++r) {
            sum[t][r] = _mm256_setzero_ps();
        }
    }
    const size_t full = m->cols / 8 * 8;
    for (size_t c = 0; c < full; c += 8) {
        __m256i inputs[2];
        for (size_t t = 0; t < tokens; ++t) {
            inputs[t] = _mm256_castps_si256(_mm256_loadu_ps(x0 + t * m->cols + c));
        }
        for (size_t r = 0; r < LB_ROWS; ++r) {
            const __m256i w = _mm256_set1_epi32(bits[(c / 8) * LB_ROWS + r]);
            const __m256i mask = _mm256_and_si256(_mm256_sllv_epi32(w, shifts), sign);
            for (size_t t = 0; t < tokens; ++t) {
                const __m256 v = _mm256_castsi256_ps(_mm256_xor_si256(inputs[t], mask));
                sum[t][r] = _mm256_add_ps(sum[t][r], v);
            }
        }
    }
    for (size_t t = 0; t < tokens; ++t) {
        const float *x = x0 + t * m->cols;
        float *y = job->y + (first + t) * m->rows + tile * LB_ROWS;
        for (size_t r = 0; r < tile_rows(m, tile); ++r) {
            __m128 pair =
                _mm_add_ps(_mm256_castps256_ps128(sum[t][r]), _mm256_extractf128_ps(sum[t][r], 1));
            pair = _mm_hadd_ps(pair, pair);
            pair = _mm_hadd_ps(pair, pair);
            float value = _mm_cvtss_f32(pair);
            for (size_t c = full; c < m->cols; ++c) {
                value += signed_value(x[c], bits[(c / 8) * LB_ROWS + r], c % 8);
            }
            y[r] = value;
        }
    }
}

__attribute__((target("avx2"))) void lb_avx2(const lb_job *job, size_t tile) {
    size_t t = 0;
    for (; job->tokens - t >= 2; t += 2) {
        avx2_tokens(job, tile, t, 2);
    }
    if (t < job->tokens) {
        avx2_tokens(job, tile, t, 1);
    }
}

__attribute__((target("avx512f"))) void lb_avx512(const lb_job *job, size_t tile) {
    const lb_matrix *m = job->matrix;
    const uint8_t *bits = m->data + tile * m->bytes * LB_ROWS;
    const __m512i sign = _mm512_set1_epi32(INT32_MIN);
    for (size_t t = 0; t < job->tokens; ++t) {
        const float *x = job->x + t * m->cols;
        float *y = job->y + t * m->rows + tile * LB_ROWS;
        __m512 sum[LB_ROWS];
        for (size_t r = 0; r < LB_ROWS; ++r) {
            sum[r] = _mm512_setzero_ps();
        }
        const size_t full = m->cols / 16 * 16;
        for (size_t c = 0; c < full; c += 16) {
            const __m512i v = _mm512_castps_si512(_mm512_loadu_ps(x + c));
            for (size_t r = 0; r < LB_ROWS; ++r) {
                const size_t i = (c / 8) * LB_ROWS + r;
                const __mmask16 mask = bits[i] | ((unsigned)bits[i + LB_ROWS] << 8);
                const __m512i flipped = _mm512_mask_xor_epi32(v, mask, v, sign);
                sum[r] = _mm512_add_ps(sum[r], _mm512_castsi512_ps(flipped));
            }
        }
        for (size_t r = 0; r < tile_rows(m, tile); ++r) {
            float value = _mm512_reduce_add_ps(sum[r]);
            for (size_t c = full; c < m->cols; ++c) {
                value += signed_value(x[c], bits[(c / 8) * LB_ROWS + r], c % 8);
            }
            y[r] = value;
        }
    }
}
#endif
