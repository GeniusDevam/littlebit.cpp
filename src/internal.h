#ifndef LITTLEBIT_INTERNAL_H
#define LITTLEBIT_INTERNAL_H

#include "littlebit.h"

#define LB_ROWS 4
#define LB_MAX_THREADS 32

typedef struct {
    size_t rows;
    size_t cols;
    size_t bytes;
    uint8_t *data;
} lb_matrix;

typedef struct {
    const lb_matrix *matrix;
    const float *x;
    const float *lut;
    float *y;
    size_t tokens;
} lb_job;

typedef void (*lb_tile)(const lb_job *, size_t);

int lb_matrix_init(lb_matrix *m, size_t rows, size_t cols, const uint32_t *words);
void lb_matrix_export(const lb_matrix *m, uint32_t *words);
void lb_make_lut(const float *x, size_t cols, float *lut);
void lb_scalar(const lb_job *job, size_t tile);
void lb_lut(const lb_job *job, size_t tile);
void lb_batched(const lb_job *job, size_t tile);
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define LB_X86 1
void lb_avx2(const lb_job *job, size_t tile);
void lb_avx512(const lb_job *job, size_t tile);
#endif

void lb_parallel(const lb_job *job, lb_tile run, unsigned threads);

#endif
