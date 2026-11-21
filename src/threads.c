#include "internal.h"

#include <pthread.h>

static pthread_mutex_t submit = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t done = PTHREAD_COND_INITIALIZER;
static unsigned workers;
static unsigned active;
static unsigned remaining;
static size_t generation;
static const lb_job *current_job;
static lb_tile current_run;

static void run_rows(const lb_job *job, lb_tile run, unsigned count, unsigned id) {
    const size_t tiles = (job->matrix->rows + LB_ROWS - 1) / LB_ROWS;
    const size_t first = tiles * id / count;
    const size_t end = tiles * (id + 1) / count;
    for (size_t tile = first; tile < end; ++tile) {
        run(job, tile);
    }
}

static void *worker(void *arg) {
    const unsigned id = (unsigned)(uintptr_t)arg;
    size_t seen = 0;
    pthread_mutex_lock(&lock);
    for (;;) {
        while (seen == generation) {
            pthread_cond_wait(&ready, &lock);
        }
        seen = generation;
        if (id < active) {
            const lb_job *job = current_job;
            lb_tile run = current_run;
            const unsigned count = active;
            pthread_mutex_unlock(&lock);
            run_rows(job, run, count, id);
            pthread_mutex_lock(&lock);
            if (--remaining == 0) {
                pthread_cond_signal(&done);
            }
        }
    }
    return NULL;
}

void lb_parallel(const lb_job *job, lb_tile run, unsigned threads) {
    const size_t tiles = (job->matrix->rows + LB_ROWS - 1) / LB_ROWS;
    if (threads > tiles) {
        threads = (unsigned)tiles;
    }
    if (threads == 1) {
        run_rows(job, run, 1, 0);
        return;
    }

    pthread_mutex_lock(&submit);
    pthread_mutex_lock(&lock);
    while (workers + 1 < threads) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, worker, (void *)(uintptr_t)(workers + 1)) != 0) {
            threads = workers + 1;
            break;
        }
        pthread_detach(thread);
        ++workers;
    }
    active = threads;
    remaining = threads - 1;
    current_job = job;
    current_run = run;
    ++generation;
    pthread_cond_broadcast(&ready);
    pthread_mutex_unlock(&lock);

    run_rows(job, run, threads, 0);

    pthread_mutex_lock(&lock);
    while (remaining) {
        pthread_cond_wait(&done, &lock);
    }
    pthread_mutex_unlock(&lock);
    pthread_mutex_unlock(&submit);
}
