#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/sorting_helpers.h"

#define MAX_THREADS 1024
#define MAX_LOCAL_ELEMENTS 2048
#define MIN_ARGS 1
#define MIN_Q 11
#define MAX_Q 27

int isAscending(int *A, int n);

__global__ void external_exchanges(int *a, int j, int k) {

    int i, jj, jjj, minmax, tid, dummy, total_threads, total_blocks;

    total_threads = blockDim.x * blockDim.y * blockDim.z;
    total_blocks = gridDim.x * gridDim.y * gridDim.z;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    jj = 1 << j;
    jjj = 2 << k;
    if ((tid & jj) != 0) {
        i = tid + total_threads * total_blocks - jj;
    } else {
        i = tid;
    }
    minmax = i & jjj;
    if (minmax == 0 && a[i] > a[i + jj]) {
        dummy = a[i];
        a[i] = a[i + jj];
        a[i + jj] = dummy;
    }
    if (minmax != 0 && a[i] < a[i + jj]) {
        dummy = a[i];
        a[i] = a[i + jj];
        a[i + jj] = dummy;
    }
}

__global__ void internal_exchanges(int *a, int k, int flow) {

    __shared__ int local_elements[MAX_LOCAL_ELEMENTS];
    int i, i_mod, j, jj, jjj, minmax, ltid, tid, dummy;

    tid = blockIdx.x * blockDim.x + threadIdx.x;
    ltid = threadIdx.x;

    // Initialize the shared memory inside the block (each thread reads two elements...)
    local_elements[ltid] = a[tid + MAX_THREADS * blockIdx.x]; // tid & ((1 << k) − 1)
    local_elements[ltid + MAX_THREADS] = a[tid + MAX_THREADS * (blockIdx.x + 1)];
    __syncthreads();

    for (j = k - 1; j >= 0; j--) {
        jj = 1 << j;
        jjj = 2 << flow;
        if ((tid & jj) != 0) {
            i = tid + MAX_THREADS * (blockIdx.x + 1) - jj;
        } else {
            i = tid + MAX_THREADS * blockIdx.x;
        }
        minmax = i & jjj;
        i_mod = i & ((1 << 11) - 1); // i_mod = i % MAX_LOCAL_ELEMENTS;
        if (minmax == 0 && local_elements[i_mod] > local_elements[i_mod + jj]) {
            dummy = local_elements[i_mod];
            local_elements[i_mod] = local_elements[i_mod + jj];
            local_elements[i_mod + jj] = dummy;
        }
        if (minmax != 0 && local_elements[i_mod] < local_elements[i_mod + jj]) {
            dummy = local_elements[i_mod];
            local_elements[i_mod] = local_elements[i_mod + jj];
            local_elements[i_mod + jj] = dummy;
        }
        __syncthreads();
    }

    a[tid + MAX_THREADS * blockIdx.x] = local_elements[ltid];
    a[tid + MAX_THREADS * (blockIdx.x + 1)] = local_elements[ltid + MAX_THREADS];
}

__global__ void prephase_exchanges(int *a) {

    __shared__ int local_elements[MAX_LOCAL_ELEMENTS];
    int i, i_mod, k, j, jj, jjj, minmax, ltid, tid, dummy;

    tid = blockIdx.x * blockDim.x + threadIdx.x;
    ltid = threadIdx.x;

    // Initialize the shared memory inside the block (each thread reads two elements...)
    local_elements[ltid] = a[tid + MAX_THREADS * blockIdx.x]; // tid & ((1 << k) − 1)
    local_elements[ltid + MAX_THREADS] = a[tid + MAX_THREADS * (blockIdx.x + 1)];
    __syncthreads();

    for (k = 0; k < 11; k++) {
        for (j = k; j >= 0; j--) {
            jj = 1 << j;
            jjj = 2 << k;
            if ((tid & jj) != 0) {
                i = tid + MAX_THREADS * (blockIdx.x + 1) - jj;
            } else {
                i = tid + MAX_THREADS * blockIdx.x;
            }
            minmax = i & jjj;
            i_mod = i & ((1 << 11) - 1); // i_mod = i % MAX_LOCAL_ELEMENTS;
            if (minmax == 0 && local_elements[i_mod] > local_elements[i_mod + jj]) {
                dummy = local_elements[i_mod];
                local_elements[i_mod] = local_elements[i_mod + jj];
                local_elements[i_mod + jj] = dummy;
            }
            if (minmax != 0 && local_elements[i_mod] < local_elements[i_mod + jj]) {
                dummy = local_elements[i_mod];
                local_elements[i_mod] = local_elements[i_mod + jj];
                local_elements[i_mod + jj] = dummy;
            }
            __syncthreads();
        }
    }

    a[tid + MAX_THREADS * blockIdx.x] = local_elements[ltid];
    a[tid + MAX_THREADS * (blockIdx.x + 1)] = local_elements[ltid + MAX_THREADS];
}

int main(int argc, char *argv[]) {

    int i, j, k, Q, A_size, blocks, threads;
    int *A, *d_a;
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaError_t err;

    if (argc < MIN_ARGS + 1) {
        printf("Missing %d argument(s)\n", MIN_ARGS + 1 - argc);
        return 1;
    }

    Q = atoi(argv[1]);
    if (Q < MIN_Q || Q > MAX_Q) {
        printf("Please insert a value for Q between %d and %d\n", MIN_Q, MAX_Q);
        return 1;
    }

    A_size = 1 << Q;
    A = (int *)malloc(A_size * sizeof(int));
    for (i = 0; i < A_size; i++) {
        A[i] = rand();
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); // Start the timing...

    cudaMalloc((void **)&d_a, A_size * sizeof(int));

    err = cudaMemcpy(d_a, A, A_size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error during cudaMemcpy (d_a): %s\n", cudaGetErrorString(err));
    }

    if (A_size < MAX_THREADS) {
        blocks = 1;
        threads = A_size;
    } else {
        blocks = (A_size / MAX_THREADS) / 2;
        threads = MAX_THREADS;
    }

    prephase_exchanges<<<blocks, threads>>>(d_a);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error during prephase_exchanges: %s\n", cudaGetErrorString(err));
    }

    for (k = 11; k < Q; k++) {
        for (j = k; j > 10; j--) {
            external_exchanges<<<blocks, threads>>>(d_a, j, k);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error during external_exchanges: %s\n", cudaGetErrorString(err));
            }
        }
        internal_exchanges<<<blocks, threads>>>(d_a, 11, k);
        if (err != cudaSuccess) {
            printf("CUDA error during internal_exchanges: %s\n", cudaGetErrorString(err));
        }
    }

    err = cudaMemcpy(A, d_a, A_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error during cudaMemcpy (B_a): %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop, 0); // Stop the timing...
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Total execution time: %f ms\n", elapsed_time);

    // Check the validity of the results
    if (isAscending(A, A_size)) {
        printf("Correctly sorted!\n");
    } else {
        printf("Falsely sorted!\n");
    }

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);

    return 0;
}

int isAscending(int *A, int n) {

    int i;

    for (i = 0; i < n - 1; i++) {
        if (A[i] > A[i + 1]) {
            return 0;
        }
    }

    return 1;
}
