#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/sorting_helpers.h"

#define MAX_THREADS 1024
#define MIN_ARGS 1
#define MIN_Q 11
#define MAX_Q 27

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

    for (k = 0; k < Q; k++) {
        for (j = k; j >= 0; j--) {
            external_exchanges<<<blocks, threads>>>(d_a, j, k);
        }
    }

    err = cudaMemcpy(A, d_a, A_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error during cudaMemcpy (A_a): %s\n", cudaGetErrorString(err));
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
