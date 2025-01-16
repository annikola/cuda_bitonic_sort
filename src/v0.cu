#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/sorting_helpers.h"

#define MAX_THREADS 1024
#define MIN_ARGS 1
#define MIN_Q 1
#define MAX_Q 27

__global__ void mtlxchange(int *a, int jj, int kk) {

    int i, ij, tid, dummy;

    tid = blockIdx.x * blockDim.x + threadIdx.x;
    i = tid - 1;

    ij = i ^ jj;
    if (ij > i) {
        if ((i & kk) == 0 && a[i] > a[ij]) {
            dummy = a[i];
            a[i] = a[ij];
            a[ij] = dummy;
        }
        if ((i & kk) != 0 && a[i] < a[ij]) {
            dummy = a[i];
            a[i] = a[ij];
            a[ij] = dummy;
        }
    }
}

int main(int argc, char *argv[]) {

    int i, j, jj, k, kk, Q, A_size, blocks, threads;
    int *A, *B, *d_a;
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
    B = (int *)malloc(A_size * sizeof(int));
    A = (int *)malloc(A_size * sizeof(int));
    for (i = 0; i < A_size; i++) {
        A[i] = rand();
        B[i] = A[i];
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
        blocks = A_size / MAX_THREADS;
        threads = MAX_THREADS;
    }
    for (k = 1; k <= Q; k++) {
        kk = 1 << k;
        for (j = k - 1; j >= 0; j--) {
            jj = 1 << j;
            mtlxchange<<<blocks, threads>>>(d_a, jj, kk);
            cudaDeviceSynchronize();
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

    qsort(B, A_size, sizeof(int), asc_compare);
    if (array_compare(A, B, A_size)) {
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
