#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/sorting_helpers.h"

#define MAX_THREADS 1024
#define MIN_ARGS 1
#define MIN_Q 1
#define MAX_Q 27

int isSortedAscending(int *arr, int size);
int isSortedDescending(int *arr, int size);

__global__ void external_exchanges(int *a, int kk, int kkk) {

    int minmax, tid, dummy;

    tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((tid & kk) == 0) {
        minmax = tid & kkk;
        if (minmax == 0 && a[tid] > a[tid + kk]) {
            dummy = a[tid];
            a[tid] = a[tid + kk];
            a[tid + kk] = dummy;
        }
        if (minmax != 0 && a[tid] < a[tid + kk]) {
            dummy = a[tid];
            a[tid] = a[tid + kk];
            a[tid + kk] = dummy;
        }
    }
}

__global__ void internal_exchanges(int *a, int k) {

    int j, jj, jjj, minmax, tid, dummy;

    tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (j = k - 1; j >= 0; j--) {
        jj = 1 << j;
        jjj = 2 << k;
        if ((tid & jj) == 0) {
            minmax = tid & jjj;
            if (minmax == 0 && a[tid] > a[tid + jj]) {
                dummy = a[tid];
                a[tid] = a[tid + jj];
                a[tid + jj] = dummy;
            }
            if (minmax != 0 && a[tid] < a[tid + jj]) {
                dummy = a[tid];
                a[tid] = a[tid + jj];
                a[tid + jj] = dummy;
            }
        }
        __syncthreads();
    }
}

__global__ void global_exchanges(int *a, int j, int k) {

    int jj, jjj, minmax, tid, dummy;

    tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    jj = 1 << j;
    jjj = 2 << k;
    if ((tid & jj) == 0) {
        minmax = tid & jjj;
        if (minmax == 0 && a[tid] > a[tid + jj]) {
            dummy = a[tid];
            a[tid] = a[tid + jj];
            a[tid + jj] = dummy;
        }
        if (minmax != 0 && a[tid] < a[tid + jj]) {
            dummy = a[tid];
            a[tid] = a[tid + jj];
            a[tid + jj] = dummy;
        }
    }
}

int main(int argc, char *argv[]) {

    int i, j, k, kk, kkk, Q, A_size, blocks, threads;
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
        // printf("%d ", A[i]);
    }
    // printf("\n");

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

    for (k = 0; k < 10; k++) {
        kk = 1 << k;
        kkk = 2 << k;
        external_exchanges<<<blocks, threads>>>(d_a, kk, kkk);
        internal_exchanges<<<blocks, threads>>>(d_a, k);
    }

    for (k = 10; k < Q; k++) {
        kk = 1 << k;
        kkk = 2 << k;
        external_exchanges<<<blocks, threads>>>(d_a, kk, kkk);
        for (j = k - 1; j > 9; j--) {
            global_exchanges<<<blocks, threads>>>(d_a, j, k);
        }
        internal_exchanges<<<blocks, threads>>>(d_a, k);
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

    // for (i = 0; i < A_size; i++) {
    //     printf("%d ", A[i]);
    // }
    // printf("\n");

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);

    return 0;
}

int isSortedAscending(int *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return 0; // Array is not sorted
        }
    }
    return 1; // Array is sorted
}

int isSortedDescending(int *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] < arr[i + 1]) {
            return 0; // Array is not sorted
        }
    }
    return 1; // Array is sorted
}
