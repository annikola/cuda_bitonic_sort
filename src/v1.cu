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

__global__ void external_exchanges(int *a, int k) {

    int i, kk, kkk, minmax, tid, dummy, total_threads, total_blocks;

    total_threads = blockDim.x * blockDim.y * blockDim.z;
    total_blocks = gridDim.x * gridDim.y * gridDim.z;
    tid = blockIdx.x * blockDim.x + threadIdx.x;

    kk = 1 << k;
    kkk = 2 << k;
    if ((tid & kk) != 0) {
        i = tid + total_threads * total_blocks - kk;
    } else {
        i = tid;
    }

    minmax = i & kkk;
    if (minmax == 0 && a[i] > a[i + kk]) {
        dummy = a[i];
        a[i] = a[i + kk];
        a[i + kk] = dummy;
    }
    if (minmax != 0 && a[i] < a[i + kk]) {
        dummy = a[i];
        a[i] = a[i + kk];
        a[i + kk] = dummy;
    }
}

__global__ void internal_exchanges(int *a, int k) {

    int i, j, jj, jjj, minmax, tid, dummy, total_threads, total_blocks;

    total_threads = blockDim.x * blockDim.y * blockDim.z;
    total_blocks = gridDim.x * gridDim.y * gridDim.z;
    tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (j = k - 1; j >= 0; j--) {
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
        __syncthreads();
    }
}

__global__ void global_exchanges(int *a, int j, int k) {

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
        blocks = (A_size / MAX_THREADS) / 2;
        threads = MAX_THREADS;
    }

    for (k = 0; k < 11; k++) {
        external_exchanges<<<blocks, threads>>>(d_a, k);
        internal_exchanges<<<blocks, threads>>>(d_a, k);
    }

    for (k = 11; k < Q; k++) {
        external_exchanges<<<blocks, threads>>>(d_a, k);
        for (j = k - 1; j >= 10; j--) {
            global_exchanges<<<blocks, threads>>>(d_a, j, k);
        }
        internal_exchanges<<<blocks, threads>>>(d_a, k);
    }

    err = cudaMemcpy(B, d_a, A_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error during cudaMemcpy (B_a): %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop, 0); // Stop the timing...
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Total execution time: %f ms\n", elapsed_time);

    qsort(A, A_size, sizeof(int), asc_compare);
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
