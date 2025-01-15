#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function to add two arrays
__global__ void add(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int size = 256;
    int a[size], b[size], c[size];
    int *d_a, *d_b, *d_c;
    cudaError_t err, kernelErr;

    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // for (int i = 0; i < size; i++) {
    //     printf("a[%d] = %d\n", i, a[i]);
    //     printf("b[%d] = %d\n", i, b[i]);
    // }

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    // Copy data from host to GPU
    err = cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error during cudaMemcpy (d_a): %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error during cudaMemcpy (d_b): %s\n", cudaGetErrorString(err));
    }

    // Launch kernel with 256 threads in a single block
    add<<<1, 256>>>(d_a, d_b, d_c, size);

    // Check for kernel launch errors
    kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(kernelErr));
    }

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();


    // Copy result back to the host
    err = cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error during cudaMemcpy (c_a): %s\n", cudaGetErrorString(err));
    }

    // Print the results
    for (int i = 0; i < size; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
