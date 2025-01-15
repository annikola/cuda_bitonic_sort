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

    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    // Copy data from host to GPU
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads in a single block
    add<<<1, 256>>>(d_a, d_b, d_c, size);

    // Copy result back to the host
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

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
