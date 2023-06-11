#include <stdio.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cmath>
#include <cuda.h>
//for __syncthreads()
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)
#include <device_functions.h>

inline double seconds()
{
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / (double)f.QuadPart;
}

# define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define BDIMX 16
#define BDIMY 16

void printMatrix(const char* msg, int* matrix, int width, int height)
{
    printf("%s:\n", msg);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%4d ", matrix[i * width + j]);
        }
        printf("\n");
    }

    printf("\n");
}

void verifyResults(int* input, int* output, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (input[i * width + j] != output[j * height + i])
            {
                printf("Verification failed at position (%d, %d)\n", i, j);
                return;
            }
        }
    }

    printf("Verification successful\n");
}

__global__ void transposeMatrix(int* input, int* output, int width, int height)
{
    __shared__ int tile[BDIMY][BDIMX + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index_in = y * width + x;
        tile[threadIdx.y][threadIdx.x] = input[index_in];
    }

    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;

    if (x < height && y < width)
    {
        int index_out = y * height + x;
        output[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Set up matrix dimensions
    int width = 256;
    int height = 256;

    size_t size = width * height * sizeof(int);

    // Allocate host memory for input and output matrices
    int* hostInput = (int*)malloc(size);
    int* hostOutput = (int*)malloc(size);

    // Initialize input matrix
    for (int i = 0; i < width * height; i++)
    {
        hostInput[i] = i;
    }

    // Allocate device memory for input and output matrices
    int* deviceInput;
    int* deviceOutput;
    CHECK(cudaMalloc((void**)&deviceInput, size));
    CHECK(cudaMalloc((void**)&deviceOutput, size));

    // Copy input matrix from host to device
    CHECK(cudaMemcpy(deviceInput, hostInput, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 block(BDIMX, BDIMY);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Perform matrix transpose using shared memory
    CHECK(cudaEventRecord(start, 0));
    transposeMatrix << <grid, block >> > (deviceInput, deviceOutput, width, height);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // Copy output matrix from device to host
    CHECK(cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost));

    // Verify the results
    verifyResults(hostInput, hostOutput, width, height);

    // Compute the elapsed time
    float elapsedTime;
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Shared Memory Transpose Time: %.5f ms\n", elapsedTime);

    // Free host and device memory
    free(hostInput);
    free(hostOutput);
    CHECK(cudaFree(deviceInput));
    CHECK(cudaFree(deviceOutput));

    // Reset device
    CHECK(cudaDeviceReset());

    return 0;
}
