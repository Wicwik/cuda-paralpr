#include <iostream>
#include <algorithm>
#include <chrono>

__global__ void add(float *x, float *y, float *z, int size)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
 
    for (int i = index; i < size; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}

cudaError_t cuda_add(float *x, float *y, float *z, int size);

int main()
{
    const int N = 1 << 20;

    float *x = new float[N];
    float *y = new float[N];
    float *z = new float[N];
 
    std::fill_n(x, N, 1.0f);
    std::fill_n(y, N, 2.0f);
 
    cudaError_t cudaStatus = cuda_add(x, y, z, N);
 
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "add_cuda failed!");
        return 1;
    }
 
    float max_err = 0.0f;
    for (int i = 0; i < N; i++)
    {
        max_err = std::fmax(max_err, std::fabs(z[i]-3.0f));
    }
    std::cout << "Max error: " << max_err << std::endl;
 
    delete[] x;
    delete[] y;
    delete[] z;
}

cudaError_t cuda_add(float *x, float *y, float *z, int size)
{
    float *dev_x = 0;
    float *dev_y = 0;
    float *dev_z = 0;
    cudaError_t cudaStatus;
 
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); 
        return cudaStatus;
    }
 
    cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(float));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_x);
        return cudaStatus;
    }
 
    cudaStatus = cudaMalloc((void**)&dev_y, size * sizeof(float));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_x);
        cudaFree(dev_y);
    
        return cudaStatus;
    }
 
    cudaStatus = cudaMalloc((void**)&dev_z, size * sizeof(float));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_x);
        cudaFree(dev_y);
        cudaFree(dev_z);
    
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_x);
        cudaFree(dev_y);
        cudaFree(dev_z);
    
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_y, y, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_x);
        cudaFree(dev_y);
        cudaFree(dev_z);
    
        return cudaStatus;
    }
 
    int block_size = 256;
    int number_of_blocks = (size + block_size - 1) / block_size; 

    auto start = std::chrono::high_resolution_clock::now();
    add<<<number_of_blocks, block_size>>>(dev_x, dev_y, dev_z, size);
    

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_x);
        cudaFree(dev_y);
        cudaFree(dev_z);
    
        return cudaStatus;
    }

    cudaStatus = cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
 
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << duration.count() << " microseconds" << std::endl;
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        cudaFree(dev_x);
        cudaFree(dev_y);
        cudaFree(dev_z);
    
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(z, dev_z, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_x);
        cudaFree(dev_y);
        cudaFree(dev_z);
    
        return cudaStatus;
    }

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_z);
 
    return cudaStatus;
}