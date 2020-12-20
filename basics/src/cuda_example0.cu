#include <iostream>
#include <cmath>
#include <chrono>

__global__ void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i]+= x[i];
    }
}

int main()
{
    int N = 1 << 20;
    float *x, *y;
 
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
 
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
 
    auto start = std::chrono::high_resolution_clock::now();
    add<<<1,1>>>(N, x, y);
 
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
 
    float max_err = 0.0f;
    for (int i = 0; i < N; i++)
    {
        max_err = std::fmax(max_err, std::fabs(y[i]-3.0f));
    }
    std::cout << duration.count() << " microseconds" << std::endl; 
    std::cout << "Max error: " << max_err << std::endl;
 
    cudaFree(x);
    cudaFree(y);
}