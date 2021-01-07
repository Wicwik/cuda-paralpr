#pragma once

#include <cassert>
#include <cmath>

#include "../matrix/matrix.hh"

#define assertm(exp, msg) assert(((void)msg, exp))

__global__ void bce_cost(float *fake, float *real, int real_x, float *cost)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < real_x)
    {
        float tmp = real[i] * logf(fake[i]) + (1.0f - real[i]) * logf(1.0f - fake[i]);
        atomicAdd(cost, -tmp/real_x);
    }
}

__global__ void bce_gradient(float *fake, float *real, float *output_derivative, int real_x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < real_x)
    {
        output_derivative[i] = -1.0f * (real[i]/fake[i] - ((1.0f - real[i])/(1.0f - fake[i])));
    }
}


class BCE
{

public:
    float cost(Matrix fake, Matrix real)
    {
        assertm(fake.dim.x == real.dim.x, "Fake and real must have equal lenght.");

        float *cost = nullptr;
        cudaMallocManaged(&cost, sizeof(float));

        *cost = 0.0f;

        dim3 block_size(256);
        dim3 number_of_blocks((fake.dim.x + block_size.x - 1)/block_size.x);

        bce_cost<<<number_of_blocks, block_size>>>(fake.d_mem.get(), real.d_mem.get(), real.dim.x, cost);

        cudaDeviceSynchronize();
        NNException::throwIfDeviceErrorsOccurred("Failied at binary cross entropy cost.");

        float output = *cost;
        cudaFree(cost);

        return output;
    }

    Matrix gradient(Matrix fake, Matrix real, Matrix output_derivative)
    {
        assertm(fake.dim.x == real.dim.x, "Fake and real must have equal lenght.");

        dim3 block_size(256);
        dim3 number_of_blocks((fake.dim.x + block_size.x - 1)/block_size.x);

        bce_gradient<<<number_of_blocks, block_size>>>(fake.d_mem.get(), real.d_mem.get(), output_derivative.d_mem.get(), real.dim.x);

        NNException::throwIfDeviceErrorsOccurred("Failied at binary cross entropy gradient.");

        return output_derivative;
    }
};