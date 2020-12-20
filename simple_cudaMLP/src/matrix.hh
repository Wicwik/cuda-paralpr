#pragma once

#include <memory>
#include "cuda_exception_handler.h"

struct MatDim
{
    size_t x;
    size_t y;
};

class Matrix
{
public:
    MatDim dim;

    std::shared_ptr<float> h_mem;
    std::shared_ptr<float> d_mem;

    Matrix(size_t x = 1, size_t y = 1)
        : dim{x, y}, h_mem{nullptr}, d_mem{nullptr}
        , h_is_allocated{false}, d_is_allocated{false}
    {
    }

    Matrix(MatDim dim)
        : Matrix{dim.x, dim.y}
    {
    }

    void allocate_mem()
    {
        h_allocate_mem();
        d_allocate_mem();
    }

    void allocate_mem(MatDim dim)
    {
        if (!h_is_allocated && !d_is_allocated)
        {
            this->dim = dim;
            allocate_mem();
        }
    }

    void copy_hd()
    {
        if (h_is_allocated && d_is_allocated)
        {
            cudaMemcpy(d_mem.get(), h_mem.get(), dim.x * dim.y * sizeof(float), cudaMemcpyHostToDevice);
            NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
        }
        else
        {
            throw NNException("Not allocated memory.");
        }
    }

    void copy_dh()
    {
        if (h_is_allocated && d_is_allocated)
        {
            cudaMemcpy(h_mem.get(), d_mem.get(), dim.x * dim.y * sizeof(float), cudaMemcpyDeviceToHost);
            NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
        }
        else
        {
            throw NNException("Not allocated memory.");
        }
    }

    float& operator[](const int i)
    {
        return h_mem.get()[i];
    }

    const float& operator[](const int i) const
    {
        return h_mem.get()[i];
    }

private:
    bool h_is_allocated;
    bool d_is_allocated;
    

    void h_allocate_mem()
    {
        if (!h_is_allocated)
        {
            h_mem = std::shared_ptr<float>(new float[dim.x * dim.y], [&](float *ptr){ delete[] ptr; });

            h_is_allocated = true;
        }
    }

    void d_allocate_mem()
    {
        if (!d_is_allocated)
        {
            float *devmem = nullptr;

            cudaMalloc(&devmem, dim.x * dim.y * sizeof(float));
            NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory.");

            d_mem = std::shared_ptr<float>(devmem, [&](float *ptr){ cudaFree(ptr); });

            d_is_allocated = true;
        }

    }
};