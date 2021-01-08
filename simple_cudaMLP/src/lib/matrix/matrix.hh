/*
 *   Matrix implemetation
 */

#pragma once

#include <memory>
#include "../exception_handler/cuda_exception_handler.hh"

// simple dimension structure
struct MatDim
{
    size_t x;
    size_t y;

    MatDim(size_t x = 1, size_t y = 1)
        : x{x}, y{y}
    {
    }
};


// matrix class representation
class Matrix
{
public:
    MatDim dim;

    std::shared_ptr<float> h_mem; // host memory
    std::shared_ptr<float> d_mem; // device memory

    Matrix(size_t x = 1, size_t y = 1)
        : dim{x, y}, h_mem{nullptr}, d_mem{nullptr}
        , _h_is_allocated{false}, _d_is_allocated{false}
    {
    }

    Matrix(MatDim dim)
        : Matrix{dim.x, dim.y}
    {
    }

    void allocate_mem()
    {
        _h_allocate_mem(); // we need to allocate host memory
        _d_allocate_mem(); // we need to allocate device memory
    }

    void allocate_mem(MatDim dim)
    {
        if (!_h_is_allocated && !_d_is_allocated)
        {
            this->dim = dim;
            allocate_mem();
        }
    }

    // copy host memory to device memory
    void copy_hd()
    {
        if (_h_is_allocated && _d_is_allocated)
        {
            cudaMemcpy(d_mem.get(), h_mem.get(), dim.x * dim.y * sizeof(float), cudaMemcpyHostToDevice);
            NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
        }
        else
        {
            throw NNException("Not allocated memory.");
        }
    }

    // copy device memory to host memory
    void copy_dh()
    {
        if (_h_is_allocated && _d_is_allocated)
        {
            cudaMemcpy(h_mem.get(), d_mem.get(), dim.x * dim.y * sizeof(float), cudaMemcpyDeviceToHost);
            NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
        }
        else
        {
            throw NNException("Not allocated memory.");
        }
    }

    // overloaded [] operator for easy access
    float& operator[](const int i)
    {
        return h_mem.get()[i];
    }

    const float& operator[](const int i) const
    {
        return h_mem.get()[i];
    }

private:
    bool _h_is_allocated;
    bool _d_is_allocated;
    

    void _h_allocate_mem()
    {
        if (!_h_is_allocated)
        {
            h_mem = std::shared_ptr<float>(new float[dim.x * dim.y], [&](float *ptr){ delete[] ptr; });

            _h_is_allocated = true;
        }
    }

    void _d_allocate_mem()
    {
        if (!_d_is_allocated)
        {
            float *devmem = nullptr;

            cudaMalloc(&devmem, dim.x * dim.y * sizeof(float));
            NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory.");

            d_mem = std::shared_ptr<float>(devmem, [&](float *ptr){ cudaFree(ptr); });

            _d_is_allocated = true;
        }

    }
};