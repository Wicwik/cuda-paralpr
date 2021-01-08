#pragma once

#include <cmath>
#include "layer.hh"

__global__ void relu_forward(float *input, float *output, int input_x, int input_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input_x * input_y))
    {
        output[i] = fmaxf(input[i], 0);
    }
}

__global__ void relu_backprop(float *input, float *input_error, float *output_error, int input_x, int input_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input_x * input_y))
    {
        if (input[i] > 0)
        {
            output_error[i] = input_error[i]; 
        }
        else
        {
            output_error[i] = 0;
        }
    }
}

class ReluLayer : public Layer
{

public:
    ReluLayer(std::string name)
    {
        this->_name = name;
    }

    ~ReluLayer()
    { 
    }

    Matrix &forward(Matrix &input)
    {
        _input = input;
        _output.allocate_mem(_input.dim);

        dim3 block_size(256);
        dim3 number_of_blocks((_input.dim.x * _input.dim.y + block_size.x - 1)/block_size.x);

        relu_forward<<<number_of_blocks, block_size>>>(_input.d_mem.get(), _output.d_mem.get(), _input.dim.x, _input.dim.y);

        NNException::throwIfDeviceErrorsOccurred("Failed at ReLU forward propagation.");

        return _output;
    }

    Matrix &backprop(Matrix &input_error, float learning_rate = 0.01)
    {
        _error.allocate_mem(_input.dim);

        dim3 block_size(256);
        dim3 number_of_blocks((_input.dim.x * _input.dim.y + block_size.x - 1)/block_size.x);

        relu_backprop<<<number_of_blocks, block_size>>>(_input.d_mem.get(), input_error.d_mem.get(), _error.d_mem.get(), _input.dim.x, _input.dim.y);

        NNException::throwIfDeviceErrorsOccurred("Failed at ReLU back propagation.");

        return _error;
    }

private:
    Matrix _output;

    Matrix _input;
    Matrix _error;
};