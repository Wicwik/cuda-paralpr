#pragma once

#include "layer.hh"

__device__ float sigmoid(float x) 
{
    return 1.0f / (1 + exp(-x));
}

__global__ void sigmoid_forward(float *input, float *output, int input_x, int input_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input_x * input_y))
    {
        output[i] = sigmoid(input[i]);
    }
}

__global__ void sigmoid_backprop(float *input, float *input_error, float *output_error, int input_x, int input_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input_x * input_y))
    {
        output_error[i] = input_error[i] * sigmoid(input[i]) * (1 - sigmoid(input[i]));
    }
}

class SigmoidLayer : public layer
{
public:
    SigmoidLayer(std::string name)
        : _name{name}
    {
    }

    ~SigmoidLayer()
    {
    }

    forward(Matrix &input)
    {
        _input = input;
        _output.allocate_mem(_input.dim);

        dim3 block_size(256);
        dim3 number_of_blocks((_input.dim.x * _input.dim.y + block_size.x - 1)/block_size.x);

        sigmoid_forward<<<number_of_blocks, block_size>>>(_input.d_mem.get(), _output.d_mem.get(), _input.dim.x, _input.dim.y);

        NNException::throwIfDeviceErrorsOccurred("Failed at sigmoid forward propagation.");
    }

    backprop(Matrix &input_error, float learning_rate = 0.01)
    {
        _error.allocate_mem(_input.dim);

        dim3 block_size(256);
        dim3 number_of_blocks((_input.dim.x * _input.dim.y + block_size.x - 1)/block_size.x);

        sigmoid_backprop<<<number_of_blocks, block_size>>>(_input.d_mem.get(), input_error.d_mem.get(), _error.d_mem.get(), _input.dim.x, _input.dim.y);

        NNException::throwIfDeviceErrorsOccurred("Failed at sigmoid back propagation.");
    }

private:
    Matrix _output;

    Matrix _input;
    Matrix _error;
} 