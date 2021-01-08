/*
 *   Sigmoid activation layer
 */

#pragma once

#include <cmath>
#include "layer.hh"

// sigmoid calculation function only callable form kernels
__device__ float sigmoid(float x) 
{
    return 1.0f / (1.0f + expf(-x));
}

// kernel - calculate sigmoid function value for each item in input matrix
__global__ void sigmoid_forward(float *input, float *output, int input_x, int input_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input_x * input_y))
    {
        output[i] = sigmoid(input[i]);
    }
}

// kernel - calcualte sigmoid function derivate for each item in input matrix
__global__ void sigmoid_backprop(float *input, float *input_error, float *output_error, int input_x, int input_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input_x * input_y))
    {
        output_error[i] = input_error[i] * sigmoid(input[i]) * (1.0f - sigmoid(input[i]));
    }
}

class SigmoidLayer : public Layer
{
    
public:
    SigmoidLayer(std::string name)
    {
        this->_name = name;
    }

    ~SigmoidLayer()
    {
    }

    Matrix &forward(Matrix &input)
    {
        _input = input;
        _output.allocate_mem(_input.dim);

        dim3 block_size(256);
        dim3 number_of_blocks((_input.dim.x * _input.dim.y + block_size.x - 1)/block_size.x);

        sigmoid_forward<<<number_of_blocks, block_size>>>(_input.d_mem.get(), _output.d_mem.get(), _input.dim.x, _input.dim.y);

        NNException::throwIfDeviceErrorsOccurred("Failed at sigmoid forward propagation.");

        return _output;
    }

    Matrix &backprop(Matrix &input_error, float learning_rate = 0.01)
    {
        _error.allocate_mem(_input.dim);

        dim3 block_size(256);
        dim3 number_of_blocks((_input.dim.x * _input.dim.y + block_size.x - 1)/block_size.x);

        sigmoid_backprop<<<number_of_blocks, block_size>>>(_input.d_mem.get(), input_error.d_mem.get(), _error.d_mem.get(), _input.dim.x, _input.dim.y);

        NNException::throwIfDeviceErrorsOccurred("Failed at sigmoid back propagation.");

        return _error;
    }

private:
    Matrix _output;

    Matrix _input;
    Matrix _error;
};