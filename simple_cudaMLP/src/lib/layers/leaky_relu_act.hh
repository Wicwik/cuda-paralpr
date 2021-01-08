/*
 *   LeakyReLU activation layer implementation
 */

#pragma once

#include "layer.hh"

// kernel - calculate output value based on input and alpha
__global__ void lrelu_forward(float *input, float *output, int input_x, int input_y, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // calculate current thread id

    if (i < (input_x * input_y)) // to be sure, we dont access something we shoul not
    {
        if (input[i] < 0) // if input is lower than zero multiply it with aplha parameter
        {
            output[i] = alpha*input[i];
        }
        else
        {
            output[i] = input[i]; // else return the same value
        }
    }
}

// kernel - calculate the dervivative value(output_error) based on input_error and alpha
__global__ void lrelu_backprop(float *input, float *input_error, float *output_error, int input_x, int input_y, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input_x * input_y))
    {
        if (input[i] < 0)
        {
            output_error[i] = alpha*input_error[i]; 
        }
        else
        {
            output_error[i] = input_error[i];
        }
    }
}

class LeakyReluLayer : public Layer
{

public:
    LeakyReluLayer(std::string name, float alpha = 0.01)
    {
        this->_name = name;
        this->_alpha = alpha;
    }

    ~LeakyReluLayer()
    { 
    }

    Matrix &forward(Matrix &input)
    {
        _input = input;
        _output.allocate_mem(_input.dim); // alocate memory for the output

        dim3 block_size(256); 
        dim3 number_of_blocks((_input.dim.x * _input.dim.y + block_size.x - 1)/block_size.x); // calculate number of blocks

        lrelu_forward<<<number_of_blocks, block_size>>>(_input.d_mem.get(), _output.d_mem.get(), _input.dim.x, _input.dim.y, _alpha);

        NNException::throwIfDeviceErrorsOccurred("Failed at LeakyReLU forward propagation.");

        return _output;
    }

    Matrix &backprop(Matrix &input_error, float learning_rate = 0.01)
    {
        // allocate memory for output
        _error.allocate_mem(_input.dim);

        dim3 block_size(256);
        dim3 number_of_blocks((_input.dim.x * _input.dim.y + block_size.x - 1)/block_size.x); // calculate number of blocks

        lrelu_backprop<<<number_of_blocks, block_size>>>(_input.d_mem.get(), input_error.d_mem.get(), _error.d_mem.get(), _input.dim.x, _input.dim.y, _alpha);

        NNException::throwIfDeviceErrorsOccurred("Failed at LeakyReLU back propagation.");

        return _error;
    }

private:
    Matrix _output;

    Matrix _input;
    Matrix _error;

    float _alpha;
};