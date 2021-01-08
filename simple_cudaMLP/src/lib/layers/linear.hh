/*
 *   Linear layer implementation
 *   Performs linear transformation of matrix
 */

#pragma once

#include <cassert>
#include <random>
#include "layer.hh"

#define assertm(exp, msg) assert(((void)msg, exp)) // macro for assert with output message

// kernel - computes dot between two matrices and adds bias to result
__global__ void linear_forward(float *weights, float *input, float *output, float *bias, int weights_x, int weights_y, int input_x, int input_y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // calculate row thread number
    int col = blockIdx.x * blockDim.x + threadIdx.x; // calculate col thread number

    int output_x = input_x;
    int output_y = weights_y;

    float tmp = 0.0f;

    // printf("%d %d | ", output_x, output_y);
 
    if ((row < output_x) && (col < output_y))
    {
        for (int i = 0; i < weights_x; i++)
        {
            tmp += weights[i*weights_y + col] * input[row*input_y + i]; // multiply cols woth rows
            
            // # if __CUDA_ARCH__>=200
            // printf("%f * %f = %f\n", weights[i*weights_y + col], input[row*input_y + i], weights[i*weights_y + col] * input[row*input_y + i]);
            // #endif 
        }

        // # if __CUDA_ARCH__>=200
        // printf("%d %d %f\n", row, col, tmp);
        // #endif 
        output[row*output_y + col] = tmp + bias[col];
    }
}

// kernel - computes dot between two matrices, simulates transposed weight matrix
__global__ void linear_backprop(float *weights, float *output_error, float *input_error, int weights_x, int weights_y, int output_error_x, int output_error_y)
{
    int col = blockIdx.y * blockDim.y + threadIdx.y; // calculate row thread number
    int row = blockIdx.x * blockDim.x + threadIdx.x; // calculate col thread number

    int input_error_x = output_error_x;
    int input_error_y = weights_x; // transposed

    float tmp = 0.0f;

    if ((row < input_error_y) && (col < input_error_x))
    {
        for (int i = 0; i < weights_y; i++)
        {
            tmp += weights[row*weights_y + i] * output_error[col*output_error_y + i];
            //  # if __CUDA_ARCH__>=200
            // printf("%f * %f = %f, %d %d\n", weights[row*weights_y + i], output_error[col*output_error_y + i], tmp , row, col);
            // #endif 
        }

        // # if __CUDA_ARCH__>=200
        // printf("%d %d %d %f\n", row, col, col*input_error_y + row ,tmp);
        // #endif 

        input_error[col*input_error_y + row] = tmp;
    }
}

// kernel - adds bias derivative to bias
__global__ void linear_update_bias(float *output_error, float *bias,  int output_error_x, int output_error_y, int bias_x, float learning_rate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // calculates current thread 

    if (i < (output_error_x*output_error_y))
    {
        int x = i % output_error_x;
        int y = i / output_error_x;
        atomicAdd(&bias[y], -learning_rate*(output_error[x*output_error_y + y]/output_error_x)); // because multiple threads are accesing single memory we use atomic add
    }
}

// kernel - calculates new weights with simulated transposed input
__global__ void linear_update_weights(float *output_error, float *input, float *weights, int output_error_x, int output_error_y, int input_x, int input_y, float learning_rate)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // calculate row thread number
    int row = blockIdx.y * blockDim.y + threadIdx.y; // calculate col thread number

    int weights_x = input_y; // transposed
    int weights_y = output_error_y;

    float tmp = 0.0f;

    if ((row < weights_y) && (col < weights_x))
    {
        for (int i = 0; i < output_error_x; i++)
        {
            tmp += output_error[i*output_error_y + row] * input[i*input_y + col];

            //  # if __CUDA_ARCH__>=200
            // printf("%f * %f = %f, %d %d\n", output_error[i*output_error_y + row], input[i*input_y + col], tmp , row, col);
            // #endif 
        }
        // # if __CUDA_ARCH__>=200
        // printf("%d %d %d %f\n", row, col, col*weights_y + row ,tmp);
        // #endif 

        weights[col*weights_y + row] = weights[col*weights_y + row] - learning_rate*(tmp/input_x);
    }
}

class LinearLayer : public Layer
{
public:
    Matrix _weights;
    Matrix _bias;

    LinearLayer(std::string name, MatDim weights_dims)
        : _weights{weights_dims}, _bias{weights_dims.y, 1}
    {
        this->_name = name;
        _bias.allocate_mem();
        _weights.allocate_mem();

        _initialize_bias();
        _initialize_weights();
    }

    ~LinearLayer()
    { 
    }

    Matrix &forward(Matrix &input)
    {
        assertm(_weights.dim.x == input.dim.y, "Weights and Input matrices cannot be multiplied.");
        this->_input = input;

        MatDim output_dim{_input.dim.x, _weights.dim.y};
        _output.allocate_mem(output_dim);

        _compute_output(_input);
        NNException::throwIfDeviceErrorsOccurred("Failed at linear layer forward propagation.");

        return _output;
    }

    Matrix &backprop(Matrix &output_error, float learning_rate = 0.01)
    {
        _input_error.allocate_mem(_input.dim);

        _compute_backprop_error(output_error);
        NNException::throwIfDeviceErrorsOccurred("Failed at linear layer back propagation.");

        _update_bias(output_error, learning_rate);
        NNException::throwIfDeviceErrorsOccurred("Failed at linear layer bias update.");

        _update_weights(output_error, learning_rate);
        NNException::throwIfDeviceErrorsOccurred("Failed at linear layer weights update.");

        return _input_error;
    }

    int get_x() const
    {
        return _weights.dim.x;
    }

    int get_y() const
    {
        return _weights.dim.y;
    }

    Matrix get_bias() const
    {
        return _bias;
    }

    Matrix get_weights() const
    {
        return _weights;
    }

private:
    const float threshold = 0.01;

    Matrix _output;
    Matrix _input;
    Matrix _input_error;

    void _initialize_bias()
    {
        for (int i = 0; i < _bias.dim.x; i++)
        {
            _bias[i] = 0;
        }

        _bias.copy_hd();
    }

    void _initialize_weights()
    {
        std::random_device rd;
        std::default_random_engine gen(rd());
        std::normal_distribution<float> dist(1.0f, 1.0f); // initialize weitgts from normal distribution

        for (int i = 0; i < _weights.dim.x; i++)
        {
            for (int j = 0; j < _weights.dim.y; j++)
            {
                _weights[i*_weights.dim.y + j] = dist(gen) * threshold;
            }
        }
        
        _weights.copy_hd();
    }

    void _compute_output(Matrix input)
    {
        dim3 block_size(8, 8); // we want 2D blocks for matrix dot
        dim3 number_of_blocks((_output.dim.y + block_size.y - 1)/block_size.y, (_output.dim.x + block_size.x - 1)/block_size.x);

        linear_forward<<<number_of_blocks, block_size>>>(_weights.d_mem.get(), _input.d_mem.get(), _output.d_mem.get(), _bias.d_mem.get(), _weights.dim.x, _weights.dim.y, _input.dim.x, _input.dim.y);
    }

    void _compute_backprop_error(Matrix &output_error)
    {
        dim3 block_size(8, 8); // we want 2D blocks for matrix dot
        dim3 number_of_blocks((_input.dim.y + block_size.y -1)/block_size.y, (_input.dim.x + block_size.x - 1)/block_size.x);

        linear_backprop<<<number_of_blocks, block_size>>>(_weights.d_mem.get(), output_error.d_mem.get(), _input_error.d_mem.get(), _weights.dim.x, _weights.dim.y, output_error.dim.x, output_error.dim.y);
    }

    void _update_bias(Matrix &output_error, float learning_rate)
    {
        dim3 block_size(256); 
        dim3 number_of_blocks((output_error.dim.x + block_size.x - 1)/block_size.x);

        linear_update_bias<<<number_of_blocks, block_size>>>(output_error.d_mem.get(), _bias.d_mem.get(), output_error.dim.x, output_error.dim.y, _bias.dim.x, learning_rate);
    }

    void _update_weights(Matrix &output_error, float learning_rate)
    {
        dim3 block_size(8, 8); // we want 2D blocks for matrix dot
        dim3 number_of_blocks((_weights.dim.x + block_size.x - 1)/block_size.x, (_weights.dim.y + block_size.y -1)/block_size.y);

        linear_update_weights<<<number_of_blocks, block_size>>>(output_error.d_mem.get(), _input.d_mem.get(), _weights.d_mem.get(), 
                                                                output_error.dim.x, output_error.dim.y, _input.dim.x, _input.dim.y, learning_rate);
    }

};