#include <cassert>
#include <random>

#include "layer.hh"

#define assertm(exp, msg) assert(((void)msg, exp))

__global__ void linear_forward(float *weights, float *input, float *output, float *bias, int weights_x, int weights_y, int input_x, int input_y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int output_x = input_x;
    int output_y = weights_y;

    float tmp = 0.0f;

    if ((row < output_y) && (col < output_x))
    {
        for (int i = 0; i < weights_x; i++)
        {
            tmp += weights[row*weights_x + i] * input[i*input_x + col];
        }

        output[row*output_x + col] = tmp + bias[row];
    }
}

__global__ void linear_backprop(float *weights, float *output_error, float *input_error, int weights_x, int weights_y, int output_error_x, int output_error_y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int input_error_x = output_error_x;
    int input_error_y = weights_y;

    float tmp = 0.0f;

    if ((row < input_error_y) && (col < input_error_x))
    {
        for (int i = 0; i < weights_y; i++)
        {
            tmp += weights[i*weights_x + row] * output_error[i*output_error_x + col];
        }
        input_error[row*input_error_x + col] = tmp;
    }
}

__global__ void linear_update_bias(float *output_error, float *bias,  int output_error_x, int output_error_y, int bias_x, float learning_rate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (output_error_x*output_error_y))
    {
        int x = i % output_error_x;
        int y = i / output_error_x;
        atomicAdd(&bias[y], - learning_rate*(output_error[y*output_error_x + x]/output_error_x));
    }
}

__global__ void linear_update_weights(float *output_error, float *input, float *weights, int output_error_x, int output_error_y, int input_x, int input_y, float learning_rate)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int weights_x = input_y;
    int weights_y = output_error_x;

    float tmp = 0.0f;

    if ((row < weights_y) && (col < weights_x))
    {
        for (int i = 0; i < output_error_x; i++)
        {
            tmp += output_error[row*output_error_x + i] * input[col*input_x + i];
        }
        weights[row*weights_x + col] = weights[row*weights_x + col] - learning_rate*(tmp/input_x);
    }
}

class LinearLayer : public Layer
{
public:
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

    Matrix _weights;
    Matrix _bias;

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
        std::default_random_engine gen;
        std::normal_distribution<float> dist;

        for (int i = 0; i < _weights.dim.x; i++)
        {
            for (int j = 0; j < _weights.dim.y; j++)
            {
                _weights[i*_weights.dim.x + j] = dist(gen) * threshold;
            }
        }
        
        _weights.copy_hd();
    }

    void _compute_output(Matrix input)
    {
        dim3 block_size(8, 8);
        dim3 number_of_blocks((_output.dim.x + block_size.x - 1)/block_size.x, (_output.dim.y + block_size.y -1)/block_size.y);

        linear_forward<<<number_of_blocks, block_size>>>(_weights.d_mem.get(), _input.d_mem.get(), _output.d_mem.get(), _bias.d_mem.get(), _weights.dim.x, _weights.dim.y, _input.dim.x, _input.dim.y);
    }

    void _compute_backprop_error(Matrix &output_error)
    {
        dim3 block_size(8, 8);
        dim3 number_of_blocks((_input.dim.x + block_size.x - 1)/block_size.x, (_input.dim.y + block_size.y -1)/block_size.y);

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
        dim3 block_size(8, 8);
        dim3 number_of_blocks((_weights.dim.x + block_size.x - 1)/block_size.x, (_weights.dim.y + block_size.y -1)/block_size.y);

        linear_update_weights<<<number_of_blocks, block_size>>>(output_error.d_mem.get(), _input.d_mem.get(), _weights.d_mem.get(), 
                                                                output_error.dim.x, output_error.dim.y, _input.dim.x, _input.dim.y, learning_rate);
    }

};