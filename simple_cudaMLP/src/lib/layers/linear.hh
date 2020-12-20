#include "layer.hh"

__global__ linear_forward(float *weights, float *input, float *output, float *bias, int weights_x, int weights_y, int input_x, int input_y)
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
            tmp += weights[row*weights_x + i] * input[i*input_y + col] //not sure if it should be input_y or input_x
        }

        output[row*output_x + col] = tmp + bias[row];
    }
}

__global__ linear_backprop(float *weights, float *output_error, float *input_error, int weights_x, int weights_y, int output_error_x, int output_error_y)
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

__global__ linear_update_bias(float *output_error, float *bias,  int output_error_x, int output_error_y, int bias_x, float learning_rate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (output_error_x*output_error_y))
    {
        int x = i % output_error_x;
        int y = i / output_error_x;
        atomicAdd(&bias[y], - learning_rate*(output_error[y*output_error_x + x]/output_error_x));
    }
}

__global__ linear_update_weights(float *output_error, float *input, float *weights, int output_error_x, int output_error_y, int input_x, int input_y, float learning_rate)
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
            tmp += output_error[row*weights_y + i] * input[col*input_x + i];
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

        initialize_bias();
        initialize_weights();
    }

    ~LinearLayer()
    { 
    }

    Matrix &forward(Matrix &input)
    {

    }

    Matrix &backprop(Matrix &output_error, float learning_rate = 0.01)
    {

    }

    int get_x() const
    {

    }

    int get_y() const
    {

    }

    Matrix get_bias() const
    {

    }

    Matrix get_weights() const
    {

    }

private:
    Matrix _weights;
    Matrix _bias;

    Matrix _output;
    Matrix _input;
    Matrix _input_error;

    void initialize_bias()
    {

    }

    void initialize_weights()
    {
        
    }

    void compute_backprop_error(Matrix &output_error)
    {

    }

    void compute_output(Matrix input)
    {

    }

    void update_bias(Matrix &output_error, float learning_rate)
    {
        
    }

    void update_weights(Matrix &output_error, float learning_rate)
    {

    }

}