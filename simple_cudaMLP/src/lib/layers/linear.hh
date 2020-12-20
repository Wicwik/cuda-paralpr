#include "layer.hh"

__global__ linear_forward()
{

}

__global__ linear_backprop()
{
    
}

class LinearLayer : public Layer
{
public:
    LinearLayer(std::string name, MatDim weights_dims)
    {

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
    Matrix _weight;
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