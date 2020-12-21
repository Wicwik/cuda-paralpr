#pragma once

#include <vector>
#include "layers/layer.hh"
#include "loss_function/bce.hh"

class NeuralNetwork()
{

public:
    NeuralNetwork(float learning_rate = 0.1)
    {

    }

    ~NeuralNetwork()
    {
    }

    void add_layer(Layer *layer)
    {
        _layers.push_back(layer);
    }

    std::vector<Layer*> get_layers() const
    {

    }

    Mattix forward(Matrix input)
    {

    }

    void backprop(Matrix fake, Matrix real)
    {

    }

private:
    std::vector<Layer*> _layers;

    Matrix _output;
    Matrix _output_derivative;

    float _learning_rate;
};