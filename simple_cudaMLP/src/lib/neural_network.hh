#pragma once

#include <vector>
#include "layers/layer.hh"
#include "loss_function/bce.hh"

class NeuralNetwork
{

public:
    NeuralNetwork(float learning_rate = 0.1)
        : _learning_rate{learning_rate}
    {
    }

    ~NeuralNetwork()
    {
        for (const auto layer: _layers)
        {
            delete layer;
        }
    }

    void add_layer(Layer *layer)
    {
        _layers.push_back(layer);
    }

    std::vector<Layer*> get_layers() const
    {
        return _layers;
    }

    Matrix forward(Matrix input)
    {
        Matrix tmp = input;

        for (const auto layer: _layers)
        {
            tmp = layer->forward(tmp);
        }

        _output = tmp;

        return _output;
    }

    void backprop(Matrix fake, Matrix real)
    {
        _output_derivative.allocate_mem(fake.dim);
        Matrix error = bce_cf.gradient(fake, real, _output_derivative);

        for (auto it = _layers.rbegin(); it != _layers.rend(); it++)
        {
            error = (*it)->backprop(error, _learning_rate);
        }

        cudaDeviceSynchronize();
    }

private:
    std::vector<Layer*> _layers;
    BCE bce_cf;

    Matrix _output;
    Matrix _output_derivative;

    float _learning_rate;
};