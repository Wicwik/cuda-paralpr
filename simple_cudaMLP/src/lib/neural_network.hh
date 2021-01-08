/*
 *   Neural net implementation
 */

#pragma once

#include <vector>
#include "layers/layer.hh"
#include "loss_function/bce.hh"

class NeuralNetwork
{

public:
    NeuralNetwork(float learning_rate = 0.01)
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

        // for (int c = 0; c < tmp.dim.x; c++)
        // {
        //     std::cout << "[" << c << "] ";
        //     for (int r = 0; r < tmp.dim.y; r++)
        //     {
        //         std::cout << tmp[c*tmp.dim.y + r] << " ";
        //     }

        //     std::cout << std::endl;
        // }

        for (auto layer: _layers)
        {
            tmp = layer->forward(tmp); // forward
            // tmp.copy_dh();
            // std::cout << layer->get_name() << std::endl;

            // for (int i = 0; i < tmp.dim.x; i++)
            // {
            //     for (int j = 0; j < tmp.dim.y; j++)
            //     {
            //         std::cout << tmp[i*tmp.dim.y + j] << " ";
            //     }

            //     std::cout << std::endl;
            // }

            // std::cout << std::endl;
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
            error = (*it)->backprop(error, _learning_rate); // backprop from back

            // std::cout << (*it)->get_name() << std::endl;

            // error.copy_dh();

            // for (int i = 0; i < error.dim.x; i++)
            // {
            //     for (int j = 0; j < error.dim.y; j++)
            //     {
            //         std::cout << error[i*error.dim.y + j] << " ";
            //     }

            //     std::cout << std::endl;
            // }
        }

        cudaDeviceSynchronize(); // wait for GPU
    }

private:
    std::vector<Layer*> _layers;
    BCE bce_cf;

    Matrix _output;
    Matrix _output_derivative;

    float _learning_rate;
};