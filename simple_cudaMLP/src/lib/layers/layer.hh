/*
 *   An Interface class for layers
 */

#pragma once

#include <string>
#include "../matrix/matrix.hh"

// layer interface expected to be derived
class Layer
{
public:
    virtual ~Layer() = 0;

    // pure virtual functions
    // every layer must have a forward and backprop
    virtual Matrix &forward(Matrix &input) = 0;
    virtual Matrix &backprop(Matrix &input_error, float learning_rate) = 0;

    std::string get_name()
    {
        return this->_name;
    }

protected:
	// name of the layer
    std::string _name;
};

inline Layer::~Layer()
{
}