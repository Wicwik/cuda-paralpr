/*
 *   An Interface class for layers
 */

#include <string>
#include "../matrix/matrix.hh"

class Layer
{
public:
    virtual ~Layer() = 0;

    virtual &Matrix forward(Matrix &input) = 0;
    virtual &Matrix backprop(Matrix &input_error, float learning_rate) = 0;

    std::string get_name()
    {
        return this->_name;
    }

protected:
    std::string _name;
}

inline Layer::~Layer()
{
}