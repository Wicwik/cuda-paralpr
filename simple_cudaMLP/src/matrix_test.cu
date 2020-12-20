#include "lib/layers/leaky_relu_act.hh"

int main()
{
    Matrix Y{4,5};
    Y.allocate_mem();

    LeakyReluLayer lrelu("Test");

    std::cout << lrelu.get_name() << std::endl;
}