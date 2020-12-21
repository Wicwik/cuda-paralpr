#include "lib/layers/linear.hh"

int main()
{
    Matrix Y{4,5};
    Y.allocate_mem();

    LinearLayer linear("Test", MatDim{4,5});

    std::cout << linear.get_name() << std::endl;
}