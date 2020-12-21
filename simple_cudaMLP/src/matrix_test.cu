#include "lib/layers/linear.hh"
#include "lib/loss_function/bce.hh"



int main()
{
    Matrix Y{3,1};
    Matrix y{3,1};

    Y.allocate_mem();
    y.allocate_mem();

    Y[0] = 1;
    Y[1] = 1;
    Y[2] = 1;

    Y.copy_hd();

    y[0] = 0.0;
    y[1] = 0.0;
    y[2] = 0.0;

    y.copy_hd();

    LinearLayer linear("Test", MatDim{4,5});

    std::cout << Y.dim.x << std::endl;

    std::cout << BCE().cost(y, Y) << std::endl;

    std::cout << linear.get_name() << std::endl;

    return 0;
}