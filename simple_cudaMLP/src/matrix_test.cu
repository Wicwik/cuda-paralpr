#include "lib/matrix/matirx.hh"

int main()
{
    Matrix Y{3,1};
    Matrix X{3,1};

    Y.allocate_mem();
    X.allocate_mem();

    Y[0] = 1;
    Y[1] = 1;
    Y[2] = 1;

    Y.copy_hd();

    X[0] = 2.2;
    X[1] = 1.0;
    X[2] = 0.4;

    X.copy_hd();
 
    for (int i = 0; i < X.dim.x; i++)
    {
        for (int j = 0; j < X.dim.y; j++)
        {
            std::cout << X[i*X.dim.y + j] << std::endl;
        }
    }

    return 0;
}