#include "lib/test/helper.hh"
#include "lib/layers/relu_act.hh"

void test_name(ReluLayer rl)
{
    std::cout << "TEST NAME\n";

    std::string name = rl.get_name();
    std:: cout << name << std::endl;

    assertm(name == "the_name", "FAILED Layer should have a name");

    std::cout << "PASSED\n";
}

void test_forward(ReluLayer rl)
{
    std::cout << "TEST FORWARD\n";

    Matrix input{20, 10};
    input.allocate_mem();

    helper::fill_matrix_with_radom_range(input, -10, 10);
    input.copy_hd();

    Matrix output = rl.forward(input);
    output.copy_dh();

    assertm(output.d_mem != nullptr, "FAILED output is nullptr");

    for (int i = 0; i < output.dim.x; i++)
    {
        for (int j = 0; j < output.dim.y; j++)
        {
            if(input[i*output.dim.y + j] < 0)
            {
                std::cout << output[i*output.dim.y + j] << " " << 0 << " | ";
                assertm(output[i*output.dim.y + j] == 0, "Output has wrong value");
            }
            else
            {
                std::cout << output[i*output.dim.y + j] << " " << input[i*output.dim.y + j] << " | ";
                assertm(output[i*output.dim.y + j] == input[i*output.dim.y + j], "Output has wrong value");
            }

        }

        std::cout << std::endl;
    }

    std::cout << "PASSED\n";
}

void test_backprop(ReluLayer rl)
{
    std::cout << "TEST BACKPROP\n";

    Matrix input{10, 5};
    input.allocate_mem();

    for (int i = 0; i < input.dim.x; i++)
    {
        for (int j = 0; j < input.dim.y; j++)
        {
            if (i < 3)
            {
                input[i*input.dim.y + j] = 4;
            }
            else
            {
                input[i*input.dim.y + j] = -2;
            }
        }
    }

    input.copy_hd();

    Matrix error{10, 5};
    error.allocate_mem();

    helper::fill_matrix_with_singlevalue(error, 2);
    error.copy_hd();

    Matrix output = rl.forward(input);
    Matrix output_error = rl.backprop(error);
    output_error.copy_dh();

    assertm(output_error.d_mem != nullptr, "FAILED output error is nullptr");
    assertm(output_error.dim.x == input.dim.x, "FAILED output error has wrong dims");
    assertm(output_error.dim.y == input.dim.y, "FAILED output error has wrong dims");

    for (int i = 0; i < output_error.dim.x; i++)
    {
        for (int j = 0; j < output_error.dim.y; j++)
        {
            if (input[i*output_error.dim.y] > 0)
            {
                std::cout << output_error[i*output_error.dim.y + j] << " " << 1*2  << " | ";
                assertm(output_error[i*output_error.dim.y + j] == 1*2, "Output error has wrong value");
            }
            else
            {
                std::cout << output_error[i*output_error.dim.y + j] << " " << 0 << " | ";
                assertm(output_error[i*output_error.dim.y + j] == 0, "Output error has wrong value");
            }
        }
        std::cout << std::endl;
    }

    std::cout << "PASSED\n";

}

int main()
{
    ReluLayer rl{"the_name"};

    test_name(rl);
    std::cout << std::endl;

    test_forward(rl);
    std::cout << std::endl;

    test_backprop(rl);
    std::cout << std::endl;
}