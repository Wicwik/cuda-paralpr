#include "lib/test/helper.hh"
#include "lib/layers/sigmoid_act.hh"

void test_name(SigmoidLayer sl)
{
    std::cout << "TEST NAME\n";

    std::string name = sl.get_name();
    std:: cout << name << std::endl;

    assertm(name == "the_name", "FAILED Layer should have a name");

    std::cout << "PASSED\n";
}

void test_forward(SigmoidLayer sl)
{
    std::cout << "TEST FORWARD\n";

    Matrix input{20, 10};
    input.allocate_mem();

    helper::fill_matrix_with_radom_range(input, -10, 10);
    input.copy_hd();

    Matrix output = sl.forward(input);
    output.copy_dh();

    assertm(output.d_mem != nullptr, "FAILED output is nullptr");
    assertm(output.dim.x == input.dim.x, "FAILED output has wrong dims");
    assertm(output.dim.y == input.dim.y, "FAILED output has wrong dims");

    for (int i = 0; i < output.dim.x; i++)
    {
        for (int j = 0; j < output.dim.y; j++)
        {
            float expected = helper::calc_sigmoid(input[i*output.dim.y + j]);
            std::cout << output[i*output.dim.y + j] << " " << expected << " | ";

            assertm(helper::cmpf(output[i*output.dim.y + j], expected, 0.0001), "FAILED wrong weight value");
        }

        std::cout << std::endl;
    }

    std::cout << "PASSED\n";
}

void test_backprop(SigmoidLayer sl)
{
    std::cout << "TEST BACKPROP\n";

    Matrix input{10, 5};
    input.allocate_mem();

    helper::fill_matrix_with_singlevalue(input, 3);
    input.copy_hd();

    Matrix error{10, 5};
    error.allocate_mem();

    helper::fill_matrix_with_singlevalue(error, 2);
    error.copy_hd();

    float expected = 2 * helper::calc_sigmoid(3) * (1 - helper::calc_sigmoid(3));

    Matrix output = sl.forward(input);
    Matrix output_error = sl.backprop(error);
    output_error.copy_dh();

    assertm(output_error.d_mem != nullptr, "FAILED output error is nullptr");
    assertm(output_error.dim.x == input.dim.x, "FAILED output error has wrong dims");
    assertm(output_error.dim.y == input.dim.y, "FAILED output error has wrong dims");

    for (int i = 0; i < output_error.dim.x; i++)
    {
        for (int j = 0; j < output_error.dim.y; j++)
        {
            std::cout << output_error[i*output_error.dim.y + j] << " " << expected << " | ";
            assertm(helper::cmpf(output_error[i*output_error.dim.y + j], expected, 0.0001), "FAILED wrong weight value");
        }
        std::cout << std::endl;
    }

    std::cout << "PASSED\n";

}


int main()
{
    SigmoidLayer sl("the_name");

    test_name(sl);
    std::cout << std::endl;

    test_forward(sl);
    std::cout << std::endl;

    test_backprop(sl);
    std::cout << std::endl;
}