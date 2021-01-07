#include "lib/layers/linear.hh"
#include "lib/test/helper.hh"

void test_name(LinearLayer ll)
{
    std::cout << "TEST NAME\n";

    std::string name = ll.get_name();
    std:: cout << name << std::endl;

    assertm(name == "the_name", "FAILED Layer should have a name");

    std::cout << "PASSED\n";
}

void test_weitght_size(LinearLayer ll, MatDim w_dim)
{
    std::cout << "TEST WEIGHT SIZE\n";

    unsigned int x = ll.get_x();
    unsigned int y = ll.get_y();

    assertm(x == w_dim.x, "FAILED Wrong weight size");
    assertm(y == w_dim.y, "FAILED Wrong weight size");

    std::cout << "PASSED\n";
}

void test_bias_iszero(LinearLayer ll, MatDim w_dim)
{
    std::cout << "TEST BIAS IS ZERO\n";

    Matrix bias = ll.get_bias();
    bias.copy_dh();

    assertm(bias.dim.x == w_dim.y, "Wrong bias size");
    assertm(bias.dim.y == 1, "Wrong bias size");

    for (int i = 0; i < bias.dim.x; i++) 
    {
        assertm(bias[i] == 0, "FAILED Wrong bias value");
    }

    std::cout << "PASSED\n";
}

void test_weitghts(LinearLayer ll, MatDim w_dim)
{
    Matrix weights = ll.get_weights();
    weights.copy_dh();

    float prev = -1.0;

    for (int i = 0; i < w_dim.x; i++)
    {
        for (int j = 0; i < w_dim.y; j++)
        {
            if (helper::cmpf(weights[i+w_dim.y + j], prev, 0.0001))
            {
                std::cout << "FAILED\n";
                return;
            }

            prev = weights[i+w_dim.y + j];
        }
    }

     std::cout << "PASSED\n";
}

void test_forward(LinearLayer ll, Matrix input, MatDim w_dim)
{
    std::cout << "TEST FORWARD\n";

    std::vector<float> bias_cols = {1, 2, 3, 4};
    std::vector<float> input_cols = {3, 5, 7};
    std::vector<float> weight_rows = {2, 4, 6, 8};

    helper::fill_cols(ll._bias, bias_cols);
    helper::fill_cols(input, input_cols);
    helper::fill_rows(ll._weights, weight_rows);

    ll._bias.copy_hd();
    ll._weights.copy_hd();
    input.copy_hd();

    Matrix output = ll.forward(input);
    output.copy_dh();

    assertm(output.d_mem != nullptr, "FAILED output is nullptr");
    assertm(output.dim.x == input.dim.x, "FAILED output has wrong dims");
    assertm(output.dim.y == w_dim.y, "FAILED output has wrong dims");

    for (int i = 0; i < output.dim.x; i++)
    {
        for (int j = 0; j < output.dim.y; j++)
        {
            float expected = weight_rows[j] * input_cols[i] * w_dim.x + bias_cols[j];
            std::cout << output[i*output.dim.y + j] << " " << expected << " | ";

            assertm(output[i*output.dim.y + j] == expected, "FAILED output has wrong value");
        }

        std::cout << std::endl;
    }

    std::cout << "PASSED\n";
}

void test_backprop(LinearLayer ll, Matrix input, Matrix error, MatDim w_dim)
{
    std::cout << "TEST BACKPROP\n";

    std::vector<float> weight_cols = {6, 8};
    std::vector<float> error_cols = {3, 5, 7};

    helper::fill_cols(ll._weights, weight_cols);
    helper::fill_cols(error, error_cols);

    ll._weights.copy_hd();
    error.copy_hd();

    Matrix output = ll.forward(input);
    Matrix input_error = ll.backprop(error);
    input_error.copy_dh();

    assertm(input_error.d_mem != nullptr, "FAILED input error is nullptr");
    assertm(input_error.dim.x == input.dim.x, "FAILED input error has wrong dims");
    assertm(input_error.dim.y == input.dim.y, "FAILED input error has wrong dims");

    for (int i = 0; i < input_error.dim.x; i++)
    {
        for (int j = 0; j < input_error.dim.y; j++)
        {
            float expected = weight_cols[j] * error_cols[i] * w_dim.y;
            std::cout << input_error[i*input_error.dim.y + j] << " " << expected << " | ";

            assertm(input_error[i*input_error.dim.y + j] == expected, "FAILED input error has wrong value");
        }

        std::cout << std::endl;
    }

    std::cout << "PASSED\n";
}

void test_wupdate(LinearLayer ll, Matrix input, Matrix error, MatDim w_dim)
{
    std::cout << "TEST WEIGHTS UPDATE\n";

    std::vector<float> weight_cols = {2, 4};
    std::vector<float> error_rows = {3, 5, 7, 9};
    std::vector<float> input_rows = {2, 4};
    float learning_rate = 0.1;

    helper::fill_cols(ll._weights, weight_cols);
    helper::fill_rows(error, error_rows);
    helper::fill_rows(input, input_rows);

    ll._weights.copy_hd();
    error.copy_hd();
    input.copy_hd();

    Matrix output = ll.forward(input);
    Matrix input_error = ll.backprop(error, learning_rate);

    ll._weights.copy_dh();

    assertm(ll._weights.d_mem != nullptr, "FAILED weights are nullptr");

    for (int i = 0; i < w_dim.x; i++)
    {
        for (int j = 0; j < w_dim.y; j++)
        {
            float expected = weight_cols[i] - learning_rate * error_rows[j] * input_rows[i];
            std::cout << ll._weights[i*w_dim.y + j] << " " << expected << " | ";

            assertm(helper::cmpf(ll._weights[i*w_dim.y + j], expected, 0.0001), "FAILED wrong weight value");
        }

        std::cout << std::endl;
    }

    std::cout << "PASSED\n";
}

void test_bupdate(LinearLayer ll, Matrix input, Matrix error)
{
    std::cout << "TEST BIAS UPDATE\n";

    std::vector<float> bias_cols = {1, 2, 3, 4};
    std::vector<float> error_rows = {3, 5, 7, 9};
    float learning_rate = 0.1;

    helper::fill_cols(ll._bias, bias_cols);
    helper::fill_rows(error, error_rows);

    ll._bias.copy_hd();
    error.copy_hd();

    Matrix output = ll.forward(input);
    Matrix input_error = ll.backprop(error, learning_rate);

    ll._bias.copy_dh();

    assertm(ll._bias.d_mem != nullptr, "FAILED bias is nullptr");

    for (int i = 0; i < ll._bias.dim.x; i++)
    {
        float expected = bias_cols[i] - learning_rate * error_rows[i];
        std::cout << ll._bias[i] << " " << expected << " | ";

        assertm(helper::cmpf(ll._bias[i], expected, 0.0001), "FAILED wrong bias value");
    }
    std::cout << std::endl;

    std::cout << "PASSED\n";
}

int main()
{
    MatDim w_dim{2, 4};

    Matrix input{3, 2};
    input.allocate_mem();

    Matrix error{3, 4};
    error.allocate_mem();

    LinearLayer ll("the_name", w_dim);

    test_name(ll);
    std::cout << std::endl;

    test_weitght_size(ll, w_dim);
    std::cout << std::endl;

    test_bias_iszero(ll, w_dim);
    std::cout << std::endl;

    test_forward(ll, input, w_dim);
    std::cout << std::endl;

    test_backprop(ll, input, error, w_dim);
    std::cout << std::endl;

    test_bupdate(ll, input, error);
    std::cout << std::endl;

    test_wupdate(ll, input, error, w_dim);
    std::cout << std::endl;
}