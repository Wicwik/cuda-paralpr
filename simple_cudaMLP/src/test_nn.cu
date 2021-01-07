#include "lib/neural_network.hh"
#include "lib/layers/linear.hh"
#include "lib/layers/relu_act.hh"
#include "lib/loss_function/bce.hh"
#include "lib/test/helper.hh"

void test_layer_order(NeuralNetwork nn)
{
    std::cout << "TEST LAYER ORDER\n";

    LinearLayer* ll1 = new LinearLayer("LL1", MatDim{1, 1});
    LinearLayer* ll2 = new LinearLayer("LL2", MatDim{2, 2});
    ReluLayer* rl1 = new ReluLayer("RL1");
    ReluLayer* rl2 = new ReluLayer("RL2");

    nn.add_layer(ll1);
    nn.add_layer(rl1);
    nn.add_layer(ll2);
    nn.add_layer(rl2);

    std::vector<Layer*> layers = nn.get_layers();

    for (const auto& l : layers)
    {
        std::cout << l->get_name() << " ";
    }
    std::cout << std::endl;

    assertm(layers.size() == 4, "FAILED incorect number of layers");
    assertm(!layers.at(0)->get_name().compare("LL1"), "FAILED incorect layer");
    assertm(!layers.at(1)->get_name().compare("RL1"), "FAILED incorect layer");
    assertm(!layers.at(2)->get_name().compare("LL2"), "FAILED incorect layer");
    assertm(!layers.at(3)->get_name().compare("RL2"), "FAILED incorect layer");

    std::cout << "PASSED\n";
}

void test_forward()
{
    std::cout << "TEST forward\n";

    NeuralNetwork nn;
    Matrix input;
    
    input.dim = MatDim{10, 20};
    input.allocate_mem();

    MatDim output_dim{input.dim.x, 5};

    LinearLayer* ll1 = new LinearLayer("LL1", MatDim{input.dim.y, 4});
    ReluLayer* rl = new ReluLayer("RL");
    LinearLayer* ll2 = new LinearLayer("LL2", MatDim{4, output_dim.y});

    helper::fill_matrix_with_singlevalue(input, 4);
    helper::fill_matrix_with_singlevalue(ll1->_weights, 2);
    helper::fill_matrix_with_singlevalue(ll2->_weights, 3);

    input.copy_hd();
    ll1->_weights.copy_hd();
    ll2->_weights.copy_hd();

    nn.add_layer(ll1);
    nn.add_layer(rl);
    nn.add_layer(ll2);

    Matrix output = nn.forward(input);
    output.copy_dh();

    assertm(output.d_mem != nullptr, "FAILED no output");
    assertm(output.dim.x == output_dim.x, "FAILED wrong dims");
    assertm(output.dim.y == output_dim.y, "FAILED wrong dims");


    for (int i = 0; i < output.dim.x; i++)
    {
        for (int j = 0; j < output_dim.y; j++)
        {
            std::cout << output[i*output_dim.y + j] << " ";
            assertm(output[i*output_dim.y + j] == 1920, "FAILED wrong values");
        }

        std::cout << std::endl;
    }

    std::cout << "PASSED\n";
}

int main()
{
    NeuralNetwork nn;

    Matrix input;
    Matrix fake;
    Matrix real;

    test_layer_order(nn);
    std::cout << std::endl;

    test_forward();
    std::cout << std::endl;
}