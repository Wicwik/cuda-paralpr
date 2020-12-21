#include "lib/neural_network.hh"
#include "lib/layers/linear.hh"
#include "lib/layers/leaky_relu_act.hh"
#include "lib/layers/sigmoid_act.hh"
#include "lib/loss_function/bce.hh"
#include "lib/datasets/random_points.hh"

int main()
{
    NeuralNetwork nn;
    nn.add_layer(new LinearLayer("Input-linear", MatDim{2, 30}));
    nn.add_layer(new LeakyReluLayer("Hidden-LReLU"));
    nn.add_layer(new LinearLayer("Hidden-linear", MatDim{30, 2}));
    nn.add_layer(new SigmoidLayer("Output-Sigmoid"));

    std::cout << "end of test" << std::endl;

    return 0;
}