#include "lib/neural_network.hh"
#include "lib/layers/linear.hh"
#include "lib/layers/relu_act.hh"
#include "lib/layers/leaky_relu_act.hh"
#include "lib/layers/sigmoid_act.hh"
#include "lib/loss_function/bce.hh"
#include "lib/datasets/random_points.hh"

float acc(const Matrix& fake, const Matrix &real)
{
    srand( time(NULL) );
    int true_positives = 0;

    for (int i = 0; i < fake.dim.x; i++)
    {
        float _fake = -1.0f;

        if (fake[i] > 0.5)
        {
            _fake = 1;
        }
        else
        {
            _fake = 0;
        }

        // std:: cout << _fake << " " << real[i] << std::endl;

        if (_fake == real[i])
        {
            true_positives++;
        }
    }

    // std::cout << true_positives << std::endl;

    return static_cast<float>(true_positives)/fake.dim.x;
}

int main()
{
    int epochs = 1001;
    int batches = 21;
    BCE bce_cf;

    RandomPoints dataset(100, batches);

    NeuralNetwork nn;

    nn.add_layer(new LinearLayer("Input-linear", MatDim{2, 30}));
    nn.add_layer(new ReluLayer("Hidden-ReLU"));
    nn.add_layer(new LinearLayer("Hidden-linear2", MatDim{30, 1}));
    nn.add_layer(new SigmoidLayer("Output-Sigmoid"));

    Matrix tmp;
    for (int i = 0; i < epochs; i++)
    {
        float cost = 0.0f;

        for (int j = 0; j < batches-1; j++)
        {
            tmp = nn.forward(dataset.get_features().at(j));
            nn.backprop(tmp, dataset.get_classes().at(j));

            cost += bce_cf.cost(tmp, dataset.get_classes().at(j));
        }


        if (!(i % 100))
        {

            std::cout << "Epoch: " << i << " | Cost: " << cost/batches << std::endl;

            // tmp.copy_dh();
            // for (int c = 0; c < tmp.dim.x; c++)
            // {
            //     std::cout << "[" << c << "] ";
            //     for (int r = 0; r < tmp.dim.y; r++)
            //     {
            //         std::cout << tmp[c*tmp.dim.y + r] << " ";
            //     }

            //     std::cout << std::endl;
            // }
        }
    }

    Matrix test = nn.forward(dataset.get_features().at(batches-1));
    test.copy_dh();
    std::cout << "Acc: " << acc(test, dataset.get_classes().at(batches-1)) << std::endl;


    std::cout << "end of test" << std::endl;

    return 0;
}