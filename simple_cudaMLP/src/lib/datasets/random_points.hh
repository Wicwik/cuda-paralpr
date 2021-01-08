/*
 *   Generate random points dataset
 */


#include <random>
#include "../matrix/matrix.hh"

class RandomPoints
{

public:
    RandomPoints(int batch_size, int number_of_batches)
        : _batch_size{batch_size}
        , _number_of_batches{number_of_batches}
    {
        std::random_device rd;
        std::default_random_engine gen(rd()); // random on hardware
        std::uniform_real_distribution<float> dist(-1, 1); // real distribution

        for (int i = 0; i < number_of_batches; i++)
        {
            _features.push_back(Matrix(MatDim(_batch_size, 2)));
            _classes.push_back(Matrix(MatDim(_batch_size, 1)));

            _features[i].allocate_mem();
            _classes[i].allocate_mem();

            for (int j = 0; j < batch_size; j++)
            {
                _features[i][j] = dist(gen);
                _features[i][_features[i].dim.x + j] = dist(gen);

                // std::cout << _features[i][j] << " " << _features[i][_features[i].dim.x + j] << " ";

                if((_features[i][j] < 0 && _features[i][_features[i].dim.x + j] < 0) || (_features[i][j] > 0 && _features[i][_features[i].dim.x + j] > 0))
                {
                    _classes[i][j] = 1.0f;
                    // std::cout << 1 << std::endl;
                }
                else
                {
                    _classes[i][j] = 0.0f;
                    // std::cout << 0 << std::endl;
                }
            }

            _features[i].copy_hd();
            _classes[i].copy_hd();
        }
    }

    int get_number_of_batches()
    {
        return _number_of_batches;
    }

    std::vector<Matrix> get_features()
    {
        return _features;
    }

    std::vector<Matrix> get_classes()
    {
        return _classes;
    }

private:
    int _batch_size;
    int _number_of_batches;

    std::vector<Matrix> _features;
    std::vector<Matrix> _classes;
};