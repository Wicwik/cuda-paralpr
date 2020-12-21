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
        std::default_random_engine gen(rd());
        std::uniform_real_distribution<float> dist;

        for (int i = 0; i < number_of_batches; i++)
        {
            _features.push_back(Matrix(MatDim{batch_size, 2}));
            _classes.push_back(Matrix(MatDim{batch_size, 1}));

            _features[i].allocate_mem();
            _classes[i].allocate_mem();

            for (int j = 0; j < batch_size; j++)
            {
                _features[i][j] = dist(gen);
                _features[i][_features.dim.x + j] = dist(gen);

                if((_features[i][j] < 0 && _features[i][_features.dim.x + j] < 0) || (_features[i][j] > 0 && _features[i][_features.dim.x + j] > 0))
                {
                    _classes[i] = 1;
                }
                else
                {
                    _classes[i] = 0;
                }
            }
        }

        _features.copy_hd();
        _classes.copy_hd();
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