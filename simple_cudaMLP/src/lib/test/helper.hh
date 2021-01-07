#pragma once

#include <random>
#include <cassert>
#include <vector>
#include <cmath>
#include "../matrix/matrix.hh"

#define assertm(exp, msg) assert(((void)msg, exp))

namespace helper
{
    void fill_matrix_with_singlevalue(Matrix X, float value)
    {
        for (int i = 0; i < X.dim.x; i++)
        {
            for (int j = 0; j < X.dim.y; j++)
            {
                X[i*X.dim.y + j] = value;
            }
        }
    }

    void fill_matrix_with_radom_range(Matrix X, float a, float b)
    {
        std::random_device rd; 
        std::mt19937 gen(rd()); 
        std::uniform_real_distribution<> dis(a, b);

        for (int i = 0; i < X.dim.x; i++)
        {
            for (int j = 0; j < X.dim.y; j++)
            {
                X[i*X.dim.y + j] = dis(gen);
            }
        }
    }

    void fill_cols(Matrix X, std::vector<float> values)
    {
        assertm(X.dim.x == values.size(), "Values vector must be the same length as matrix height");

        for (int i = 0; i < X.dim.x; i++)
        {
            for (int j = 0; j < X.dim.y; j++)
            {
                X[i*X.dim.y + j] = values[i];
            }
        }
    }

    void fill_rows(Matrix X, std::vector<float> values)
    {
        assertm(X.dim.y == values.size(), "Values vector must be the same length as matrix width");

        for (int i = 0; i < X.dim.x; i++)
        {
            for (int j = 0; j < X.dim.y; j++)
            {
                X[i*X.dim.y + j] = values[j];
            }
        }
    }

    float calc_sigmoid(float x)
    {
        return 1.0f / (1.0f + expf(-x));
    }

    bool cmpf(float A, float B, float epsilon = 0.005f)
    {
        return (fabs(A - B) < epsilon);
    }
}