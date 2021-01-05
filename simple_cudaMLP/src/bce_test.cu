#include "lib/test/helper.hh"
#include "lib/loss_function/bce.hh"

void test_bce(Matrix real, Matrix fake)
{
    std::cout << "BCE COST FUNCTION TEST\n";

    BCE bce_cf;

    helper::fill_matrix_with_singlevalue(fake, 0.0001);
    helper::fill_matrix_with_singlevalue(real, 1);

    fake.copy_hd();
    real.copy_hd();

    float exptected = -log(0.0001);
    float cost = bce_cf.cost(fake, real);

    std::cout << "Calculated: " << cost << " Real value: " << exptected << std::endl;

    assertm(helper::cmpf(cost, exptected, 0.0001), "Values must be equal");
    std::cout << "PASSED\n";
}

void test_deriv_bce(Matrix real, Matrix fake)
{
    std::cout << "BCE COST DERIVATIVE FUNCTION TEST\n";

    BCE bce_cf;

    helper::fill_matrix_with_singlevalue(fake, 0.0001);
    helper::fill_matrix_with_singlevalue(real, 1);

    Matrix tmp{fake.dim};
    tmp.allocate_mem();

    fake.copy_hd();
    real.copy_hd();

    float exptected = (0.0001 - 1) / ((1 - 0.0001) * 0.0001);

    Matrix output = bce_cf.gradient(fake, real, tmp);
    output.copy_dh();

    for (int i = 0; i < output.dim.x; i++)
    {
        std::cout << "Calculated: " << output[i] << " Real value: " << exptected << std::endl;
        assertm(helper::cmpf(output[i], exptected, 0.0001), "Values must be equal");
    }
    std::cout << "PASSED\n";
}

int main()
{
    Matrix real{100, 1};
    real.allocate_mem();

    Matrix fake{100, 1};
    fake.allocate_mem();

    test_bce(real, fake);
    test_deriv_bce(real, fake);

    return 0;
}