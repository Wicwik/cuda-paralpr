# Simple neural net implemetation
This implementation was inspired by [this](https://luniak.io/cuda-neural-network-implementation-part-1/) article.

## Compile and run
`nvcc test_bce.cu && ./a.out`
`nvcc test_linear.cu && ./a.out`
`nvcc test_matrix.cu && ./a.out`
`nvcc test_nn.cu && ./a.out`
`nvcc test_relu.cu && ./a.out`
`nvcc test_sigmoid.cu && ./a.out`
`nvcc main.cu && ./a.out`

## Perfomance
`nvcc main.cu && nvprof ./a.out`