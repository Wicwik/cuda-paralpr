# CUDA basics

## Compile and run
`g++ test_cpu.cpp && ./a.out`
`nvcc cuda_example0.cu && ./a.out`
`nvcc cuda_example1.cu && ./a.out`
`nvcc cuda_example2.cu && ./a.out`

## Perfomance
`nvcc cuda_example0.cu && nvprof ./a.out`
`nvcc cuda_example1.cu && nvprof ./a.out`
`nvcc cuda_example2.cu && nvprof ./a.out`
