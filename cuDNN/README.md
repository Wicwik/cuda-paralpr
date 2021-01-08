# cuDNN library
This implementation was inspired by [this](http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/) article.

## Compile and run
- ``nvcc -lcudart -lcublas -lcudnn `pkg-config --cflags --libs opencv` && ./a.out``
- ``nvcc -lcudart -lcublas -lcudnn `pkg-config --cflags --libs opencv` && ./a.out``
- ``nvcc -lcudart -lcublas -lcudnn `pkg-config --cflags --libs opencv` convolutions.cpp && ./a.out``

## Perfomance
- ``nvcc -lcudart -lcublas -lcudnn `pkg-config --cflags --libs opencv` convolutions.cpp && nvprof ./a.out``
