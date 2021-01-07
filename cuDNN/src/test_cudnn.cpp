#include <iostream>
#include <cuda.h>
#include <cudnn.h>

int main() 
{

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);
  cudnnDestroy(cudnn);

  return 0;
}
