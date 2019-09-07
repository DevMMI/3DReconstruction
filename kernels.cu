#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "kernels.h"
__global__ void my_kernel(){
  printf("hello world\n");
}



void myfunc(){
  my_kernel<<<1,1>>>();
}
