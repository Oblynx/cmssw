// CUDA-specific test resources
#include <iostream>
#include "FWCore/Services/interface/utils/cuda_launch_configuration.cuh"

#define BLOCK_SIZE 32

__global__ void long_kernel(const int n, const int times, const float* in, float* out)
{
  int x= blockIdx.x*blockDim.x + threadIdx.x;
  if (x < n){
    out[x]= 0;
    for(int i=0; i<times; i++){
      out[x]+= in[x];
    }
  }
}
__global__ void matAdd_kernel(int m, int n, const float* __restrict__ A, 
                              const float* __restrict__ B, float* __restrict__ C)
{
  int x= blockIdx.x*blockDim.x + threadIdx.x;
  int y= blockIdx.y*blockDim.y + threadIdx.y;

  // ### Difference between manual and automatic kernel grid:
  if (x<n && y<m)
    C[y*n+x]= A[y*n+x]+B[y*n+x];
  //if (y*n+x < n*m)
    //C[y*n+x]= A[y*n+x]+B[y*n+x];
}
__global__ void original_kernel(unsigned meanExp, float* cls, float* clx, float* cly)
{
  int i= blockDim.x*blockIdx.x+threadIdx.x;
  if(i<meanExp){
    if (cls[i] != 0){
      clx[i] /= cls[i];
      cly[i] /= cls[i];
    }
    cls[i]= 0;
  }
}
__global__
void simpleTask_GPU(unsigned meanExp, float* cls, float* clx, float* cly)
{
  unsigned i= blockDim.x*blockIdx.x+threadIdx.x;
  if(i<meanExp){
    if (cls[i] != 0){
      clx[i] /= cls[i];
      cly[i] /= cls[i];
    }
    cls[i]= 0;
  }
}
void simpleTask_CPU(unsigned meanExp, float* cls, float* clx, float* cly)
{
  for (unsigned int subcl_idx = 0;
       subcl_idx < meanExp; subcl_idx++){
    if (cls[subcl_idx] != 0) {
      clx[subcl_idx] /= cls[subcl_idx];
      cly[subcl_idx] /= cls[subcl_idx];
    }
    cls[subcl_idx] = 0;
  }
}

//@@@@@@@@@@@@@@@@
  void long_auto(bool gpu, unsigned& launchSize,
                 const int n, const int times, const float* in, float* out){
    auto execPol= cuda::AutoConfig()(launchSize, (void*)long_kernel);
    long_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(n,times,in,out);
  }
  void matAdd_auto(bool gpu, unsigned& launchSize,
                   int m, int n, const float* __restrict__ A, 
                   const float* __restrict__ B, float* __restrict__ C){
    auto execPol= cuda::AutoConfig()(launchSize, (void*)matAdd_kernel);
    matAdd_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(m,n,A,B,C);
  }
  void original_auto(bool gpu, unsigned& launchSize,
                     unsigned meanExp, float* cls, float* clx, float* cly){
    auto execPol= cuda::AutoConfig()(launchSize, (void*)original_kernel);
    original_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(meanExp,cls,clx,cly);
  }
//@@@@@@@@@@@@@@@@
  void long_man(bool gpu, const cuda::ExecutionPolicy& execPol,
                const int n, const int times, const float* in, float* out){
    long_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(n,times,in,out);
  }
  void matAdd_man(bool gpu, const cuda::ExecutionPolicy& execPol,
                  int m, int n, const float* __restrict__ A,
                  const float* __restrict__ B, float* __restrict__ C){
    matAdd_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(m,n,A,B,C);
  }
  void original_man(bool gpu, const cuda::ExecutionPolicy& execPol,
                    unsigned meanExp, float* cls, float* clx, float* cly){
    original_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(meanExp,cls,clx,cly);
  }
