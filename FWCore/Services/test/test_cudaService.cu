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

//@@@@@@@@@@@@@@@@
  void long_auto(unsigned& launchSize, const int n, const int times, const float* in, float* out){
    auto execPol= cudaConfig::configure(true, launchSize, long_kernel);
    long_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(n,times,in,out);
  }
  void matAdd_auto(unsigned& launchSize, int m, int n, const float* __restrict__ A, 
                            const float* __restrict__ B, float* __restrict__ C){
    auto execPol= cudaConfig::configure(true, launchSize, matAdd_kernel);
    matAdd_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(m,n,A,B,C);
  }
  void original_auto(unsigned& launchSize, unsigned meanExp, float* cls, float* clx, float* cly){
    auto execPol= cudaConfig::configure(true, launchSize, original_kernel);
    original_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(meanExp,cls,clx,cly);
  }
//@@@@@@@@@@@@@@@@
  void long_man(const cudaConfig::ExecutionPolicy& execPol, const int n,
            const int times, const float* in, float* out){
    long_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(n,times,in,out);
  }
  void matAdd_man(const cudaConfig::ExecutionPolicy& execPol,int m, int n, const float*
              __restrict__ A, const float* __restrict__ B, float* __restrict__ C){
    matAdd_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(m,n,A,B,C);
  }
  void original_man(const cudaConfig::ExecutionPolicy& execPol,
                unsigned meanExp, float* cls, float* clx, float* cly){
    original_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(meanExp,cls,clx,cly);
  }
//@@@@@@@@@@@@@@@@

void cudaTaskImplement(int n, int i, const float* din, int times){
  float *dout;
  cudaMalloc((void **) &dout, n*sizeof(float));
  dim3 grid((n-1)/BLOCK_SIZE/BLOCK_SIZE+1);
  dim3 block(BLOCK_SIZE*BLOCK_SIZE);
  long_kernel<<<grid,block>>>(n, times, din, dout);
  cudaStreamSynchronize(cudaStreamPerThread);
  float out;
  cudaMemcpy(&out, dout+i, 1*sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "GPU::" << out << "\t";
  cudaFree(dout);
}
