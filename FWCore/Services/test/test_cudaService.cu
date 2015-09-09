//! Kernels and kernel wrappers used by the CudaService test suite
//! @sa test_cudaService_gcc.cppunit.cc
#include "FWCore/Services/interface/utils/cuda_execution_policy.h"
#include "FWCore/Services/interface/utils/cuda_pointer.h"

#define BLOCK_SIZE 32

//@@@@@@@@@@@@@@@@ KERNELS
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

//@@@@@@@@@@@@@@@@ AUTO WRAPPERS (without fallbacks)
  void long_auto(bool gpu, unsigned& launchSize,
                 const int n, const int times, const float* in, float* out){
    auto execPol= cuda::AutoConfig()(launchSize, (void*)long_kernel);
    if(gpu) long_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(n,times,in,out);
  }
  void matAdd_auto(bool gpu, unsigned& launchSize,
                   int m, int n, const float* __restrict__ A, 
                   const float* __restrict__ B, float* __restrict__ C){
    auto execPol= cuda::AutoConfig()(launchSize, (void*)matAdd_kernel);
    if(gpu) matAdd_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(m,n,A,B,C);
  }
  void original_auto(bool gpu, unsigned& launchSize,
                     unsigned meanExp, float* cls, float* clx, float* cly){
    auto execPol= cuda::AutoConfig()(launchSize, (void*)original_kernel);
    if(gpu) original_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(meanExp,cls,clx,cly);
  }
//@@@@@@@@@@@@@@@@ MANUAL WRAPPERS
  void long_man(bool gpu, const cuda::ExecutionPolicy& execPol,
                const int n, const int times, const float* in, float* out){
    if(gpu) long_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(n,times,in,out);
  }
  void matAdd_man(bool gpu, const cuda::ExecutionPolicy& execPol,
                  int m, int n, const float* __restrict__ A,
                  const float* __restrict__ B, float* __restrict__ C){
    if(gpu) matAdd_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(m,n,A,B,C);
  }
  void original_man(bool gpu, const cuda::ExecutionPolicy& execPol,
                    unsigned meanExp, float* cls, float* clx, float* cly){
    if(gpu) original_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(meanExp,cls,clx,cly);
  }

//@@@@@@@@@@@@@@@@ FALLBACK for "original_kernel"
void original_CPU(unsigned meanExp, float* cls, float* clx, float* cly)
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

//@@@@@@@@@@@@@@@@ Advanced cudaPointer tests
// Compound data structure containing cudaPointer
struct KernelData: ContainsCudaPointer{
  int a, b;
  cudaPointer<float[]> arrayIn;
  cudaPointer<float[]> arrayOut;
};
// "cudaPointer[]: calling host function from kernel"
// "illegal memory access encountered"?
__global__ void actOnStructKernel(KernelData* data){
  int i= blockDim.x*blockIdx.x+threadIdx.x;
  if (i < data->arrayIn.size(true))
    data->arrayOut.at(i)= data->arrayIn.at(i)+data->a*data->b;
}
void actOnStructWrapper(bool gpu, const cuda::ExecutionPolicy& execPol,
                        KernelData* data){
  if(gpu) actOnStructKernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(data);
}

// Simple arithmetic data structure
struct ArithmStruct{
  int a, b;
  float c;
};
__global__ void actOnArithmStructKernel(int n, ArithmStruct* inStruct, float* out){
  int i= blockDim.x*blockIdx.x+threadIdx.x;
  if (i < n){
    auto& tmp= inStruct[i];
    out[i]= sinf(tmp.c)+tmp.b-tmp.a;
  }
}
void actOnArithmStructWrapper(bool gpu, const cuda::ExecutionPolicy& execPol,
                              int n, ArithmStruct* inStruct, float* out){
  if (gpu) actOnArithmStructKernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(
            n, inStruct, out);
}
