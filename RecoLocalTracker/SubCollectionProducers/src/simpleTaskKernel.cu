#include "FWCore/Services/interface/utils/cuda_launch_configuration.cuh"

// <name>_kernel(...)
__global__ void simpleTask_kernel(unsigned meanExp, float* cls, float* clx, float* cly)
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

//@@@@@@@@@@@@@@@@@@@@@@@@@@

// <name>_auto(launchSize, ...) (1D)
void simpleTask_auto(unsigned& launchSize, unsigned meanExp, float* cls, float* clx, float* cly)
{
  auto execPol= cuda::AutoConfig()(launchSize, (void*)simpleTask_kernel);
  simpleTask_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(
                  meanExp, cls, clx, cly);
}
// <name>(execPol, ...)
void simpleTask_man(const cuda::ExecutionPolicy& execPol,
                    unsigned meanExp, float* cls, float* clx, float* cly)
{
  simpleTask_kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(
                    meanExp, cls, clx, cly);
}
////////////////////////////////////////////////////////////////////////////////
//class KernelWrap{};
//  simpleTask_auto(launchSize,...)= 
// template<typename F, typename... Args>
// autoLaunch(size, F&& kernel, Args... args){
//   auto execPol= cuda::autoConfig<???>(true, launchSize, (???)simpleTask_kernel);


//   cudaLaunchKernel()
// }


// cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, 
//                   void** args, size_t sharedMem, cudaStream_t stream )

//@@@@@@@@@@@@@@@@@@@@@@@@@
// #define GenerateKernelWrappers(kernelName, args...) \
// void kernelName ## _auto(int launchSize, ##args){\
//   auto execPol= cuda::autoConfig(true, launchSize, kernelName ## _kernel);\
//   kernelName ## _kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(args);\
// }\
// void name(const cuda::ExecutionPolicy execPol, ##args){\
//   name ## _kernel<<<execPol.getGridSize(), execPol.getBlockSize()>>>(args);\
// }\
// cuda::ExecutionPolicy name ## _config(bool cudaStatus, int launchSize){\
//   return cuda::autoConfig(cudaStatus, launchSize, name ## _kernel);\
// }
//@@@@@@@@@@@@@@@@@@@@@@@@@
