#include "FWCore/Services/interface/utils/cuda_launch_configuration.cuh"
#include "FWCore/Services/interface/utils/GPU_presence_static.h"

//Wrapper for auto kernel launch config that can be called from .cc
//Needs kernel function for argument
namespace cuda{
ExecutionPolicy AutoConfig::operator()(int totalThreads, const void* f){
  cuda::ExecutionPolicy execPol;
  if(cuda::GPUPresenceStatic::getStatus(this)){
    configurePolicy(execPol, f, totalThreads);
  }
  return execPol;		
}

}  //namespace cuda

namespace edm{namespace service{
__global__ void paramKernel(const int param, const int param2) {}
__global__ void inOutKernel(const int in, int* out) {*out= in;}
}}
