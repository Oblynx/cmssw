#include "FWCore/Services/interface/utils/cuda_launch_configuration.cuh"
#include "FWCore/Services/interface/utils/GPU_presence_static.h"

//Wrapper for auto kernel launch config that can be called from .cc
//Needs kernel function for argument
namespace cuda{
ExecutionPolicy AutoConfig::operator()(int totalThreads, const void* f){
  cuda::ExecutionPolicy execPol;
  if(cuda::GPUPresenceStatic::getStatus(this))
    configurePolicy(execPol, f, totalThreads);
  return execPol;		
}
}  //namespace cuda
