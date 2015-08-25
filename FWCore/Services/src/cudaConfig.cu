#include "FWCore/Services/interface/utils/cuda_launch_configuration.cuh"

//Wrapper for auto kernel launch config that can be called from .cc
//Needs kernel function for argument
namespace cudaConfig{
cudaConfig::ExecutionPolicy configure(bool cudaStatus, int totalThreads, const void* f){
  cudaConfig::ExecutionPolicy execPol;
  if(cudaStatus)
    configurePolicy(execPol, f, totalThreads);
  return execPol;
}
}  //namespace cudaConfig
