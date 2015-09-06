//! Finds optimal block size for a kernel based on the max occupancy heuristic
//! __Never__ include this file. (it is needed for the services library)
///////////////////////////////////////////////////////////////////////////////
// 
// "Hemi" CUDA Portable C/C++ Utilities
// 
// Copyright 2012-2014 NVIDIA Corporation
//
// License: BSD License, see LICENSE file in Hemi home directory
//
// The home for Hemi is https://github.com/harrism/hemi
//
///////////////////////////////////////////////////////////////////////////////
// Please see the file README.md (https://github.com/harrism/hemi/README.md) 
// for fullManual documentation and discussion.
///////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_LAUNCH_CONFIGURATION_CUH
#define CUDA_LAUNCH_CONFIGURATION_CUH

#include "cuda_execution_policy.h"
#include <functional>
#include <cuda_runtime_api.h>
//#include <cuda_occupancy.h>

/*! Convenience function for checking CUDA runtime API results when `DEBUG` macro is defined.
  Can be wrapped around any runtime API call, no-op in release builds.
*/
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

namespace cuda{

/*size_t availableSharedBytesPerBlock(size_t sharedMemPerMultiprocessor,
                                    size_t sharedSizeBytesStatic,
                                    int blocksPerSM, int smemAllocationUnit)
{
  size_t bytes = __occRoundUp(sharedMemPerMultiprocessor / blocksPerSM, 
                              smemAllocationUnit) - smemAllocationUnit;
  return bytes - sharedSizeBytesStatic;    
}*/

/*! Automatically configures ExecutionPolicy objects.
  _Shouldn't be called explicitly_ , but through the `cuda::AutoConfig()()` wrapper
  @sa cuda_execution_policy.h cuda::AutoConfig
*/
template <typename F>
inline cudaError_t configurePolicy(ExecutionPolicy& p, F&& kernel, int totalThreads= 1,
                      size_t dynamicSMemSize= 0, int blockSizeLimit= 0)
{
  int configState = p.getConfigState();
  if (configState == ExecutionPolicy::FullManual) return cudaSuccess;

  int suggestedBlockSize=0, minGridSize=0;
  cudaError_t status= cudaSuccess;
  if ((configState & ExecutionPolicy::BlockSize) == 0) {
    status= cudaOccupancyMaxPotentialBlockSize(&minGridSize, &suggestedBlockSize,
                                             kernel, dynamicSMemSize, blockSizeLimit);
    if (status != cudaSuccess) return status;
    p.setBlockSize({static_cast<unsigned>(suggestedBlockSize),1,1});
  }
  if ((configState & ExecutionPolicy::GridSize) == 0)
    p.setGridSize({(totalThreads+p.getBlockSize().x-1)/p.getBlockSize().x,1,1});
  /*if ((configState & ExecutionPolicy::SharedMem) == 0) {
    int smemGranularity = 0;
    cudaOccError occErr = cudaOccSMemAllocationGranularity(&smemGranularity, &occProp);
    if (occErr != CUDA_OCC_SUCCESS) return cudaErrorInvalidConfiguration;
    size_t sbytes = availableSharedBytesPerBlock(props.sharedMemPerBlock,
                                                 attribs.sharedSizeBytes,
                                                 __occDivideRoundUp(p.getGridSize(), numSMs),
                                                 smemGranularity);
    p.setSharedMemBytes(sbytes);
  }*/
#if defined(DEBUG) || defined(_DEBUG)
  printf("%d %d %ld\n", p.getBlockSize(), p.getGridSize(), p.getSharedMemBytes());
#endif
  return cudaSuccess;
}

}   // namespace cuda

#endif // CUDA_LAUNCH_CONFIGURATION_CUH
