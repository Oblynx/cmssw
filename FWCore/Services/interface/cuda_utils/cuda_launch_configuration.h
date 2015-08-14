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
#ifndef CUDA_LAUNCH_CONFIGURATION_H
#define CUDA_LAUNCH_CONFIGURATION_H

//#include <cuda_occupancy.h>
#include "cuda_execution_policy.h"


// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
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

namespace cudaConfig{

/*size_t availableSharedBytesPerBlock(size_t sharedMemPerMultiprocessor,
                                    size_t sharedSizeBytesStatic,
                                    int blocksPerSM, int smemAllocationUnit)
{
  size_t bytes = __occRoundUp(sharedMemPerMultiprocessor / blocksPerSM, 
                              smemAllocationUnit) - smemAllocationUnit;
  return bytes - sharedSizeBytesStatic;    
}*/

template <typename F>
cudaError_t configure(ExecutionPolicy& p, F&& kernel, int totalThreads= 1,
                      size_t dynamicSMemSize= 0, int blockSizeLimit= 0)
{
  int configState = p.getConfigState();
  if (configState == ExecutionPolicy::FullManual) return cudaSuccess;

  int suggestedBlockSize, minGridSize;
  cudaError_t status;
  if ((configState & ExecutionPolicy::BlockSize) == 0) {
    status= cudaOccupancyMaxPotentialBlockSize(&minGridSize, &suggestedBlockSize,
                                               kernel, dynamicSMemSize, blockSizeLimit);
    if (status != cudaSuccess) return status;
    p.setBlockSize({suggestedBlockSize,1,1});
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

}   // namespace cudaConfig

#endif // CUDA_LAUNCH_CONFIGURATION_H
