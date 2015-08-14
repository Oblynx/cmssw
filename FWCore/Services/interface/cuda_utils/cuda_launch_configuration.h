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

#include <cuda_occupancy.h>
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

size_t availableSharedBytesPerBlock(size_t sharedMemPerMultiprocessor,
                                    size_t sharedSizeBytesStatic,
                                    int blocksPerSM,
                                    int smemAllocationUnit)
{
    size_t bytes = __occRoundUp(sharedMemPerMultiprocessor / blocksPerSM, 
                                smemAllocationUnit) - smemAllocationUnit;
    return bytes - sharedSizeBytesStatic;    
}

template <typename KernelFunc>
cudaError_t configureGrid(ExecutionPolicy &p, KernelFunc k)
{
    int configState = p.getConfigState();

    if (configState == ExecutionPolicy::FullManual) return cudaSuccess;

    cudaDeviceProp props;
    cudaFuncAttributes attribs;
    
    int devId;
    cudaError_t status = cudaGetDevice(&devId);
    if (status != cudaSuccess) return status;
    status = cudaGetDeviceProperties(&props, devId);
    if (status != cudaSuccess) return status;
    cudaOccDeviceProp occProp(props);

    status = cudaFuncGetAttributes(&attribs, k);
    if (status != cudaSuccess) return status;
    cudaOccFuncAttributes occAttrib(attribs);
    
    cudaFuncCache cacheConfig;
    status = cudaDeviceGetCacheConfig(&cacheConfig);
    if (status != cudaSuccess) return status;
    cudaOccDeviceState occState;
    occState.cacheConfig = (cudaOccCacheConfig)cacheConfig;

    int numSMs = props.multiProcessorCount;

    if ((configState & ExecutionPolicy::BlockSize) == 0) {
        int bsize = 0, minGridSize = 0;
        cudaOccError occErr = cudaOccMaxPotentialOccupancyBlockSize(&minGridSize,
                                                                    &bsize,
                                                                    &occProp, 
                                                                    &occAttrib, 
                                                                    &occState, 
                                                                    p.getSharedMemBytes());
        if (occErr != CUDA_OCC_SUCCESS || bsize < 0) return cudaErrorInvalidConfiguration;
        p.setBlockSize(bsize);
    }

    if ((configState & ExecutionPolicy::GridSize) == 0) {
        cudaOccResult result;
        cudaOccError occErr = cudaOccMaxActiveBlocksPerMultiprocessor(&result,
                                                                      &occProp, 
                                                                      &occAttrib, 
                                                                      &occState,
                                                                      p.getBlockSize(), 
                                                                       p.getSharedMemBytes());
        if (occErr != CUDA_OCC_SUCCESS) return cudaErrorInvalidConfiguration;
        p.setGridSize(result.activeBlocksPerMultiprocessor * numSMs);
        if (p.getGridSize() < numSMs) return cudaErrorInvalidConfiguration;
    }

    if ((configState & ExecutionPolicy::SharedMem) == 0) {

        int smemGranularity = 0;
        cudaOccError occErr = cudaOccSMemAllocationGranularity(&smemGranularity, &occProp);
        if (occErr != CUDA_OCC_SUCCESS) return cudaErrorInvalidConfiguration;
        size_t sbytes = availableSharedBytesPerBlock(props.sharedMemPerBlock,
                                                     attribs.sharedSizeBytes,
                                                     __occDivideRoundUp(p.getGridSize(), numSMs),
                                                     smemGranularity);
        p.setSharedMemBytes(sbytes);
    }

#if defined(DEBUG) || defined(_DEBUG)
    printf("%d %d %ld\n", p.getBlockSize(), p.getGridSize(), p.getSharedMemBytes());
#endif
    
    return cudaSuccess;
}
	
}   // namespace cudaConfig

#endif // CUDA_LAUNCH_CONFIGURATION_H
