//! Defines `cuda::ExecutionPolicy` and a factory that makes these objects
//! __Must__ be included in .cu files _defining kernel wrappers._ Included with cuda_service.h
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
// for full documentation and discussion.
///////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_EXECUTION_POLICY_H
#define CUDA_EXECUTION_POLICY_H

#include <cuda_runtime_api.h>

namespace cuda{
  //! Stores the launch configuration for kernel launches
  class ExecutionPolicy {
  public:
    enum ConfigurationState {
      Automatic = 0,
      SharedMem = 1,
      BlockSize = 2,
      GridSize = 4,
      FullManual = GridSize | BlockSize | SharedMem
    };
    //! Default constructor
    ExecutionPolicy(dim3 blockSize= {0,0,0}, dim3 gridSize= {0,0,0}, size_t sharedMemBytes=0)
    : state_(0) {
      setGridSize(gridSize);
      setBlockSize(blockSize);
      setSharedMemBytes(sharedMemBytes);  
    }
    //!@{
    //! Move semantics
    ExecutionPolicy(ExecutionPolicy&&) =default;
    ExecutionPolicy& operator=(ExecutionPolicy&&) =default;
    //!@}
    //!@{
    int    getConfigState()    const { return state_;          }
    dim3   getGridSize()       const { return gridSize_;       }
    dim3   getBlockSize()      const { return blockSize_;      }
    int    getMaxBlockSize()   const { return maxBlockSize_;   }
    size_t getSharedMemBytes() const { return sharedMemBytes_; }
    //!@}
    //!@{
    //! Setter methods can be chained.
    ExecutionPolicy& setGridSize(const dim3 arg) { 
      gridSize_ = arg;
      if (gridSize_.x > 0) state_ |= GridSize; 
      else state_ &= (FullManual - GridSize);
      return *this;
    }
    ExecutionPolicy& setBlockSize(const dim3 arg) {
      blockSize_ = arg; 
      if (blockSize_.x > 0) state_ |= BlockSize; 
      else state_ &= (FullManual - BlockSize);
      return *this;
    }
    ExecutionPolicy& setMaxBlockSize(const int arg) {
    	maxBlockSize_ = arg;
      return *this;
    }
    ExecutionPolicy& setSharedMemBytes(const size_t arg) { 
      sharedMemBytes_ = arg; 
      state_ |= SharedMem; 
      return *this;
    }
    //!@}
    /*! If blockSize is already set, calculates required grid size based on problem size
      @param threadN Number of threads required in each dimension (problem size)
    */
    ExecutionPolicy& autoGrid(const dim3 threadN){
      gridSize_.x= (blockSize_.x>0)? (threadN.x-1)/blockSize_.x+1: 0;
      gridSize_.y= (blockSize_.y>0)? (threadN.y-1)/blockSize_.y+1: 0;
      gridSize_.z= (blockSize_.z>0)? (threadN.z-1)/blockSize_.z+1: 0;
      return *this;
    }
  private:
    int    state_;
    dim3   gridSize_;
    dim3   blockSize_;
    int    maxBlockSize_;
    size_t sharedMemBytes_;
  };

  //! Functor for auto kernel launch config that can be called from normal c++ files.
  struct AutoConfig{
    ExecutionPolicy operator()(int totalThreads, const void* kernel);
  };
}   // namespace cuda

#endif // CUDA_EXECUTION_POLICY_H
