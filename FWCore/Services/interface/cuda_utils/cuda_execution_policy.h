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

namespace cudaConfig{

class ExecutionPolicy {
public:
    enum ConfigurationState {
        Automatic = 0,
        SharedMem = 1,
        BlockSize = 2,
        GridSize = 4,
        FullManual = GridSize | BlockSize | SharedMem
    };

    ExecutionPolicy(dim3 blockSize= {0,0,0}, dim3 gridSize= {0,0,0}, size_t sharedMemBytes=0)
    : state_(0) {
      setGridSize(gridSize);
      setBlockSize(blockSize);
      setSharedMemBytes(sharedMemBytes);  
    }
    ExecutionPolicy(ExecutionPolicy&&) =default;
    ExecutionPolicy& operator=(ExecutionPolicy&&) =default;
    ~ExecutionPolicy() {}

    int    getConfigState()    const { return state_;          }
    dim3   getGridSize()       const { return gridSize_;       }
    dim3   getBlockSize()      const { return blockSize_;      }
    int    getMaxBlockSize()   const { return maxBlockSize_;   }
    size_t getSharedMemBytes() const { return sharedMemBytes_; }
 
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

private:
    int    state_;
    dim3   gridSize_;
    dim3   blockSize_;
    int    maxBlockSize_;
    size_t sharedMemBytes_;
};

}   // namespace cudaConfig

#endif // CUDA_EXECUTION_POLICY_H
