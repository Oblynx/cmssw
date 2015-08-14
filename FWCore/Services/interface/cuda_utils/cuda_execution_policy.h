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

    ExecutionPolicy(int gridSize=0, int blockSize=0, size_t sharedMemBytes=0)
    : mState(0) {
      setGridSize(gridSize);
      setBlockSize(blockSize);
      setSharedMemBytes(sharedMemBytes);  
    }
          
    ~ExecutionPolicy() {}

    int    getConfigState()    const { return mState;          }
    
    int    getGridSize()       const { return mGridSize;       }
    int    getBlockSize()      const { return mBlockSize;      }
    int    getMaxBlockSize()   const { return mMaxBlockSize;   }
    size_t getSharedMemBytes() const { return mSharedMemBytes; }
 
    ExecutionPolicy& setGridSize(int arg) { 
        mGridSize = arg;  
        if (mGridSize > 0) mState |= GridSize; 
        else mState &= (FullManual - GridSize);
        return *this;
    }   
    ExecutionPolicy& setBlockSize(int arg) { mBlockSize = arg; 
        if (mBlockSize > 0) mState |= BlockSize; 
        else mState &= (FullManual - BlockSize);
        return *this;
    }
    ExecutionPolicy& setMaxBlockSize(int arg) {
    	mMaxBlockSize = arg;
        return *this;
    }
    ExecutionPolicy& setSharedMemBytes(size_t arg) { 
        mSharedMemBytes = arg; 
        mState |= SharedMem; 
        return *this;
    }

private:
    int    mState;
    int    mGridSize;
    int    mBlockSize;
    int    mMaxBlockSize;
    size_t mSharedMemBytes;
};

}   // namespace cudaConfig

#endif // CUDA_EXECUTION_POLICY_H
