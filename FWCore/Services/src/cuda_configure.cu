/*! CUDA definition of `cuda::AutoConfig` functor.
	It creates automatically configured `cuda::ExecutionPolicy` objects, based on
	the maximum occupancy heuristic
	*/
#include "FWCore/Services/interface/utils/cuda_launch_configuration.cuh"
#include "FWCore/Services/interface/utils/GPU_presence_static.h"

namespace cuda{
	/*! Wrapper for auto kernel launch config that can be called from normal c++ files.
		@param totalThreads Total number of threads the kernel should be launched with
		(problem size)

		Example use:
		`auto execPol= cuda::AutoConfig()(num_of_threads, (void*)kernel)`
	*/
	ExecutionPolicy AutoConfig::operator()(int totalThreads, const void* kernel){
	  cuda::ExecutionPolicy execPol;
	  if(cuda::GPUPresenceStatic::getStatus(this))
	    configurePolicy(execPol, kernel, totalThreads);
	  return execPol;		
	}
}  //namespace cuda
