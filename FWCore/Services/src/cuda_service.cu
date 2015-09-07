//! CudaService non-template method definitions
#include "FWCore/Services/interface/cuda_service.h"

namespace edm{namespace service{
CudaService::CudaService(const edm::ParameterSet& pSet, edm::ActivityRegistry& actR):
  cudaDevCount_(0)
{
  std::cout<<"[CudaService]: Constructing CudaService\n";
  /**Checking presence of GPU**/
  int deviceCount = 0;
  cudaError_t error_id= cudaSuccess;
  error_id= cudaGetDeviceCount(&deviceCount);
  //DANGER: Uncomment only for testing fallbacks on a machine with GPU
          // /*!!!*/deviceCount=0;/*!!!*/
  if (error_id == cudaErrorNoDevice || deviceCount == 0){
    std::cout<<"[CudaService]: No device available!\n";
    cudaDevCount_= 0; cuda::GPUPresenceStatic::setStatus(this, false);
    gpuFreeMem_= 0, gpuTotalMem_= 0;
  }else{
    cudaDevCount_= deviceCount; cuda::GPUPresenceStatic::setStatus(this, true);
    size_t gpuFreeMem= 0, gpuTotalMem= 0;
    cudaMemGetInfo(&gpuFreeMem, &gpuTotalMem);
    gpuFreeMem_= gpuFreeMem, gpuTotalMem_= gpuTotalMem;
  }
  if (pSet.exists("thread_num"))
    threadNum_= pSet.getParameter<int>("thread_num");
  //Default thread number: if (gpu) then (=4*gpu), else (=hardware_concurrency)
  if(!threadNum_)
    threadNum_= (cudaDevCount_)? 4*cudaDevCount_:
                                 std::thread::hardware_concurrency();
  
  actR.watchPostBeginJob(this, &ThreadPool::startWorkers);
  actR.watchPostEndJob(this, &ThreadPool::stopWorkers);
}

void ThreadPool::startWorkers()
{
  // continue only if !beginworking
  if (beginworking_.test_and_set()) return;

  stop_= false;
  if(!threadNum_) throw std::invalid_argument("[CudaService]: More than zero threads expected");
  workers_.reserve(threadNum_);
  for(; threadNum_; --threadNum_)
    workers_.emplace_back([this] (){
      while(!stop_)
      {
        std::function<void()> task;
        try{
          tasks_.pop(task);
          task();
        }catch(tbb::user_abort){
          // Normal control flow when the destructor is called
        }catch(...){
          std::cout << "[CudaService]: Unhandled exception!\n";
          throw;
        }
      }
    });
  endworking_.clear();
}

void ThreadPool::stopWorkers()
{
  // continue only if !endworking
  if (endworking_.test_and_set()) return;

  stop_= true;
  tasks_.abort();
  for(std::thread& worker: workers_)
    worker.join();
  workers_.clear();
  beginworking_.clear();
}

}} // namespace edm::service
