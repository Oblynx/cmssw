#include "FWCore/Services/interface/cudaService_TBBQueueBlocking.h"

/**$$$~~~~~ CudaService non-template method definitions ~~~~~$$$**/
namespace edm{namespace service{
CudaService::CudaService(const edm::ParameterSet& pSet, edm::ActivityRegistry& actR):
  stop_(false), cudaDevCount_(0)
{
  beginworking_.clear(); endworking_.test_and_set();
  std::cout<<"[CudaService]: Constructing CudaService\n";
  /**Checking presence of GPU**/
  int deviceCount = 0;
  cudaError_t error_id= cudaSuccess;
  error_id= cudaGetDeviceCount(&deviceCount);
          // /*!!!*/deviceCount=0;/*!!!*/
  if (error_id == cudaErrorNoDevice || deviceCount == 0){
    std::cout<<"[CudaService]: No device available!\n";
    cudaDevCount_= 0; CudaStatusStatic::setStatus(this, false);
  }else{
    cudaDevCount_= deviceCount; CudaStatusStatic::setStatus(this, true);
  }
  if (pSet.exists("thread_num"))
    threadNum_= pSet.getParameter<int>("thread_num");
  
  actR.watchPostBeginJob(this, &CudaService::startWorkers);
  actR.watchPostEndJob(this, &CudaService::stopWorkers);
}

void CudaService::startWorkers()
{
  // continue only if !beginworking
  if (beginworking_.test_and_set()) return;

  size_t gpuFreeMem= 0, gpuTotalMem= 0;
  if (cudaDevCount_) cudaMemGetInfo(&gpuFreeMem,&gpuTotalMem);
  gpuFreeMem_= gpuFreeMem, gpuTotalMem_= gpuTotalMem;
  //Default thread number: if (gpu) then (=4*gpu), else (=hardware_concurrency)
  if(!threadNum_)
    threadNum_= (cudaDevCount_)? 4*cudaDevCount_:
                                 std::thread::hardware_concurrency();
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

void CudaService::stopWorkers()
{
  // continue only if !endworking
  if (endworking_.test_and_set()) return;

  stop_= true;
  tasks_.abort();
  for(std::thread& worker: workers_)
    worker.join();
  beginworking_.clear();
}

}} // namespace edm::service
