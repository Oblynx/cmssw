/*
CMSSW CUDA management and Thread Pool Service
Author: Konstantinos Samaras-Tsakiris, kisamara@auth.gr
*//*
  --> Thread Pool:
Copyright (c) 2012 Jakob Progsch, Václav Zeman
This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:
   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
   3. This notice may not be removed or altered from any source
   distribution.

--> This is an altered version of the original code.
*/

#ifndef Thread_Pool_Service_H
#define Thread_Pool_Service_H

#include <iostream>

#include <vector>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include <functional>
#include <stdexcept>
#include <cuda_runtime_api.h>

#include <tbb/concurrent_queue.h>

#include "utils/cuda_launch_configuration.h"
#include "utils/cuda_execution_policy.h"
#include "utils/cuda_pointer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"


namespace edm{
namespace service{

/* Why not a singleton:
http://jalf.dk/blog/2010/03/singletons-solving-problems-you-didnt-know-you-never-had-since-1995/
*/
class ThreadPoolService {
public:
  //!< @brief Checks CUDA and registers callbacks
  ThreadPoolService(const edm::ParameterSet&, edm::ActivityRegistry& actR);
  // deleted copy&move ctors&assignments
	ThreadPoolService(const ThreadPoolService&) = delete;
	ThreadPoolService& operator=(const ThreadPoolService&) = delete;
	ThreadPoolService(ThreadPoolService&&) = delete;
	ThreadPoolService& operator=(ThreadPoolService&&) = delete;
  static void fillDescriptions(edm::ConfigurationDescriptions& descr){
    descr.add("ThreadPoolService", edm::ParameterSetDescription());
  }

  //!< @brief Schedule task and get its future handle
  template<typename F, typename... Args>
	inline std::future<typename std::result_of<F(Args...)>::type>
    getFuture(F&& f, Args&&... args)
  {
    using packaged_task_t = std::packaged_task<typename std::result_of<F(Args...)>::type ()>;

    std::shared_ptr<packaged_task_t> task(new packaged_task_t(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    ));
    auto resultFut = task->get_future();
    tasks_.emplace([task](){ (*task)(); });
    return resultFut;
  }
  // Configure execution policy before launch!
  //!< @brief Launch kernel function with args
  template<typename F, typename... Args>
  inline std::future<cudaError_t>
    cudaLaunchManaged(const cudaConfig::ExecutionPolicy& execPol, F&& f, Args&&... args)
  {
    using packaged_task_t = std::packaged_task<cudaError_t()>;

    std::shared_ptr<packaged_task_t> task(new packaged_task_t([&]()-> cudaError_t{
        cudaError_t status;
        #ifdef __NVCC__
          f<<<execPol.getGridSize(), execPol.getBlockSize(),
              execPol.getSharedMemBytes()>>>(utils::passKernelArg<Args>(args)...);
        #endif
        status= cudaStreamSynchronize(cudaStreamPerThread);
        return status;
    }));
    std::future<cudaError_t> resultFut= task->get_future();
    tasks_.emplace([task](){ (*task)(); });
    return resultFut;
  }

  template<typename F>
  static cudaConfig::ExecutionPolicy configureLaunch(int totalThreads, F&& f){
    cudaConfig::ExecutionPolicy execPol;
    checkCuda(cudaConfig::configure(execPol, std::forward<F>(f), totalThreads));
    return execPol;
  }
  //!< @brief Clears tasks queue
  void clearTasks(){ tasks_.clear(); }
	virtual ~ThreadPoolService(){
    std::cout << "---| Destroying service |---\n";
    stopWorkers();
  }
  //!< @brief Constructs workers and sets them spinning
  void startWorkers();
  //!< @brief Joins all worker threads
  void stopWorkers();
private:
  size_t threadNum= 0;
  // need to keep track of threads so we can join them
	std::vector< std::thread > workers_;
  // the task concurrent queue
	tbb::concurrent_bounded_queue< std::function<void()> > tasks_;

  // workers_ finalization flag
  std::atomic_bool stop_;
  std::atomic_flag beginworking_;   //init: false
  std::atomic_flag endworking_;    //init: true
  // {F,T}: not working, {T,T}: transition, {T,F}: working
  std::atomic_bool cuda_;
};

ThreadPoolService::ThreadPoolService(const edm::ParameterSet& pSet, edm::ActivityRegistry& actR):
  stop_(false), cuda_(false)
{
  beginworking_.clear(); endworking_.test_and_set();
  std::cout<<"Constructing ThreadPoolService\n";
  /**Checking presence of GPU**/
  int deviceCount = 0;
  cudaError_t error_id= cudaSuccess;
  #ifdef __NVCC__
    error_id= cudaGetDeviceCount(&deviceCount);
  #endif
  if (error_id == cudaErrorNoDevice || deviceCount == 0){
    std::cout<<"No device available!\n";
    cuda_= false;
  } else cuda_= true;
  //size_t threadNum = 4*deviceCount;
            /*DEBUG*/ if (deviceCount==0) return;
  if (pSet.exists("thread_num"))
    threadNum= pSet.getParameter<int>("thread_num");
  actR.watchPostBeginJob(this, &ThreadPoolService::startWorkers);
  actR.watchPostEndJob(this, &ThreadPoolService::stopWorkers);
}

void ThreadPoolService::startWorkers()
{
  // continue only if !beginworking
  if (beginworking_.test_and_set()) return;

  //Default thread number
  if(!threadNum) threadNum = std::thread::hardware_concurrency();
  if(!threadNum)
    throw std::invalid_argument("more than zero threads expected");
  workers_.reserve(threadNum);
  for(; threadNum; --threadNum)
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
          std::cout << "Unhandled exception!\n";
          throw;
        }
      }
    });
  endworking_.clear();
}

void ThreadPoolService::stopWorkers()
{
  // continue only if !endworking
  if (endworking_.test_and_set()) return;

  stop_= true;
  tasks_.abort();
  for(std::thread& worker: workers_)
    worker.join();
  beginworking_.clear();
}

} // namespace service
} // namespace edm

#endif // Thread_Pool_Service_H
