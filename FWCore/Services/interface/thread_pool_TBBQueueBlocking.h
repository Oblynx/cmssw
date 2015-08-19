/*
CMSSW CUDA management and thread pool Service
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
#include "utils/template_utils.h"
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
  //!< Checks CUDA and registers callbacks
  ThreadPoolService(const edm::ParameterSet&, edm::ActivityRegistry& actR);
  // deleted copy&move ctors&assignments
	ThreadPoolService(const ThreadPoolService&) = delete;
	ThreadPoolService& operator=(const ThreadPoolService&) = delete;
	ThreadPoolService(ThreadPoolService&&) = delete;
	ThreadPoolService& operator=(ThreadPoolService&&) = delete;
  static void fillDescriptions(edm::ConfigurationDescriptions& descr){
    descr.add("ThreadPoolService", edm::ParameterSetDescription());
  }

  // Schedule task and get its future handle
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
  // Launch kernel function with args
  // Configure execution policy before launch!
  template<typename F, typename... Args>
  inline std::future<void>
    cudaLaunchManaged(const cudaConfig::ExecutionPolicy& execPol, F&& f, Args&&... args)
  {
    using packaged_task_t = std::packaged_task<void()>;

    std::shared_ptr<packaged_task_t> task(new packaged_task_t([&](){
      #ifdef __NVCC__
        f<<<execPol.getGridSize(), execPol.getBlockSize(),
            execPol.getSharedMemBytes()>>>(
                                              std::forward<Args>(args)...);
      #endif
        //std::cout<<"[In task]: Launched\n";
        cudaStreamSynchronize(cudaStreamPerThread);
        //std::cout<<"[In task]: Synced\n";
    }));
    std::future<void> resultFut = task->get_future();
    tasks_.emplace([task](){ (*task)(); });
    return resultFut;
  }

  /*template<typename T>
  ??? attachManagedMemory(T&& arg)
  {
    IFcudaPointer(std::forward<Head>(first));
    *//*struct CudaPtrArg{
      void operate(){

      }
    };
    struct NonCudaPtrArg{
      void operate(){}
    };
    std::conditional<std::is_same<Head, cudaPointer>, CudaPtrArg, NonCudaPtrArg>::
      type::operate(std::forward<Head>(first));*/
  //}

  template<typename F>
  static cudaConfig::ExecutionPolicy configureLaunch(int totalThreads, F&& f){
    cudaConfig::ExecutionPolicy execPol;
    checkCuda(cudaConfig::configure(execPol, std::forward<F>(f), totalThreads));
    return execPol;
  }
  // Clears tasks queue
  void clearTasks(){ tasks_.clear(); }
	virtual ~ThreadPoolService(){
    std::cout << "---| Destroying service |---\n";
    stopWorkers();
  }
  //!< @brief Constructs workers
  void startWorkers();
  //!< @brief Joins all threads
  void stopWorkers();
private:
  template<typename T, typename std::enable_if<std::is_base_of<cudaPtrBase, T>::value>::type>
  auto preprocessManagedMem(T&& cudaPtr) -> decltype(cudaPtr.p){
    std::cout<<"[memAttach]: Managed arg!\n";
    cudaPtr.attachStream();
    return cudaPtr.p;
  }
  template<typename T, typename std::enable_if<!std::is_base_of<cudaPtrBase, T>::value>::type>
  auto preprocessManagedMem(T&& valueArg) -> decltype(valueArg){
    //Do nothing
    std::cout<<"[memAttach]: value arg\n";
    return valueArg;
  }
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

ThreadPoolService::ThreadPoolService(const edm::ParameterSet&, edm::ActivityRegistry& actR):
  stop_(false), cuda_(false)
{
  beginworking_.clear(); endworking_.test_and_set();
  std::cout<<"Constructing ThreadPoolService\n";
  /**Checking presence of GPU**/
  int deviceCount = 0;
#ifdef __NVCC__
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id == cudaErrorNoDevice || deviceCount == 0){
    std::cout<<"No device available!\n";
    cuda_= false;
  } else cuda_= true;
#endif
  //size_t threads_n = 4*deviceCount;
            /*DEBUG*/ if (deviceCount==0) return;
  actR.watchPostBeginJob(this, &ThreadPoolService::startWorkers);
  actR.watchPostEndJob(this, &ThreadPoolService::stopWorkers);
}

void ThreadPoolService::startWorkers()
{
  // continue only if !beginworking
  if (beginworking_.test_and_set()) return;

  size_t threads_n = std::thread::hardware_concurrency();
  if(!threads_n)
    throw std::invalid_argument("more than zero threads expected");
  workers_.reserve(threads_n);
  for(; threads_n; --threads_n)
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
