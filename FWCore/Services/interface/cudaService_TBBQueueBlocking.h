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

#ifndef Cuda_Service_H
#define Cuda_Service_H

// Debug
#include <iostream>
#include <exception>

#include <vector>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include <functional>
#include <stdexcept>
#include <cuda_runtime.h>

#include <tbb/concurrent_queue.h>

#include "utils/cuda_execution_policy.h"
#include "utils/cuda_pointer.h"
#include "utils/GPU_presence_static.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

/**$$$~~~~~ CudaService class declaration ~~~~~$$$**/
namespace edm{namespace service{

  class ThreadPool{
  public:
    ThreadPool(): stop_(false) {
      beginworking_.clear(); endworking_.test_and_set();
    }
    ThreadPool(const ThreadPool&) =delete;
    ThreadPool& operator=(const ThreadPool&) =delete;
    ThreadPool(ThreadPool&&) =delete;
    ThreadPool& operator=(ThreadPool&&) =delete;
    
    //!< @brief Schedule task and get its future handle
    template<typename F, typename... Args>
    inline std::future<typename std::result_of<F(Args...)>::type>
      schedule(F&& f, Args&&... args)
    {
      using packaged_task_t = std::packaged_task<typename std::result_of<F(Args...)>::type ()>;

      std::shared_ptr<packaged_task_t> task(new packaged_task_t(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
      ));
      auto resultFut = task->get_future();
      tasks_.emplace([task](){ (*task)(); });
      return resultFut;
    }

    //!< @brief Clears tasks queue
    void clearTasks(){ tasks_.clear(); }
    //!< @brief Constructs workers and sets them waiting
    void startWorkers();
    //!< @brief Joins all worker threads
    void stopWorkers();
    virtual ~ThreadPool(){
      std::cout << "[ThreadPool]: ---| Destroying pool |---\n";
      stopWorkers();
    }
  protected:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers_;
    // the task concurrent queue
    tbb::concurrent_bounded_queue< std::function<void()> > tasks_;
    size_t threadNum_= 0;
  private:
    // workers_ finalization flag
    std::atomic_bool stop_;
    std::atomic_flag beginworking_;   //init: false
    std::atomic_flag endworking_;     //init: true
  };

  /* Why not a singleton:
      http://jalf.dk/blog/2010/03/singletons-solving-problems-you-didnt-know-you-never-had-since-1995/ */
  class CudaService: public ThreadPool {
  public:
    //!< @brief Checks CUDA and registers callbacks
    CudaService(const edm::ParameterSet&, edm::ActivityRegistry& actR);
    // deleted copy&move ctors&assignments
    CudaService(const CudaService&) =delete;
    CudaService& operator=(const CudaService&) =delete;
    CudaService(CudaService&&) =delete;
    CudaService& operator=(CudaService&&) =delete;
    static void fillDescriptions(edm::ConfigurationDescriptions& descr){
      descr.add("CudaService", edm::ParameterSetDescription());
    }

    // Configure execution policy before launch!
    //!< @brief !ONLY .cu FILES! Launch kernel function with args
    template<typename F, typename... Args>
    inline std::future<cudaError_t>
      cudaLaunchManaged(const cuda::ExecutionPolicy& execPol, F&& f, Args&&... args);

    template<typename F, typename... Args, typename LaunchType, typename
        std::enable_if< std::is_same<unsigned, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value ||
        std::is_same<cuda::ExecutionPolicy, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value, int >::type= 0>
    inline std::future<cudaError_t>
      cudaLaunch(LaunchType&& launchParam, F&& kernelWrap, Args&&... args);
    // virtual ~CudaService(){
    //   std::cout << "[CudaService]: ---| Destroying service |---\n";
    //   stopWorkers();
    // }
    
    bool GPUpresent() const { return cudaDevCount_ > 0; }
  private:
    int maxKernelAttempts_= 10;
    std::atomic<size_t> gpuFreeMem_;
    std::atomic<size_t> gpuTotalMem_;
    // {F,T}: not working, {T,T}: transition, {T,F}: working
    std::atomic_int cudaDevCount_;
  };

  template<typename F, typename... Args>
  inline std::future<cudaError_t>
    CudaService::cudaLaunchManaged(const cuda::ExecutionPolicy& execPol, F&& f, Args&&... args)
  {
    if (!cudaDevCount_){
      std::cout<<"[CudaService]: GPU not available\n";
      return schedule([]()->cudaError_t {
        return cudaErrorNoDevice;
      });
    }
    
    using packaged_task_t = std::packaged_task<cudaError_t()>;
    std::shared_ptr<packaged_task_t> task(new packaged_task_t([&] ()-> cudaError_t
    {
      int attempt= 0;
      cudaError_t status;
      // If device is not available, retry kernel up to maxKernelAttempts_ times
      do{
        #ifdef __NVCC__
          f<<<execPol.getGridSize(), execPol.getBlockSize(),
              execPol.getSharedMemBytes()>>>(utils::passKernelArg<Args>(args)...);
        #endif
        attempt++;
        status= cudaStreamSynchronize(cudaStreamPerThread);
        if (status!= cudaSuccess) std::this_thread::sleep_for(
                                              std::chrono::microseconds(50));
      }while(status == cudaErrorDevicesUnavailable && attempt < maxKernelAttempts_);
      return status;
    }));
    std::future<cudaError_t> resultFut= task->get_future();
    tasks_.emplace([task](){ (*task)(); });
    return resultFut;
  }

  template<typename F, typename... Args, typename LaunchType, typename
        std::enable_if< std::is_same<unsigned, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value ||
        std::is_same<cuda::ExecutionPolicy, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value, int >::type>
  inline std::future<cudaError_t> CudaService::cudaLaunch(LaunchType&& launchParam, F&& kernelWrap, Args&&... args){
    if (!cudaDevCount_){
      std::cout<<"[CudaService]: GPU not available\n";
      return schedule([]()->cudaError_t {
        return cudaErrorNoDevice;
      });
    }
    
    using packaged_task_t = std::packaged_task<cudaError_t()>;
    std::shared_ptr<packaged_task_t> task(new packaged_task_t([&] ()-> cudaError_t
    {
      int attempt= 0;
      cudaError_t status;
      // If device is not available, retry kernel up to maxKernelAttempts_ times
      do{
        kernelWrap(launchParam, utils::passKernelArg<Args>(args)...);
        attempt++;
        status= cudaStreamSynchronize(cudaStreamPerThread);
        if (status!= cudaSuccess) std::this_thread::sleep_for(
                                              std::chrono::microseconds(50));
      }while(status == cudaErrorDevicesUnavailable && attempt < maxKernelAttempts_);
      return status;
    }));
    std::future<cudaError_t> resultFut= task->get_future();
    tasks_.emplace([task](){ (*task)(); });
    return resultFut;
  }

  // The other non-template methods are defined in the .cu file
}} // namespace edm::service


#endif // Cuda_Service_H
