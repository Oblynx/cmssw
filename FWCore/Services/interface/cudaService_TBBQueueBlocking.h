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
#include "utils/template_utils.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
//Convenience include, since everybody including this also needs "Service.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

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

    //!< @brief For testing
    /*DEBUG*/void setWorkerN(const int& n) { threadNum_= n; }
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
    // {F,T}: not working, {T,T}: transition, {T,F}: working
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

    //!< @brief Launch kernel function with args
    template<typename F, typename... Args, typename LaunchType, typename
        std::enable_if< std::is_same<unsigned, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value ||
        std::is_same<cuda::ExecutionPolicy, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value, int >::type= 0>
    inline std::future<cudaError_t>
      cudaLaunch(LaunchType&& launchParam, F&& kernel, Args&&... args);

    template<typename F, typename... Args, typename LaunchType, typename
        std::enable_if< std::is_same<unsigned, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value ||
        std::is_same<cuda::ExecutionPolicy, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value, int >::type= 0>
    inline std::future<cudaError_t>
      cudaLaunchWrapper(LaunchType&& launchParam, F&& kernelWrapper, Args&&... args);
    
    bool GPUpresent() const { return cudaDevCount_ > 0; }
  private:
    int maxKernelAttempts_= 10;
    std::atomic<size_t> gpuFreeMem_;
    std::atomic<size_t> gpuTotalMem_;
    std::atomic_int cudaDevCount_;
  };

  //If passed unsigned, auto create ExecutionPolicy
  template<typename LaunchType, typename std::enable_if<
      std::is_same<unsigned, typename std::remove_cv<typename std::remove_reference<
      LaunchType>::type>::type>::value, int >::type= 0>
  inline cuda::ExecutionPolicy policyFromLaunchparam(LaunchType&& size, const void* kernel){
    // std::cout<<"[policyFromLaunchparam]: Autocreate ExecPol, size="<<size<<"\n";
    // std::cout << "[policyFromLaunchparam]: Auto ExecPol config:\n\t"
    //         <<"Grid/Block/Shared= "<<execPol.getGridSize().x<<'/'<<execPol.getBlockSize().x<<'/'<<execPol.getSharedMemBytes()<<'\n';
    return cuda::AutoConfig()(size, kernel);
  }
  //If passed an ExecutionPolicy, move it.
  template<typename LaunchType, typename std::enable_if<
      std::is_same<cuda::ExecutionPolicy, typename std::remove_cv<typename
      std::remove_reference<LaunchType>::type>::type>::value, int >::type= 0>
  inline LaunchType&& policyFromLaunchparam(LaunchType&& launchParam, const void*){
    std::cout<<"[policyFromLaunchparam]: Move ExecPol\n";
    return static_cast<LaunchType&&>(launchParam);
  }
  
  template<typename F, typename... Args, typename LaunchType, typename
        std::enable_if< std::is_same<unsigned, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value ||
        std::is_same<cuda::ExecutionPolicy, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value, int >::type>
  inline std::future<cudaError_t> CudaService::cudaLaunch(LaunchType&& launchParam, F&& kernelWrap, Args&&... args){
    if (!cudaDevCount_){
      std::cout<<"[CudaService]: GPU not available. Falling back to CPU.\n";
      return schedule([&] ()-> cudaError_t {
        kernelWrap(false, launchParam, utils::passKernelArg<Args>(args)...);
        return cudaErrorNoDevice;
      });
    }
    
    using packaged_task_t = std::packaged_task<cudaError_t()>;
    std::shared_ptr<packaged_task_t> task(new packaged_task_t([&] ()-> cudaError_t{
      int attempt= 0;
      cudaError_t status;
      // If device is not available, retry kernel up to maxKernelAttempts_ times
      do{
        // std::cout<<"[CudaService>Task]: Attempting kernel launch...\n";
        kernelWrap(true, launchParam, utils::passKernelArg<Args>(args)...);
        attempt++;
        status= cudaStreamSynchronize(cudaStreamPerThread);
        if (status!= cudaSuccess) std::this_thread::sleep_for(
                                              std::chrono::microseconds(50));
      }while(status == cudaErrorDevicesUnavailable && attempt < maxKernelAttempts_);
      utils::operateOnParamPacks(utils::releaseKernelArg<Args>(args)...);
      return status;
    }));
    std::future<cudaError_t> resultFut= task->get_future();
    tasks_.emplace([task](){ (*task)(); });
    return resultFut;
  }

#include <cstdio>
__global__ void paramKernel(const int param, const int param2);
__global__ void inOutKernel(const int in, int* out);

  template<typename F, typename... Args, typename LaunchType, typename
        std::enable_if< std::is_same<unsigned, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value ||
        std::is_same<cuda::ExecutionPolicy, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value, int >::type>
  inline std::future<cudaError_t> 
    CudaService::cudaLaunchWrapper(LaunchType&& launchParam, F&& kernelWrapper, Args&&... args)
  {
    std::cout << "[CudaService]: cudaLaunchWrapper\n";
    if (!cudaDevCount_){
      std::cout<<"[CudaService]: GPU not available. Falling back to CPU.\n";
      return schedule([&] ()-> cudaError_t {
        if (kernelWrapper.fallbackPresent)
          kernelWrapper.fallback(utils::passKernelArg<Args>(args)...);
        return cudaErrorNoDevice;
      });
    }
    unsigned argNumber= sizeof...(args);
    using packaged_task_t = std::packaged_task<cudaError_t()>;
    std::shared_ptr<packaged_task_t> task(new packaged_task_t([&,argNumber] ()-> cudaError_t{
      // void** argList=(void**)malloc(argNumber*sizeof(void*));
      // void* argList[argNumber];
      // Unwrap each arg's address into a void** list
      // utils::unwrapToList<0>(argList, utils::passKernelArg<Args>(args)...);
      // Create ExecutionPolicy automatically iff LaunchType==unsigned, otherwise move it
      auto execPol= policyFromLaunchparam(std::move(launchParam), kernelWrapper.kernel);
      int attempt= 0;
      cudaError_t status;
      // If device is not available, retry kernel up to maxKernelAttempts_ times
      do{
        std::cout << "[CudaService>Task]: Launching kernel with config:\n\t"
            <<"Grid/Block/Shared= "<<execPol.getGridSize().x<<'/'<<execPol.getBlockSize().x<<'/'
            <<execPol.getSharedMemBytes()<<" CudaStream: "<<cudaStreamPerThread<<'\n';
        // status= cudaLaunchKernel(kernelWrapper.kernel, execPol.getGridSize(), execPol.getBlockSize(),
        //                  const_cast<void**>(argList), execPol.getSharedMemBytes(), cudaStreamPerThread);
        void* argList[2];
        int arg1=5, anInteger= 6;
        int* darg2; cudaMalloc(&darg2, sizeof(int)); cudaMemcpy(darg2, &anInteger, sizeof(int), cudaMemcpyHostToDevice);
        argList[0]= &arg1, argList[1]= &darg2;
        status= cudaLaunchKernel((void*)inOutKernel, 1,1, argList,0,cudaStreamPerThread);
        std::cout<<"[CudaService>Task]: Cuda status after launch: "<<status<<"\n";
        attempt++;
        status= cudaStreamSynchronize(cudaStreamPerThread);
        std::cout<<"[CudaService>Task]: Cuda status after launch: "<<status<<"\n";
        cudaMemcpy(&anInteger, darg2, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout<<"[CudaService>Task]: arg2="<<anInteger<<'\n';
        if (status!= cudaSuccess) std::this_thread::sleep_for(
                                              std::chrono::microseconds(30));
      }while(status == cudaErrorDevicesUnavailable && attempt < maxKernelAttempts_);
      utils::operateOnParamPacks(utils::releaseKernelArg<Args>(args)...);
      return status;
    }));
    std::future<cudaError_t> resultFut= task->get_future();
    tasks_.emplace([task](){ (*task)(); });
    return resultFut;
  }

  // The other non-template methods are defined in the .cu file
}} // namespace edm::service


#endif // Cuda_Service_H
