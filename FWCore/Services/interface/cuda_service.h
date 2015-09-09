//! GPU-managing service and Thread Pool -- Should be included whenever the service is needed
/*  The ThreadPool is based on a Github project. Notice:
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

// Debug includes
#include <iostream>
//< Debug includes

//System
#include <vector>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include <functional>
#include <cuda_runtime.h>

//Other tools
#include <tbb/concurrent_queue.h>

//Local components
#include "utils/cuda_execution_policy.h"
#include "utils/cuda_pointer.h"
#include "utils/GPU_presence_static.h"
#include "utils/template_utils.h"

//CMSSW
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
//Convenience include, since everybody including this also needs "Service.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace edm{namespace service{
  /*! Maintains a vector of waiting threads, which consume tasks from a task queue.

    - Although a tbb::concurrent_bounded_queue is used by default, an std::queue could
    also be used if an external synchronization mechanism is provided (less efficient).
    - Also a tbb::concurrent_queue can be used, if an external mechanism for waiting
    before queue pop is provided (mutex + condition variable)
  */ 
  class ThreadPool{
  public:
    //! Default constructor.
    ThreadPool(): stop_(false) {
      beginworking_.clear(); endworking_.test_and_set();
    }
    //!@{
    //! Copy and move constructors and assignments explicitly deleted.
    ThreadPool(const ThreadPool&) =delete;
    ThreadPool& operator=(const ThreadPool&) =delete;
    ThreadPool(ThreadPool&&) =delete;
    ThreadPool& operator=(ThreadPool&&) =delete;
    //!@}
    
    /*! Schedule a (cpu) task and get its future handle
      @param f Any callable object (function name, lambda, `std::function`, ...)
      @param args Variadic template parameter pack that holds all the arguments
      that should be forwarded to the callable object
      @return An `std::future` handle to the task's result
    */
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

    //! Clears tasks queue
    void clearTasks(){ tasks_.clear(); }
    //! Constructs workers and sets them waiting. [release]->protected
    void startWorkers();
    //! Joins all worker threads. [release]->protected
    void stopWorkers();
    //! Destructor stops workers, if they are still running.
    virtual ~ThreadPool(){
      std::cout << "[ThreadPool]: ---| Destroying pool |---\n";
      stopWorkers();
    }

    //! Only for testing. [release]->remove
    void setWorkerN(const int& n) { threadNum_= n; }
  protected:
    //! Threads that consume tasks
    std::vector< std::thread > workers_;
    //! Concurrent queue that produces tasks
    tbb::concurrent_bounded_queue< std::function<void()> > tasks_;
    size_t threadNum_= 0;
  private:
    //! workers_ finalization flag
    std::atomic_bool stop_;
    //!@{
    /*! Start/end workers synchronization flags.
      {beginworking_, endworking}: (initially: {F,T})
      1. {F,T}: not working
      2. {T,T}: transition
      3. {T,F}: working
    */
    std::atomic_flag beginworking_;
    std::atomic_flag endworking_;
    //!@}
  };

  /*! Manages the GPUs and launches kernels on them.

    Only 1 instance of this class should ever exist and it should be implicitly
    created by cmsRun like any other service, never explicitly by the user. It is
    meant to be shared between threads and thus is thread-safe.
    - Model: submit task -> get future
    - Is callable from normal C++
    - Handles some GPU failures and forwards the rest to the user
    - Allows for alternative CPU fallbacks, if no GPU is available
    - [Why not a singleton?](http://jalf.dk/blog/2010/03/singletons-solving-problems-you-didnt-know-you-never-had-since-1995/)
    - 
    @sa ThreadPool
  */
  class CudaService: public ThreadPool {
  public:
    //! Checks number of GPUs and registers start/end worker callbacks for the thread pool
    CudaService(const edm::ParameterSet&, edm::ActivityRegistry& actR);
    //!@{
    //! deleted copy&move ctors&assignments
    CudaService(const CudaService&) =delete;
    CudaService& operator=(const CudaService&) =delete;
    CudaService(CudaService&&) =delete;
    CudaService& operator=(CudaService&&) =delete;
    //!@}
    static void fillDescriptions(edm::ConfigurationDescriptions& descr){
      descr.add("CudaService", edm::ParameterSetDescription());
    }

    /*! Launches a kernel on the GPU or its CPU fallback
      @param launchParam Either `cuda::ExecutionPolicy` for specifying an explicit
      launch configuration or an `unsigned` for automatic. Other types will not
      match the template and produce a compiler error.
      @param kernelWrap A kernel wrapper (with or without CPU fallback), written
      according to the documentation.
      @param args Variadic template parameter pack that holds all the arguments
      that should be forwarded to the kernel (or the CPU fallback) on execution.
      @return An `std::future` handle to the task's result that returns any error
      code produced by the CUDA Runtime API while attempting to launch the kernel
    */
    template<typename F, typename... Args, typename LaunchType, typename
        std::enable_if< std::is_same<unsigned, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value ||
        std::is_same<cuda::ExecutionPolicy, typename std::remove_cv<
        typename std::remove_reference<LaunchType>::type>::type>::value, int >::type= 0>
    inline std::future<cudaError_t>
      cudaLaunch(LaunchType&& launchParam, F&& kernelWrap, Args&&... args);
    //! Informs whether a GPU is present at runtime
    bool GPUpresent() const { return cudaDevCount_ > 0; }
  private:
    //! Max times to attempt kernel launch due to insufficient resources before forwarding failure
    int maxKernelAttempts_= 10;
    std::atomic<size_t> gpuFreeMem_;
    std::atomic<size_t> gpuTotalMem_;
    std::atomic_int cudaDevCount_;
  };

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
        kernelWrap(true, launchParam, utils::passKernelArg<Args>(args)...);
        attempt++;
        status= cudaStreamSynchronize(cudaStreamPerThread);
        utils::operateParamPack(utils::releaseKernelArg<Args>(args)...);
        if (status!= cudaSuccess) std::this_thread::sleep_for(
                                              std::chrono::microseconds(30));
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
