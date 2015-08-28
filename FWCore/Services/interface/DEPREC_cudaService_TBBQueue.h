/**
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
Editor: Konstantinos Samaras-Tsakiris, kisamara@auth.gr
*/

#ifndef Thread_Pool_Service_H
#define Thread_Pool_Service_H

#include <iostream>

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>
#include <functional>
#include <stdexcept>

#include <tbb/concurrent_queue.h>

#ifdef __NVCC__
#include "cuda_utils/cuda_launch_configuration.h"
#include "cuda_utils/cuda_execution_policy.h"
#endif

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm{
namespace service{

template<int ...> struct Seq {};
template<int N, int ...S> struct GenSeq: GenSeq<N-1, N-1, S...> { };
template<int ...S> struct GenSeq<0, S...> {
  typedef Seq<S...> type;
};
template<typename... Args> struct KernelArgs{
  KernelArgs(Args&&... args): args_(std::forward<Args>(args)...) {}
  std::tuple<Args...> args_;
};
template<typename... Args> struct ManagedArgs: KernelArgs<Args...>{
  ManagedArgs(Args&&... args): KernelArgs<Args...>(std::forward<Args>(args)...)
  {}
};
template<typename... Args> struct NonManagedArgs: KernelArgs<Args...>{
  NonManagedArgs(Args&&... args): KernelArgs<Args...>(std::forward<Args>(args)...)
  {}
};

// std::thread pool for resources recycling
class CudaService {
public:
  // the constructor just launches some amount of workers
  CudaService(const edm::ParameterSet&, edm::ActivityRegistry&);
  // deleted copy&move ctors&assignments
  CudaService(const CudaService&) = delete;
  CudaService& operator=(const CudaService&) = delete;
  CudaService(CudaService&&) = delete;
  CudaService& operator=(CudaService&&) = delete;
  static void fillDescriptions(edm::ConfigurationDescriptions& descr){
    descr.add("CudaService", edm::ParameterSetDescription());
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
    this->condition_.notify_one();
    return resultFut;
  }

#ifdef __NVCC__
  // Launch kernel function with args
  // Configure execution policy before launch!
  template<typename F, typename... Args>
  inline std::future<typename std::result_of<F(Args...)>::type>
    cudaLaunchManaged(const cuda::ExecutionPolicy& execPol, F&& f, Args&&... args)
  {
    return getFuture([&](){
      f<<<execPol.getGridSize(), execPol.getBlockSize(),
          execPol.getSharedMemBytes()>>>(args...);
      //std::cout<<"[In task]: Launched\n";
      cudaStreamSynchronize(cudaStreamPerThread);
      //std::cout<<"[In task]: Synced\n";
    });
  }
  template<typename F>
  static cuda::ExecutionPolicy configureLaunch(int totalThreads, F&& f){
    cuda::ExecutionPolicy execPol;
    checkCuda(cuda::autoConfig(execPol, std::forward<F>(f), totalThreads));
    return execPol;
  }
#else
  // Declared but undefined
  template<typename F, typename... Args>
  inline std::future<typename std::result_of<F(Args...)>::type>
    cudaLaunchManaged(void*, F&& f, Args&&... args);
  template<typename F>
  static int configureLaunch(int totalThreads, F&& f);
#endif 
  // Overload: differentiate between managed-nonmanaged args
  /*template<typename F, typename... NMArgs, template<typename...> class NM,
            typename... MArgs, template<typename...> class M>
  typename std::enable_if<std::is_same<NM<NMArgs...>,NonManagedArgs<NMArgs...>>::value
                          && std::is_same<M<MArgs...>,ManagedArgs<MArgs...>>::value,
                          std::future<typename std::result_of<F(void)>>>::type
      cudaLaunchManaged(F&& f, NM<NMArgs...>&& nonManaged, M<MArgs...>&& managed)
  {
    std::cout<<"Separate managed-unmanaged args!";
    //return cudaLaunchManaged(std::forward<F>(f), nonManaged.forward(), managed.forward());
    return unpackArgsTuple(typename GenSeq<sizeof...(NMArgs)+
                          sizeof...(MArgs)>::type(), std::forward<F>(f), 
                          merge(nonManaged,managed));
  }*/

  // the destructor joins all threads
  virtual ~CudaService(){
    std::cout << "---| Destroying service |---\n";
    stop_= true;
    this->condition_.notify_all();
    for(std::thread& worker : workers_)
      worker.join();
  }
private:
  /*template<int... S, typename F, typename... NMArgs, template<typename...> class NM,
                                 typename... MArgs, template<typename...> class M>
  void unpackArgsTuple(Seq<S...>, F&& f, NM<NMArgs...>&& nonMan, M<MArgs...>&& man){
  }*/
  // need to keep track of threads so we can join them
  std::vector< std::thread > workers_;
  // the task concurrent queue
  tbb::concurrent_queue< std::function<void()> > tasks_;

  std::mutex signalMutex_;
  std::condition_variable condition_;
  // workers_ finalization flag
  std::atomic_bool stop_;
  bool cuda_;
};

// the constructor just launches some amount of workers
CudaService::CudaService(const edm::ParameterSet&, edm::ActivityRegistry&):
  stop_(false)
{
  std::cout<<"Constructing CudaService\n";
  // TODO(ksamaras): Check num GPUs, threads_n= 4*GPUs
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
  size_t threads_n = std::thread::hardware_concurrency();
  if(!threads_n)
    throw std::invalid_argument("more than zero threads expected");

  workers_.reserve(threads_n);
  for(; threads_n; --threads_n)
    workers_.emplace_back([this] (){
      while(!stop_)
      {
        std::function<void()> task;
        // Wait to receive a task submission or stop signal
        {
          std::unique_lock<std::mutex> lock(signalMutex_);
          condition_.wait(lock, [this]{ return stop_ || !tasks_.empty(); });
        }
        if (tasks_.try_pop(task))
          task();
      }
    });
}

} // namespace service
} // namespace edm

#endif // Thread_Pool_Service_H
