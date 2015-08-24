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
#include <cuda_runtime_api.h>

#include <tbb/concurrent_queue.h>

#include "utils/cuda_launch_configuration.h"
#include "utils/cuda_execution_policy.h"
#include "utils/cuda_pointer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

/**$$$~~~~~ CudaService class declaration ~~~~~$$$**/
namespace edm{namespace service{

  /* Why not a singleton:
  http://jalf.dk/blog/2010/03/singletons-solving-problems-you-didnt-know-you-never-had-since-1995/
  */
  class CudaService {
  public:
    //!< @brief Checks CUDA and registers callbacks
    CudaService(const edm::ParameterSet&, edm::ActivityRegistry& actR);
    // deleted copy&move ctors&assignments
    CudaService(const CudaService&) = delete;
    CudaService& operator=(const CudaService&) = delete;
    CudaService(CudaService&&) = delete;
    CudaService& operator=(CudaService&&) = delete;
    static void fillDescriptions(edm::ConfigurationDescriptions& descr){
      descr.add("CudaService", edm::ParameterSetDescription());
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
      cudaLaunchManaged(const cudaConfig::ExecutionPolicy& execPol, F&& f, Args&&... args);

    //!< @brief Clears tasks queue
    void clearTasks(){ tasks_.clear(); }
    virtual ~CudaService(){
      std::cout << "---| Destroying service |---\n";
      stopWorkers();
    }

    template<typename F>
    static cudaConfig::ExecutionPolicy configureLaunch(int totalThreads, F&& f){
      cudaConfig::ExecutionPolicy execPol;
      checkCuda(cudaConfig::configure(execPol, std::forward<F>(f), totalThreads));
      return execPol;
    }
    
    bool cudaState() const { return cudaDevCount_ > 0; }
    //!< @brief Constructs workers and sets them spinning. Gets initially available GPU memory.
    void startWorkers();
    //!< @brief Joins all worker threads
    void stopWorkers();
  private:
    size_t threadNum_= 0;
    int maxKernelAttempts_= 10;
    std::atomic<size_t> gpuFreeMem_;
    std::atomic<size_t> gpuTotalMem_;
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers_;
    // the task concurrent queue
    tbb::concurrent_bounded_queue< std::function<void()> > tasks_;

    // workers_ finalization flag
    std::atomic_bool stop_;
    std::atomic_flag beginworking_;   //init: false
    std::atomic_flag endworking_;     //init: true
    // {F,T}: not working, {T,T}: transition, {T,F}: working
    std::atomic_int cudaDevCount_;
  };
}}  //namespace edm::service


/**$$$~~~~~ CudaPointer method definitions ~~~~~$$$**/
  //Constructor
  template<typename T>
  cudaPointer<T>::cudaPointer(edm::Service<edm::service::CudaService>& service, unsigned elementN, Attachment flag):
      attachment(flag), sizeOnDevice(elementN*sizeof(T)), freeFlag(false),
      elementN(elementN), service_(service)
  {
    static_assert(!std::is_const<T>::value && !std::is_reference<T>::value,
                  "\nCannot allocate cuda managed memory for const-qualified "
                  "or reference types.\nConsult the CUDA C Programming Guide "
                  "section E.2.2.2. __managed__ Qualifier.\n");
    allocate();
  }
  //Move assignment
  template<typename T>
  cudaPointer<T>& cudaPointer<T>::operator=(cudaPointer&& other) noexcept{
    p= other.p; other.p= NULL;
    sizeOnDevice= other.sizeOnDevice;
    attachment= other.attachment;
    status= other.status;
    freeFlag= other.freeFlag;
    service_= other.service_;
    elementN= other.elementN;
    return *this;
  }
  //Constructor vector copy
  template<typename T>
  cudaPointer<T>::cudaPointer(edm::Service<edm::service::CudaService>& service, const std::vector<T>& vec, Attachment flag):
      attachment(flag), freeFlag(false), elementN(vec.size()), service_(service)
  {
    static_assert(!std::is_const<T>::value && !std::is_reference<T>::value,
                  "\nCannot allocate cuda managed memory for const-qualified "
                  "or reference types.\nConsult the CUDA C Programming Guide "
                  "section E.2.2.2. __managed__ Qualifier.\n");
    sizeOnDevice= elementN*sizeof(T);
    allocate();
    if (status==cudaSuccess)
      for(unsigned i=0; i<elementN; i++)
        p[i]= vec[i];
  }
  //If cudaStreamAttachMemAsync doesn't throw if no GPU, simplify design
  template<typename T>
  void cudaPointer<T>::attachStream(cudaStream_t stream){
    attachment= single;
    if (service_->cudaState()) cudaStreamAttachMemAsync(stream, p, 0, attachment);
  }
  template<typename T>
  std::vector<T> cudaPointer<T>::getVec(bool release){
    std::vector<T> vec;
    vec.reserve(elementN);
    for(unsigned i=0; i<elementN; i++)
      vec.push_back(std::move(p[i]));
    if(release){
      freeFlag= true;
      deallocate();
    }
    return vec;
  }
  template<typename T>
  void cudaPointer<T>::allocate(){
    if (service_->cudaState())
      status= cudaMallocManaged((void**)&p, sizeOnDevice, attachment);
    else if (elementN==1)
      p= new T;
    else
      p= new T[elementN];
  }
  template<typename T>
  void cudaPointer<T>::deallocate(){
    if (service_->cudaState()) status= cudaFree(p);
    else if (elementN==1)
      delete p;
    else
      delete[] p;
  }

/**$$$~~~~~ Kernel argument wrapper templates ~~~~~$$$**/
namespace edm{namespace service{namespace utils{

  //!< @brief Manipulate cudaPtr and pass the included pointer (expect lvalue ref)
  template<typename T, typename std::enable_if< std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
                  ::type >::value, int >::type= 0>
  inline auto passKernelArg(typename std::remove_reference<T>::type& cudaPtr) noexcept
    -> decltype(cudaPtr.p)
  {
    //std::cout<<"[passKernelArg]: Managed arg!\n";
    cudaPtr.attachStream();
    return cudaPtr.p;
  }

  //!< @brief Perfectly forward non-cudaPointer args (based on std::forward)
  template<typename T, typename std::enable_if< !std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
                  ::type >::value, int >::type= 0>
  inline T&& passKernelArg(typename std::remove_reference<T>::type& valueArg) noexcept{
    //std::cout<<"[passKernelArg]: value arg\n";
    return static_cast<T&&>(valueArg);
  }
  template<typename T, typename std::enable_if< !std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
                  ::type >::value, int >::type= 0>
  inline T&& passKernelArg(typename std::remove_reference<T>::type&& valueArg) noexcept{
    static_assert(!std::is_lvalue_reference<T>::value,
                  "Can not forward an rvalue as an lvalue.");
    //std::cout<<"[passKernelArg]: value arg (rvalue ref)\n";
    return static_cast<T&&>(valueArg);
  }
}}} //namespace edm::service::utils

/**$$$~~~~~ CudaService method definitions ~~~~~$$$**/
namespace edm{namespace service{

  CudaService::CudaService(const edm::ParameterSet& pSet, edm::ActivityRegistry& actR):
    stop_(false), cudaDevCount_(0)
  {
    beginworking_.clear(); endworking_.test_and_set();
    std::cout<<"Constructing CudaService\n";
    /**Checking presence of GPU**/
    int deviceCount = 0;
    cudaError_t error_id= cudaSuccess;
    #ifdef __NVCC__
      error_id= cudaGetDeviceCount(&deviceCount);
    #endif
    if (error_id == cudaErrorNoDevice || deviceCount == 0){
      std::cout<<"No device available!\n";
      cudaDevCount_= 0;
    } else cudaDevCount_= deviceCount;
    //size_t threadNum = 4*deviceCount;
              /*DEBUG*/ if (deviceCount==0) return;
    if (pSet.exists("thread_num"))
      threadNum_= pSet.getParameter<int>("thread_num");
    actR.watchPostBeginJob(this, &CudaService::startWorkers);
    actR.watchPostEndJob(this, &CudaService::stopWorkers);
  }
  template<typename F, typename... Args>
  inline std::future<cudaError_t>
    CudaService::cudaLaunchManaged(const cudaConfig::ExecutionPolicy& execPol, F&& f, Args&&... args)
  {
    if (!cudaDevCount_){
      std::cout<<"GPU not available\n";
      return getFuture([]()->cudaError_t {
        return cudaErrorNoDevice;
      });
      //throw new std::runtime_error("No gpu available\n");
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

  void CudaService::startWorkers()
  {
    // continue only if !beginworking
    if (beginworking_.test_and_set()) return;

    size_t gpuFreeMem= 0, gpuTotalMem= 0;
    #ifdef __NVCC__
      if (cudaDevCount_) cudaMemGetInfo(&gpuFreeMem,&gpuTotalMem);
    #endif
    gpuFreeMem_= gpuFreeMem, gpuTotalMem_= gpuTotalMem;
    //Default thread number
    if(!threadNum_){
      if(cudaDevCount_)
        threadNum_= 4*cudaDevCount_;
      else
        threadNum_= std::thread::hardware_concurrency();
    }
    if(!threadNum_) throw std::invalid_argument("more than zero threads expected");
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
            std::cout << "Unhandled exception!\n";
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


#endif // Thread_Pool_Service_H
