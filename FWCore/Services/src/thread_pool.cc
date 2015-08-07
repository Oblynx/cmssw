#include "FWCore/Services/interface/thread_pool.h"

#include <iostream>

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <memory>
#include <functional>
#include <stdexcept>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace edm::service;

// the constructor just launches some amount of workers
ThreadPoolService::ThreadPoolService(const edm::ParameterSet&, edm::ActivityRegistry&): stop_(false)
{
  std::cout<<"\nConstructing a ThreadPoolService!\n";
  size_t threads_n = std::thread::hardware_concurrency();
  if(!threads_n)
    throw std::invalid_argument("more than zero threads expected");

  this->workers_.reserve(threads_n);
  for(; threads_n; --threads_n)
    this->workers_.emplace_back([this] (){
      while(true)
      {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex_);
          this->condition_.wait(lock,
                               [this]{ return this->stop_ || !this->tasks_.empty(); });
          if(this->stop_ && this->tasks_.empty())
            return;
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }

        task();
     }
    });
}

void ThreadPoolService::fillDescriptions(edm::ConfigurationDescriptions& descr){
    descr.add("ThreadPoolService", edm::ParameterSetDescription());
}

// add new work item to the pool
template<class F, class... Args>
std::future<typename std::result_of<F(Args...)>::type> 
      ThreadPoolService::enqueue(F&& f, Args&&... args)
{
  using packaged_task_t = std::packaged_task<typename std::result_of<F(Args...)>::type ()>;

  std::shared_ptr<packaged_task_t> task(new packaged_task_t(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
  ));
  auto resultFut = task->get_future();
  {
    std::unique_lock<std::mutex> lock(this->queue_mutex_);
    this->tasks_.emplace([task](){ (*task)(); });
  }
  this->condition_.notify_one();
  return resultFut;
}

// the destructor joins all threads
ThreadPoolService::~ThreadPoolService()
{
  this->stop_ = true;
  this->condition_.notify_all();
  for(std::thread& worker : this->workers_)
    worker.join();
}
