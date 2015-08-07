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
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <memory>
#include <functional>
#include <stdexcept>


namespace edm{
class ParameterSet;
class ActivityRegistry;
class ConfigurationDescriptions;
namespace service{

// std::thread pool for resources recycling
class ThreadPoolService {
public:
  ThreadPoolService(){
    std::cout<<"\nDEFAULT constructing a ThreadPoolService!\n";
  }
  // the constructor just launches some amount of workers
  ThreadPoolService(const edm::ParameterSet&, edm::ActivityRegistry&);
  // deleted copy&move ctors&assignments
	ThreadPoolService(const ThreadPoolService&) = delete;
	ThreadPoolService& operator=(const ThreadPoolService&) = delete;
	ThreadPoolService(ThreadPoolService&&) = delete;
	ThreadPoolService& operator=(ThreadPoolService&&) = delete;
  static void fillDescriptions(edm::ConfigurationDescriptions& descr);

  // add new work item to the pool
  template<class F, class... Args>
	std::future<typename std::result_of<F(Args...)>::type> enqueue(F&& f, Args&&... args);

  // the destructor joins all threads
	virtual ~ThreadPoolService();

private:
  // need to keep track of threads so we can join them
	std::vector< std::thread > workers_;
  // the task queue
	std::queue< std::function<void()> > tasks_;

  // synchronization
	std::mutex queue_mutex_;
	std::condition_variable condition_;
  // workers_ finalization flag
	std::atomic_bool stop_;
};

} // namespace service
} // namespace edm

#endif // Thread_Pool_Service_H
