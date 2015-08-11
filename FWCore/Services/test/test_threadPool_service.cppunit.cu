// Service to test
#include "FWCore/Services/interface/thread_pool.h"

// std
#include <iostream>
#include <vector>
#include <future>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <thread>
#include <chrono>

// CMSSW
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/PluginManager.h"

// cppunit-specific
#include "cppunit/extensions/HelperMacros.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"

class TestThreadPoolService: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestThreadPoolService);
  CPPUNIT_TEST(basicUseTest);
  CPPUNIT_TEST(passServiceArgTest);
  CPPUNIT_TEST(CUDATest);
  CPPUNIT_TEST(CUDAAutolaunchTest);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp();
  void tearDown() {std::cout<<"\n";}
  void basicUseTest();
  void CUDATest();
  void passServiceArgTest();
  void CUDAAutolaunchTest();
private:
  void print_id(int id);
  void go();
  void cudaTask(int n, int i, const float* din, int times);
  //--$--//
  std::mutex mtx;
  std::condition_variable cv;
  bool ready= false;
  long sum= 0;
  const int BLOCK_SIZE= 32;
  edm::ServiceToken serviceToken;
  const std::string serviceConfig= 
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('testThreadPoolService')\n"
      "process.ThreadPoolService = cms.Service('ThreadPoolService')\n";
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestThreadPoolService);

__global__ void longKernel(int n, int times, const float* in, float* out)
{
  int x= blockIdx.x*blockDim.x + threadIdx.x;
  if (x < n){
    out[x]= 0;
    for(int i=0; i<times; i++){
      out[x]+= in[x];
    }
  }
}
void TestThreadPoolService::setUp(){
  static std::atomic_flag notFirstTime= ATOMIC_FLAG_INIT;
  if (!notFirstTime.test_and_set()){
    // Init modelled after "FWCore/Catalog/test/FileLocator_t.cpp"
    // Make the services.
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }
  serviceToken= edm::ServiceRegistry::createServicesFromConfig(serviceConfig);
}
void TestThreadPoolService::print_id(int id) {
  std::unique_lock<std::mutex> lck(mtx);
  while (!ready) cv.wait(lck);
  // ...
  std::cout << id << "\t";
  sum+= id;
}
void TestThreadPoolService::go() {
  std::unique_lock<std::mutex> lck(mtx);
  ready = true;
  cv.notify_all();
}
void TestThreadPoolService::cudaTask(int n, int i, const float* din, int times){
  float *dout;
  cudaMalloc((void **) &dout, n*sizeof(float));
  dim3 grid((n-1)/BLOCK_SIZE/BLOCK_SIZE+1);
  dim3 block(BLOCK_SIZE*BLOCK_SIZE);
  longKernel<<<grid,block>>>(n, times, din, dout);
  cudaStreamSynchronize(cudaStreamPerThread);
  float out;
  cudaMemcpy(&out, dout+i, 1*sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "GPU::" << out << "\t";
  cudaFree(dout);
}

void TestThreadPoolService::basicUseTest()
{
  //make the services available
  edm::ServiceRegistry::Operate operate(serviceToken);
  edm::Service<edm::service::ThreadPoolService> pool;
  std::cout<<"\nStarting basic test...\n";
  pool->enqueue([]() {std::cout<<"Empty task\n";}).get();

  std::cout<<"[ThreadPoolService::basicUseTest] Service initialized\n";
  std::vector<std::future<void>> futures;
  const int N= 30;

  // spawn N threads:
  for (int i=0; i<N; ++i)
    futures.emplace_back(pool->enqueue(&TestThreadPoolService::print_id, this,i+1));
  go();

  for (auto& future: futures) future.get();
  std::cout << "\n[ThreadPoolService::basicUseTest] DONE, sum= "<<sum<<"\n";
	for(int i=0; i<N; i++)
		sum-= i+1;
  CPPUNIT_ASSERT_EQUAL(sum, 0l);
}
//!< @brief Test behaviour if the task itself enqueues another task in same pool
void TestThreadPoolService::passServiceArgTest()
{
  //make the services available
  edm::ServiceRegistry::Operate operate(serviceToken);
  edm::Service<edm::service::ThreadPoolService> pool;
  std::cout<<"\nStarting passServiceArg test...\n";
  pool->enqueue([&]() {
    std::cout<<"Recursive enqueue #1\n";
    edm::ServiceRegistry::Operate operate(serviceToken);
    pool->enqueue([]() {std::cout<<"Pool service captured\n";}).get();
  }).get();
  pool->enqueue([this](edm::Service<edm::service::ThreadPoolService> poolArg){
    std::cout<<"Recursive enqueue #2\n";
    edm::ServiceRegistry::Operate operate(serviceToken);
    poolArg->enqueue([]() {std::cout<<"Pool service passed as arg\n";}).get();
  }, pool).get();
}
//!< @brief Test scheduling many threads that launch CUDA kernels
void TestThreadPoolService::CUDATest()
{
  //make the services available
  edm::ServiceRegistry::Operate operate(serviceToken);
  edm::Service<edm::service::ThreadPoolService> pool;
  std::cout<<"\nStarting CUDA test...\n";
  std::vector<std::future<void>> futures;
  const int N= 30;

  float *in, *din;
  int n= 2000;
  in= new float[n];
  for(int i=0; i<n; i++) in[i]= 10*cos(3.141592/100*i);
  // Make GPU input data available for all threads
  cudaMalloc((void **) &din, n*sizeof(float));
  cudaMemcpy(din, in, n*sizeof(float), cudaMemcpyHostToDevice);

  // spawn N threads
  for (int i=0; i<N; ++i){
    futures.emplace_back(pool->enqueue(&TestThreadPoolService::cudaTask, this,
                         n, i, din, 2));
  }
  for (auto& future: futures) future.get();
}
//!< @brief 
void TestThreadPoolService::CUDAAutolaunchTest()
{
  //make the services available
  edm::ServiceRegistry::Operate operate(serviceToken);
  edm::Service<edm::service::ThreadPoolService> pool;
  std::cout<<"\nStarting CUDA autolaunch test...\n";
  float *in, *out;
  const int n= 20, times= 1;
  cudaMallocManaged(&in, n*sizeof(float), cudaMemAttachHost);
  cudaMallocManaged(&out, n*sizeof(float));
  for(int i=0; i<n; i++) in[i]= 10*cos(3.141592/100*i);

  pool->launchKernelManaged(longKernel, n,times,in,out);

  for(int i=0; i<n; i++) CPPUNIT_ASSERT_EQUAL(out[i],times*in[i]);
  cudaFree(in);
  cudaFree(out);
}
