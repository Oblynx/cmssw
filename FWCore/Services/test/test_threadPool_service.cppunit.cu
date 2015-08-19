// Service to test
#include "FWCore/Services/interface/thread_pool_TBBQueueBlocking.h"

// std
#include <iostream>
#include <vector>
#include <future>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <algorithm>

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

using namespace std;
using namespace edm;

class TestThreadPoolService: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestThreadPoolService);
  CPPUNIT_TEST(basicUseTest);
  //CPPUNIT_TEST(passServiceArgTest);
  CPPUNIT_TEST(CUDATest);
  CPPUNIT_TEST(CUDAAutolaunchManagedTest);
  CPPUNIT_TEST(timeBenchmark);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp();
  void tearDown() {
    //(*poolPtr)->clearTasks();
    cout<<"\n";
  }
  void basicUseTest();
  //!< @brief Test behaviour if the task itself enqueues another task in same pool
  void passServiceArgTest();
  //!< @brief Test scheduling many threads that launch CUDA kernels
  void CUDATest();
  //!< @brief Test auto launch cuda kernel with its arguments in managed memory
  void CUDAAutolaunchManagedTest();
  void timeBenchmark();
private:
  void print_id(int id);
  void go();
  void cudaTask(int n, int i, const float* din, int times);
  //--$--//
  mutex mtx;
  condition_variable cv;
  bool ready= false;
  long sum= 0;
  const int BLOCK_SIZE= 32;

  ServiceToken serviceToken;
  unique_ptr<Service<service::ThreadPoolService>> poolPtr;
  unique_ptr<ServiceRegistry::Operate> operate;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestThreadPoolService);

__global__ void longKernel(const int n, const int times, const float* in, float* out)
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
  static atomic_flag notFirstTime= ATOMIC_FLAG_INIT;
  if (!notFirstTime.test_and_set()){
    // Init modelled after "FWCore/Catalog/test/FileLocator_t.cpp"
    // Make the services.
    edmplugin::PluginManager::configure(edmplugin::standard::config());
    //serviceToken= edm::ServiceRegistry::createServicesFromConfig(serviceConfig);
    ParameterSet pSet;
    pSet.addParameter("@service_type", string("ThreadPoolService"));
    vector<ParameterSet> vec;
    vec.push_back(pSet);
    operate= unique_ptr<ServiceRegistry::Operate>(
        new ServiceRegistry::Operate(edm::ServiceRegistry::createSet(vec)));
    poolPtr= unique_ptr<Service<service::ThreadPoolService>>(
        new Service<service::ThreadPoolService>);
    //(*poolPtr)->startWorkers();
    cout<<"[ThreadPoolServiceTest::init] Service initialized\n";
  }
}
void TestThreadPoolService::print_id(int id) {
  unique_lock<mutex> lck(mtx);
  while (!ready) cv.wait(lck);
  // ...
  cout << id << "\t";
  sum+= id;
}
void TestThreadPoolService::go() {
  unique_lock<mutex> lck(mtx);
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
  cout << "GPU::" << out << "\t";
  cudaFree(dout);
}

void TestThreadPoolService::basicUseTest()
{
  cout<<"\nStarting basic test...\n";
  (*poolPtr)->getFuture([]() {cout<<"Empty task\n";}).get();
  vector<future<void>> futures;
  const int N= 30;

  // spawn N threads:
  for (int i=0; i<N; ++i)
    futures.emplace_back((*poolPtr)->getFuture(&TestThreadPoolService::print_id, this,i+1));
  go();

  for (auto& future: futures) future.get();
  cout << "\n[basicUseTest] DONE, sum= "<<sum<<"\n";
	for(int i=0; i<N; i++)
		sum-= i+1;
  CPPUNIT_ASSERT_EQUAL(sum, 0l);
}
void TestThreadPoolService::passServiceArgTest()
{
  cout<<"\nStarting passServiceArg test...\n"
      <<"(requires >1 thread, otherwise will never finish)\n";
  (*poolPtr)->getFuture([&]() {
    cout<<"Recursive enqueue #1\n";
    //ServiceRegistry::Operate operate(serviceToken);
    (*poolPtr)->getFuture([]() {cout<<"Pool service captured\n";}).get();
  }).get();
  (*poolPtr)->getFuture([this](Service<service::ThreadPoolService> poolArg){
    cout<<"Recursive enqueue #2\n";
    //ServiceRegistry::Operate operate(serviceToken);
    poolArg->getFuture([]() {cout<<"Pool service passed as arg\n";}).get();
  }, (*poolPtr)).get();
}
void TestThreadPoolService::CUDATest()
{
  cout<<"\nStarting CUDA test...\n";
  vector<future<void>> futures;
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
    futures.emplace_back((*poolPtr)->getFuture(&TestThreadPoolService::cudaTask, this,
                         n, i, din, 2));
  }
  for (auto& future: futures) future.get();
}

#define TOLERANCE 5e-1
void TestThreadPoolService::CUDAAutolaunchManagedTest()
{
  cout<<"\nStarting CUDA autolaunch (managed) test...\n";
  float *in, *out;
  const int n= 10000000, times= 1000;
  cudaMallocManaged(&in, n*sizeof(float));  //cudaMemAttachHost?
  cudaMallocManaged(&out, n*sizeof(float));
  for(int i=0; i<n; i++) in[i]= 10*cos(3.141592/100*i);

  cout<<"Launching auto...\n";
  // Auto launch config
  cudaConfig::ExecutionPolicy execPol((*poolPtr)->configureLaunch(n, longKernel));
  (*poolPtr)->cudaLaunchManaged(execPol, longKernel, (int)n,(int)times,
                          const_cast<const float*>(in),out).get();
  for(int i=0; i<n; i++) if (times*in[i]-out[i]>TOLERANCE || times*in[i]-out[i]<-TOLERANCE){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in[i], out[i], TOLERANCE);
  }

  cout<<"Launching manual...\n";
  // Manual launch config
  execPol= cudaConfig::ExecutionPolicy(320, (n-1+320)/320);
  (*poolPtr)->cudaLaunchManaged(execPol, longKernel, (int)n,(int)times,
                          const_cast<const float*>(in),out).get();
  for(int i=0; i<n; i++) if (times*in[i]-out[i]>TOLERANCE || times*in[i]-out[i]<-TOLERANCE){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in[i], out[i], TOLERANCE);
  }

  cudaFree(in);
  cudaFree(out);
}

void TestThreadPoolService::timeBenchmark()
{
  cout << "Starting quick time benchmark...\n";
  long N= 10000000;
  auto start= chrono::steady_clock::now();
  auto end = start;
  auto diff= start-start;
  future<void> fut;
  int threadN= std::thread::hardware_concurrency();

  vector<future<void>> futVec(threadN);
  diff= start-start;
  for (int i = 0; i <= N/threadN; ++i)
  {
    start = chrono::steady_clock::now();
    for(register int thr=0; thr<threadN; thr++)
      futVec[thr]= (*poolPtr)->getFuture([] (){
        this_thread::sleep_for(chrono::microseconds(1));
      });
    for_each(futVec.begin(), futVec.end(), [] (future<void>& elt) {
      elt.get();
    });
    end = chrono::steady_clock::now();

    diff += (i>0)? end-start: start-start;
  }
  cout << "ThreadPoolService normal operation: "<< chrono::duration <double, nano> (diff).count()/N << " ns" << endl;
}
