/*! Unit test suite for the `CudaService`. Each test examines a particular function
  of the service. The way the service is used here __is not typical__ of how
  it should be used in CMSSW code, because:
  - In order to make each test independent, a new `CudaService` is constructed for
  each test, while the previous instance is stopped. In actual CMSSW code services
  are only created implicitly by the framework __and not__ in physics code.
  - The `Service` smart pointer is wrapped in an `std::unique_ptr`.

  This test suite can provide some specific examples of using `CudaService`, but for
  a complete example use of `CudaService`, look at the CudaService integration test.
  @sa integrationTest_cudaService_cfg.py
*/
// Service to test
#include "FWCore/Services/interface/cuda_service.h"
// system
#include <iostream>
#include <vector>
#include <future>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <thread>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
// CMSSW
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
// cppunit-specific
#include "cppunit/extensions/HelperMacros.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"

#define PI 3.141592
using namespace std;
using namespace edm;

class TestCudaService: public CppUnit::TestFixture {
  //CPPUNIT: declare tests to be run
  CPPUNIT_TEST_SUITE(TestCudaService);
  CPPUNIT_TEST(basicUseTest);
  CPPUNIT_TEST(passServiceArgTest);
  CPPUNIT_TEST(cudaLaunch_managedDataTest);
  CPPUNIT_TEST(cudaLaunch_managedData_2DLaunchConfigTest);
  CPPUNIT_TEST(cudaPointer_cudaLaunchTest);
  CPPUNIT_TEST(timeBenchmarkTask);
  CPPUNIT_TEST(latencyHiding);
  CPPUNIT_TEST(originalKernelTest);
  CPPUNIT_TEST_SUITE_END();
public:
  //! Construct a new `CudaService` for each test, to make the tests independent
  void setUp();
  //! Release all service resources, but service destructor can't be called here
  void tearDown() {
    (*cuSerPtr)->clearTasks();
    (*cuSerPtr)->stopWorkers();
    cout<<"\n\n";
  }
  //! Test basic `ThreadPool` functionality
  void basicUseTest();
  //! Test behaviour if the task itself enqueues another task in same `ThreadPool`
  void passServiceArgTest();
  //! Test cudaLaunch API with kernel args declared manually in managed memory
  void cudaLaunch_managedDataTest();
  //! Use `AutoConfig()` as a recommendation to manually configure a 2D kernel launch
  void cudaLaunch_managedData_2DLaunchConfigTest();
  //! Test usage of the smart cuda pointer class `cudaPointer`
  void cudaPointer_cudaLaunchTest();
  //! Time the assignment and launch of a few (CPU) tasks in the `ThreadPool`
  void timeBenchmarkTask();
  //! Experiment on how more threads in `CudaService` affects kernel launch latency
  void latencyHiding();
  //! Test performance of a kernel made from actual CMS physics CPU code
  void originalKernelTest();
private:
  void print_id(int id);
  void go();
  //--$--//
  mutex mtx;
  condition_variable cv;
  bool ready= false;
  atomic<long> sum;
  const int BLOCK_SIZE= 32;

  ServiceToken serviceToken;
  unique_ptr<ServiceRegistry::Operate> operate;
  unique_ptr<Service<service::CudaService>> cuSerPtr;
};

//! CPPUNIT: Registration of the test suite so that the test runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestCudaService);

/*$$$--- KERNEL DECLARATIONS ---$$$*/
__global__ void long_kernel(const int n, const int times, const float* in, float* out);
void long_auto(bool gpu, unsigned& launchSize, const int n, const int times, const float* in, float* out);
void long_man(bool gpu, const cuda::ExecutionPolicy& execPol, const int n,
                     const int times, const float* in, float* out);
__global__ void matAdd_kernel(int m, int n, const float* __restrict__ A, 
                              const float* __restrict__ B, float* __restrict__ C);
void matAdd_auto(bool gpu, unsigned& launchSize, int m, int n, const float* __restrict__ A, 
                            const float* __restrict__ B, float* __restrict__ C);
void matAdd_man(bool gpu, const cuda::ExecutionPolicy& execPol,int m, int n, const float*
              __restrict__ A, const float* __restrict__ B, float* __restrict__ C);
__global__ void original_kernel(unsigned meanExp, float* cls, float* clx, float* cly);
void original_man(bool gpu, const cuda::ExecutionPolicy& execPol,
                unsigned meanExp, float* cls, float* clx, float* cly);

/*$$$--- TESTS BEGIN ---$$$*/
void TestCudaService::basicUseTest(){
  cout<<"Starting basic test...\n";
  (*cuSerPtr)->schedule([]() {cout<<"Empty task\n";}).get();
  vector<future<void>> futures;
  const int N= 30;
  sum= 0;
  // spawn N threads:
  for (int i=0; i<N; ++i)
    futures.emplace_back((*cuSerPtr)->schedule(&TestCudaService::print_id, this,i+1));
  go();

  for (auto& future: futures) future.get();
  cout << "\n[basicUseTest] DONE, sum= "<<sum<<"\n";
  for(int i=0; i<N; i++)
    sum-= i+1;
  CPPUNIT_ASSERT_EQUAL(sum.load(), 0l);
}
void TestCudaService::passServiceArgTest(){
  cout<<"Starting passServiceArg test...\n"
      <<"(requires service with >1 thread)\n";
  (*cuSerPtr)->schedule([&]() {
    cout<<"Recursive enqueue #1\n";
    ServiceRegistry::Operate operate(serviceToken);
    (*cuSerPtr)->schedule([]() {cout<<"Pool service ref captured\n";}).get();
  }).get();
  (*cuSerPtr)->schedule([this](const Service<service::CudaService> poolArg){
    cout<<"Recursive enqueue #2\n";
    ServiceRegistry::Operate operate(serviceToken);
    poolArg->schedule([]() {cout<<"Pool service passed as arg (Service<>->val)\n";}).get();
  }, *cuSerPtr).get();
  (*cuSerPtr)->schedule([this](const Service<service::CudaService>& poolArg){
    cout<<"Recursive enqueue #3\n";
    ServiceRegistry::Operate operate(serviceToken);
    poolArg->schedule([]() {cout<<"Pool service passed as arg (Service<>->cref)\n";}).get();
  }, std::cref(*cuSerPtr)).get();
}
#define TOLERANCEmul 5e-1
void TestCudaService::cudaLaunch_managedDataTest(){
  if (!(*cuSerPtr)->GPUpresent()){
    cout<<"GPU not available, skipping test.\n";
    return;
  }
  cout<<"Starting cudaLaunch with manual managed data test...\n";
  float *in, *out;
  const int n= 10000000, times= 1000;
  cudaMallocManaged(&in, n*sizeof(float));  //cudaMemAttachHost?
  cudaMallocManaged(&out, n*sizeof(float));
  for(int i=0; i<n; i++) in[i]= 10*cos(3.141592/100*i);

  cout<<"Launching auto config...\n";
  // Auto launch config
  (*cuSerPtr)->cudaLaunch((unsigned)n, long_auto, n,times,in,out).get();
  for(int i=0; i<n; i++) if (abs(times*in[i]-out[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in[i], out[i], TOLERANCEmul);
  }

  cout<<"Launching manual config...\n";
  // Manual launch config
  auto execPol= cuda::ExecutionPolicy(320, (n-1+320)/320);
  (*cuSerPtr)->cudaLaunch(execPol, long_man, n,times,in,out).get();
  for(int i=0; i<n; i++) if (abs(times*in[i]-out[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in[i], out[i], TOLERANCEmul);
  }

  cudaFree(in);
  cudaFree(out);
}
#define TOLERANCEadd 1e-15
void TestCudaService::cudaLaunch_managedData_2DLaunchConfigTest(){
  if (!(*cuSerPtr)->GPUpresent()){
    cout<<"GPU not available, skipping test.\n";
    return;
  }
  cout<<"Starting cudaLaunch with manual managed data 2D launch configuration test...\n";
  const int threadN= std::thread::hardware_concurrency();
  vector<future<cudaError_t>> futVec(threadN);
  float *A[threadN], *B[threadN], *C[threadN];
  // m: number of rows
  // n: number of columns
  unsigned m= 1000, n= 100;
  // Setup data
  for(int thread=0; thread<threadN; thread++){
    cudaMallocManaged(&A[thread], m*n*sizeof(float));
    cudaMallocManaged(&B[thread], m*n*sizeof(float));
    cudaMallocManaged(&C[thread], m*n*sizeof(float));
    for (unsigned i=0; i<n*m; i++){
      A[thread][i]= 10*(thread+1)*sin(PI/100*i);
      B[thread][i]= (thread+1)*sin(PI/100*i+PI/6)*sin(PI/100*i+PI/6);
    }
  }
  //Get recommended launch configuration (1D)
  auto execPol= cuda::AutoConfig()(n*m, (void*)matAdd_kernel);
  //Explicitly set desired launch config (2D) based on the previous recommendation
  const unsigned blockSize= sqrt(execPol.getBlockSize().x);
  //Explicitly set blockSize, automatically demand adequate grid size to map the input
  execPol.setBlockSize({blockSize, blockSize}).autoGrid({n,m});

  //Semi-manually launch GPU tasks
  for(int thread=0; thread<threadN; thread++){
    futVec[thread]= (*cuSerPtr)->cudaLaunch(execPol, matAdd_man, m, n,
                                            A[thread],B[thread],C[thread]);
  }
  cout<<"Launch config:\nBlock="<<execPol.getBlockSize().x<<", "<<execPol.getBlockSize().y;
  cout<<"\nGrid="<<execPol.getGridSize().x<<", "<<execPol.getGridSize().y<<"\n\n";
  //... <work with other data here>
  for(auto&& elt: futVec) {
    elt.get();
  }
  for(int thread=0; thread<threadN; thread++){
    //Assert matrix addition correctness
    for (unsigned i=0; i<n*m; i++)
      if (abs(A[thread][i]+B[thread][i]-C[thread][i]) > TOLERANCEadd){
        CPPUNIT_ASSERT_DOUBLES_EQUAL(A[thread][i]+B[thread][i],
                                     C[thread][i], TOLERANCEadd);
      }
    cudaFree(A[thread]); cudaFree(B[thread]); cudaFree(C[thread]);
  }
}
void TestCudaService::cudaPointer_cudaLaunchTest(){
  cout<<"Starting \"cudaPointer\" test...\n";
  const int n= 10000000, times= 1000;
  cudaPointer<float> in (n);
  cudaPointer<float> out(n);
  for(int i=0; i<n; i++) in[i]= 10*cos(3.141592/100*i);
  cout<<"Launching auto...\n";
  // Auto launch config
  (*cuSerPtr)->cudaLaunch((unsigned)n, long_auto, n,times,in,out).get();
  for(int i=0; i<n; i++) if (abs(times*in[i]-out[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in[i], out[i], TOLERANCEmul);
  }
  // 2nd kernel launch...
  for(int i=0; i<n; i++) in[i]= 5*cos(3.141592/100*i);
  cout<<"Launching manual with explicit auto config...\n";
  // Auto config
  auto execPol= cuda::AutoConfig()(n, (void*)long_kernel);
  (*cuSerPtr)->cudaLaunch(execPol, long_man, n,times,in,out).get();
  for(int i=0; i<n; i++) if (abs(times*in[i]-out[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in[i], out[i], TOLERANCEmul);
  }
}
void TestCudaService::timeBenchmarkTask(){
  cout << "Starting quick task launch && completion time benchmark...\n";
  long N= 5000;
  auto start= chrono::steady_clock::now();
  auto end = start;
  auto diff= start-start;
  future<void> fut;
  int threadN= ((*cuSerPtr)->GPUpresent())? 4:
                  std::thread::hardware_concurrency();
  //threadN= std::thread::hardware_concurrency();

  vector<future<void>> futVec(threadN);
  diff= start-start;
  for (int i = 0; i <= N; ++i)
  {
    //Assign [threadN] tasks and wait for results
    start = chrono::steady_clock::now();
    for(register int thr=0; thr<threadN; thr++)
      futVec[thr]= (*cuSerPtr)->schedule([] (){
        for (register short k=0; k<2; k++)
          cout<<"";
      });
    for(auto&& elt: futVec) {
      elt.get();
    }
    end = chrono::steady_clock::now();

    diff += (i>0)? end-start: start-start;
  }
  cout << "CudaService at \"natural\" task burden (tasks = threads):\t\t"
       << chrono::duration <double, nano> (diff).count()/N << " ns" << endl;

  const int heavyBurden= 10;
  threadN*= heavyBurden;
  futVec.resize(threadN);
  diff= start-start;
  for (int i = 0; i <= N; ++i)
  {
    //Assign [threadN] tasks (heavyBurden * threads in pool) and wait for results
    start = chrono::steady_clock::now();
    for(register int thr=0; thr<threadN; thr++)
      futVec[thr]= (*cuSerPtr)->schedule([] (){
        for (register short k=0; k<2; k++)
          cout<<"";
      });
    for(auto&& elt: futVec) {
      elt.get();
    }
    end = chrono::steady_clock::now();

    diff += (i>0)? end-start: start-start;
  }
  cout << "CudaService at \"heavy\" task burden (tasks = "<<heavyBurden<<" x threads):\t"
       << chrono::duration <double, nano> (diff).count()/N/heavyBurden << " ns" << endl;
  //Result: 6561.77 ns
}
void TestCudaService::latencyHiding()
{
  cout << "Starting latency hiding benchmark...\n";
  const long N= 3000;
  auto start= chrono::steady_clock::now();
  auto end = start;
  auto diff= start-start;
  future<void> fut;
  const short threadN= std::thread::hardware_concurrency();
  const int kernelSize= 10000, times= 1000;
  vector<cudaPointer<float>> in, out;             //threadN data chunks
  for(int thread=0; thread<threadN; thread++){
    in.emplace_back(kernelSize), out.emplace_back(kernelSize);
    for(int i=0; i<kernelSize; i++)
      in[thread][i]=i;
  }
  auto execPol= cuda::AutoConfig()(kernelSize, (void*)long_kernel);
  for(int threads=1; threads<=threadN; threads++){
    //Reset thread pool size
    (*cuSerPtr)->clearTasks();
    (*cuSerPtr)->stopWorkers();
    (*cuSerPtr)->setWorkerN(threads);
    (*cuSerPtr)->startWorkers();
    vector<future<cudaError_t>> futVec(threads);
    diff= start-start;
    for (int i = 0; i <= N; ++i)
    {
      //Assign [threads] tasks and wait for results
      start = chrono::steady_clock::now();
      for(register short thr=0; thr<threads; thr++)
        futVec[thr]= (*cuSerPtr)->cudaLaunch(execPol,long_man,kernelSize,times,
                                             in[threads-1],out[threads-1]);
      for(auto&& elt: futVec) {
        elt.get();
      }
      end = chrono::steady_clock::now();
      (*cuSerPtr)->clearTasks();
      diff += (i>0)? end-start: start-start;
    }
    cout << "Latency (tasks=threads="<<threads<<"): "<< chrono::duration<double, nano>(diff).count()/N/1000
         << " μs\t"<<chrono::duration<double, nano>(diff).count()/N/threads/1000<<" μs per thread\n";
  }
}
#define TOLERANCEorig 1e-5
void TestCudaService::originalKernelTest(){
  cout<< "Starting original kernel test...\n";
  random_device rd;
  mt19937 mt(rd());
  uniform_real_distribution<float> randFl(0, 1000);
  vector<future<void>> futVec(3);
  unsigned meanExp= 1000000;
  cudaPointer<float> cls(meanExp), clx(meanExp),
                     cly(meanExp);
  //Initialize
  futVec[0]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++) cls[i]= randFl(mt); });
  futVec[1]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++) clx[i]= randFl(mt); });
  futVec[2]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++) cly[i]= randFl(mt); });
  for(auto&& fut: futVec) fut.get();

  //Calculate results on CPU
  vector<float> cpuCls(meanExp), cpuClx(meanExp), cpuCly(meanExp);
  for (unsigned i= 0; i < meanExp; i++)
  {
    if (cls[i] != 0) {
      cpuClx[i]= clx[i]/cls[i];
      cpuCly[i]= cly[i]/cls[i];
    }
    cpuCls[i]= 0;
  }
  //Calculate results on GPU
  auto execPol= cuda::AutoConfig()(meanExp, (void*)original_kernel);
  auto result= (*cuSerPtr)->cudaLaunch(execPol, original_man, meanExp,
                                       cls, clx, cly);
  result.get();

  futVec[0]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++)
      CPPUNIT_ASSERT_DOUBLES_EQUAL(cpuCls[i], cls[i], TOLERANCEorig);
  });
  futVec[1]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++)
      CPPUNIT_ASSERT_DOUBLES_EQUAL(cpuCls[i], cls[i], TOLERANCEorig);
  });
  futVec[2]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++)
      CPPUNIT_ASSERT_DOUBLES_EQUAL(cpuCls[i], cls[i], TOLERANCEorig);
  });
  for(auto&& fut: futVec) fut.get();  
  cout <<clx[100];
}
/*$$$--- TESTS END ---$$$*/

void TestCudaService::setUp(){
  static atomic_flag notFirstTime= ATOMIC_FLAG_INIT;
  if (!notFirstTime.test_and_set()){
    // Init modelled after "FWCore/Catalog/test/FileLocator_t.cpp"
    // Make the services.
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }
  //Alternative way to setup Services
  // ParameterSet pSet;
  // pSet.addParameter("@service_type", string("CudaService"));
  // pSet.addParameter("thread_num", 2*std::thread::hardware_concurrency());
  // vector<ParameterSet> vec;
  // vec.push_back(pSet);
  serviceToken= edm::ServiceRegistry::createServicesFromConfig(
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('testThreadPoolService')\n"
        "process.CudaService = cms.Service('CudaService')\n");
  operate= unique_ptr<ServiceRegistry::Operate>(
      //new ServiceRegistry::Operate(edm::ServiceRegistry::createSet(vec)));
      new ServiceRegistry::Operate(serviceToken));
  cuSerPtr.reset(new Service<service::CudaService>());
  (*cuSerPtr)->startWorkers();
}

//~~ Functions only used by some tests ~~//
void TestCudaService::print_id(int id) {
  unique_lock<mutex> lck(mtx);
  while (!ready) cv.wait(lck);
  // ...
  cout << id << "\t";
  sum+= id;
}
void TestCudaService::go() {
  unique_lock<mutex> lck(mtx);
  ready = true;
  cv.notify_all();
}
