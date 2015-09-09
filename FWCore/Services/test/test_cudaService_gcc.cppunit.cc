/*! Unit test suite for the `CudaService`. Each test examines a particular function
  of the service. The way the service is used here __is not typical__ of how
  it should be used in CMSSW code, because:
  - In order to make each test independent, a new `CudaService` is constructed for
  each test, while the previous instance is stopped. In actual CMSSW code services
  are only created implicitly by the framework __and not__ in physics code.
  - The `Service` smart pointer is wrapped in an `std::unique_ptr`, which is not
  the normal use case.

  This test suite can however provide some specific examples of using `CudaService`, but
  for a complete example use of `CudaService`, study the CudaService integration test.
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
  // CPPUNIT_TEST(cudaPointer_compoundDataStructureTest);
  CPPUNIT_TEST(cudaPointerToArithmStructureTest);
  CPPUNIT_TEST(originalKernelTest);
  CPPUNIT_TEST(CPUtaskTimeBenchmark);
  CPPUNIT_TEST(latencyHiding);
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
  //! Test cudaPointer to arithmetic structure
  void cudaPointerToArithmStructureTest();
  //! Test complex data structure with cudaPointer. FAILS! Illegal memory access.
  void cudaPointer_compoundDataStructureTest();
  //! Time the assignment and launch of a few CPU tasks in the `ThreadPool`
  void CPUtaskTimeBenchmark();
  /*! Experiment on how different numbers of threads in `CudaService` affects the
    kernel launch latency. An almost trivial kernel is used and a single thread block
    is launched on the GPU.
  */
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
// for cudaPointer_complexDataStructureTest
struct KernelData{
  int a, b;
  cudaPointer<float[]> arrayIn;
  cudaPointer<float[]> arrayOut;
};
void actOnStructWrapper(bool gpu, const cuda::ExecutionPolicy& execPol,
                        KernelData* data);
// for cudaPointerToArithmStructureTest
struct ArithmStruct{
  int a, b;
  float c;
};
void actOnArithmStructWrapper(bool gpu, const cuda::ExecutionPolicy& execPol,
                              int n, ArithmStruct* inStruct, float* out);
// alternative for latencyHiding
void trivialWrapper(bool gpu, const cuda::ExecutionPolicy& execPol);

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
  for(int i=0; i<n; i++) in[i]= 10*cos(PI/100*i);

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
  cudaPointer<float[]> in (n);
  cudaPointer<float[]> out(n);
  for(int i=0; i<n; i++) in[i]= 10*cos(PI/100*i);
  cout<<"Launching auto...\n";
  // Auto launch config
  (*cuSerPtr)->cudaLaunch((unsigned)n, long_auto, n,times,in,out).get();
  for(int i=0; i<n; i++) if (abs(times*in[i]-out[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in[i], out[i], TOLERANCEmul);
  }
  // 2nd kernel launch...
  for(int i=0; i<n; i++) in[i]= 5*cos(PI/100*i);
  cout<<"Launching manual with explicit auto config...\n";
  // Auto config
  auto execPol= cuda::AutoConfig()(n, (void*)long_kernel);
  (*cuSerPtr)->cudaLaunch(execPol, long_man, n,times,in,out).get();
  for(int i=0; i<n; i++) if (abs(times*in[i]-out[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in[i], out[i], TOLERANCEmul);
  }
}
void TestCudaService::cudaPointer_compoundDataStructureTest(){
  cout<<"Starting \"cudaPointer\" compound data structure test...\n";
  KernelData data;
  int n= 100000;
  data.a= 1, data.b= 2; data.arrayIn.reset(n); data.arrayOut.reset(n);
  for(int i=0; i<n; i++) data.arrayIn[i]= 10*cos(PI/100*i);
  cuda::ExecutionPolicy execPol;
  execPol.setBlockSize({1024}).autoGrid({(unsigned)n});
  cout<<"Launch\n";
  auto status= (*cuSerPtr)->cudaLaunch(execPol, actOnStructWrapper, &data).get();
  cout<<"Cuda status after launch: "<<cudaGetErrorString(status)<<'\n';
  for(int i=0; i<n; i++) if (abs((data.arrayIn[i]+data.a*data.b)-data.arrayOut[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(data.arrayIn[i]+data.a*data.b, data.arrayOut[i], TOLERANCEmul);
  }
}
#define TOLERANCEpow 1e+1
void TestCudaService::cudaPointerToArithmStructureTest(){
  cout<<"Starting \"cudaPointer\" to arithmetic data structure test...\n";
  const int n=100000;
  cudaPointer<ArithmStruct[]> data(n);
  cudaPointer<float[]>        result(n);
  for(int i=0; i<n; i++) {
    data[i].a= round(3*(sin(PI/100*i)+1)); data[i].b= round(3*(cos(PI/100*i)+1));
    data[i].c=i;
  }
  cuda::ExecutionPolicy execPol;
  execPol.setBlockSize({1024}).autoGrid({(unsigned)n});
  cout<<"Launch\n";
  auto status= (*cuSerPtr)->cudaLaunch(execPol, actOnArithmStructWrapper, n,data,result).get();
  cout<<"Cuda status after launch: "<<cudaGetErrorString(status)<<'\n';
  for(int i=0; i<n; i++){
    auto& tmp= data[i];
    if (abs(sin(tmp.c)+tmp.b-tmp.a-result[i])>TOLERANCEpow){
      cout<<"ERROR: i="<<i<<'\n';
      CPPUNIT_ASSERT_DOUBLES_EQUAL(sin(tmp.c)+tmp.b-tmp.a,
                                   result[i], TOLERANCEpow);
    }
  }
}
void TestCudaService::CPUtaskTimeBenchmark(){
  cout << "Starting quick task launch && completion time benchmark...\n";
  const int N= 200000;
  int heavyBurden= 1;
  auto start= chrono::steady_clock::now();
  auto end = start;
  auto diff= start-start;
  future<void> fut;
  //This is how many threads the pool is running with 1 GPU
  int threadN= 4;

  //Make different experiments with different heavyBurden
  for(int repeat=0; repeat<3; repeat++){
    vector<future<void>> futVec(heavyBurden*threadN);
    diff= start-start;
    //Run benchmark N times
    for (int i = 0; i <= N; ++i)
    {
      //Assign [heavyBurden*threadN] tasks and wait for results
      start = chrono::steady_clock::now();
      for(register short task=0; task<heavyBurden*threadN; task++)
        futVec[task]= (*cuSerPtr)->schedule([] (){
          for (register short k=0; k<2; k++)
            cout<<"";
        });
      for(auto&& elt: futVec) {
        elt.get();
      }
      end = chrono::steady_clock::now();

      diff += (i>0)? end-start: start-start;
    }
    cout << "Latency: tasks="<<heavyBurden*threadN<<", threads="<<threadN<<":\t"
         << chrono::duration <double, nano> (diff).count()/N/threadN/heavyBurden << " ns per task" << endl;
    //For the next runs...
    heavyBurden++;
  }
  //tasks=4,  threads=4: 2.2μs per task
  //tasks=8,  threads=4: 1.7μs per task
  //tasks=12, threads=4: 1.6μs per task
}
void TestCudaService::latencyHiding(){
  cout << "Starting kernel launch latency hiding benchmark...\n";
  const long N= 100000;
  auto start= chrono::steady_clock::now();
  auto end = start;
  auto diff= start-start;
  future<void> fut;
  const short threadN= std::thread::hardware_concurrency()-3, heavyBurden= 1;
  const int kernelSize= 1*1024, times= 0;
  // Produce heavyBurden*threadN independent data chunks
  vector<cudaPointer<float[]>> in, out;
  for(int thread=0; thread<heavyBurden*threadN; thread++){
    in.emplace_back(kernelSize), out.emplace_back(kernelSize);
    for(int i=0; i<kernelSize; i++)
      in[thread][i]=i;
  }
  cuda::ExecutionPolicy execPol;
  execPol.setBlockSize({1024}).autoGrid({kernelSize});
  for(int threadsInPool=1; threadsInPool<=threadN; threadsInPool++){
    //Reset thread pool size
    (*cuSerPtr)->clearTasks();
    (*cuSerPtr)->stopWorkers();
    (*cuSerPtr)->setWorkerN(threadsInPool);
    (*cuSerPtr)->startWorkers();
    vector<future<cudaError_t>> futVec(heavyBurden*threadsInPool);
    diff= start-start;
    for (int i = 0; i <= N; ++i)
    {
      //Assign [heavyBurden*threadsInPool] tasks and wait for results
      start = chrono::steady_clock::now();
      for(register short task=0; task<heavyBurden*threadsInPool; task++)
        futVec[task]= (*cuSerPtr)->cudaLaunch(execPol,long_man,kernelSize,times,
                                             in[task],out[task]);
        // futVec[task]= (*cuSerPtr)->cudaLaunch(execPol,trivialWrapper);
      for(auto&& elt: futVec) {
        elt.get();
      }
      end = chrono::steady_clock::now();
      (*cuSerPtr)->clearTasks();
      diff += (i>0)? end-start: start-start;
    }
    cout << "Latency: threads="<<threadsInPool<<", tasks="<<heavyBurden*threadsInPool
         <<": "<< chrono::duration<double, micro>(diff).count()/N<< " μs\t"
         <<chrono::duration<double, micro>(diff).count()/N/threadsInPool/heavyBurden
         <<" μs per task\n";
  }
  //  --> Results for NON-trivial kernel:
  //tasks=4,  threads=4: 20.5μs per task
  //tasks=8,  threads=4: 19.4μs per task
  //tasks=12, threads=4: 18.2μs per task
  //tasks=40, threads=4: 17.1μs per task

  //  --> Results for trivial kernel:
  //tasks=4,  threads=4: 9.3μs per task
  //tasks=8,  threads=4: 8μs per task

  //Observation about the kernel launch latency:
  //  For heavy task burdens, the minimum shifts towards 2-3 threads/GPU;
  //  for low task burdens, the minimum is between 4-5 threads/GPU.
  //  The thread pool's basic latency (CPU tasks) is ~13μs, so the additional steps CudaService
  //  goes through + the GPU's launch overhead vs. the CPU add a little extra latency of ~5μs.
}
#define TOLERANCEorig 1e-5
void TestCudaService::originalKernelTest(){
  cout<< "Starting original kernel test...\n";
  random_device rd;
  mt19937 mt(rd());
  uniform_real_distribution<float> randFl(0, 1000);
  vector<future<void>> futVec(3);
  unsigned meanExp= 1000000;
  cudaPointer<float[]> cls(meanExp), clx(meanExp),
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
