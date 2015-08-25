// Service to test
#include "FWCore/Services/interface/cudaService_TBBQueueBlocking.h"
#include "FWCore/Services/interface/utils/cuda_launch_configuration.cuh"

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

#define PI 3.141592
using namespace std;
using namespace edm;

class TestCudaService: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestCudaService);
  CPPUNIT_TEST(basicUseTest);
  CPPUNIT_TEST(passServiceArgTest);
  CPPUNIT_TEST(basicCUDATest);
  CPPUNIT_TEST(CUDAAutolaunchManagedTest);
  CPPUNIT_TEST(CUDAAutolaunch2Dconfig);
  CPPUNIT_TEST(CudaPointerAutolaunchTest);
  CPPUNIT_TEST(timeBenchmarkTask);
  //CPPUNIT_TEST(timeBenchmarkKernel);
  CPPUNIT_TEST(originalKernelTest);
  CPPUNIT_TEST_SUITE_END();
public:
  //!< @brief 
  void setUp();
  //!< @brief Release all service resources, but service destructor can't be called
  void tearDown() {
    (*cuSerPtr)->clearTasks();
    (*cuSerPtr)->stopWorkers();
    cout<<"\n\n";
  }
  void basicUseTest();
  //!< @brief Test behaviour if the task itself enqueues another task in same pool
  void passServiceArgTest();
  //!< @brief Test scheduling many threads that launch CUDA kernels (CudaService.schedule)
  void basicCUDATest();
  //!< @brief Test auto launch cuda kernel with its arguments in managed memory
  void CUDAAutolaunchManagedTest();
  //!< @brief Use auto config to manually configure a 2D kernel launch
  void CUDAAutolaunch2Dconfig();
  //!< @brief Test usage of the smart cuda pointer class "cudaPointer"
  void CudaPointerAutolaunchTest();
  //!< @brief Time a simple use case of the service
  void timeBenchmarkTask();
  //!< @brief Time kernel executions
  void timeBenchmarkKernel();
  //!< @brief Test performance of a kernel made from actual CMS CPU code
  void originalKernelTest();
private:
  void print_id(int id);
  void go();
  void cudaTask(int n, int i, const float* din, int times);
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
///Registration of the test suite so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestCudaService);
__global__ void longKernel(const int n, const int times, const float* in, float* out);
__global__ void matAddKernel(int m, int n, const float* __restrict__ A, 
                              const float* __restrict__ B, float* __restrict__ C);
__global__ void originalKernel(unsigned meanExp, float* cls, float* clx, float* cly);

/*$$$---TESTS BEGIN---$$$*/

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
void TestCudaService::basicCUDATest(){
  cout<<"Starting CUDA test...\n";
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
    futures.emplace_back((*cuSerPtr)->schedule(&TestCudaService::cudaTask, this,
                         n, i, din, 2));
  }
  for (auto& future: futures) future.get();
}
#define TOLERANCEmul 5e-1
void TestCudaService::CUDAAutolaunchManagedTest(){
  if (!(*cuSerPtr)->cudaStatus()){
    cout<<"GPU not available, skipping test.\n";
    return;
  }

  cout<<"Starting CUDA autolaunch (managed) test...\n";
  float *in, *out;
  const int n= 10000000, times= 1000;
  cudaMallocManaged(&in, n*sizeof(float));  //cudaMemAttachHost?
  cudaMallocManaged(&out, n*sizeof(float));
  for(int i=0; i<n; i++) in[i]= 10*cos(3.141592/100*i);

  cout<<"Launching auto config...\n";
  // Auto launch config
  cudaConfig::ExecutionPolicy execPol(cudaConfig::configure((*cuSerPtr)->cudaStatus(),
                                                            n, longKernel));
  (*cuSerPtr)->cudaLaunchManaged(execPol, longKernel, n,times,in,out).get();
  for(int i=0; i<n; i++) if (abs(times*in[i]-out[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in[i], out[i], TOLERANCEmul);
  }

  cout<<"Launching manual config...\n";
  // Manual launch config
  execPol= cudaConfig::ExecutionPolicy(320, (n-1+320)/320);
  (*cuSerPtr)->cudaLaunchManaged(execPol, longKernel, n,times,in,out).get();
  for(int i=0; i<n; i++) if (abs(times*in[i]-out[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in[i], out[i], TOLERANCEmul);
  }

  cudaFree(in);
  cudaFree(out);
}
#define TOLERANCEadd 1e-15
void TestCudaService::CUDAAutolaunch2Dconfig(){
  if (!(*cuSerPtr)->cudaStatus()){
    cout<<"GPU not available, skipping test.\n";
    return;
  }

  cout<<"Starting CUDA 2D launch config test...\n";
  const int threadN= std::thread::hardware_concurrency();
  vector<future<cudaError_t>> futVec(threadN);
  float *A[threadN], *B[threadN], *C[threadN];
  // m: number of rows
  // n: number of columns
  unsigned m= 1000, n= 100;
  //Setup data
  for(int thread=0; thread<threadN; thread++){
    cudaMallocManaged(&A[thread], m*n*sizeof(float));
    cudaMallocManaged(&B[thread], m*n*sizeof(float));
    cudaMallocManaged(&C[thread], m*n*sizeof(float));
    for (int i=0; i<n*m; i++){
      A[thread][i]= 10*(thread+1)*sin(PI/100*i);
      B[thread][i]= (thread+1)*sin(PI/100*i+PI/6)*sin(PI/100*i+PI/6);
    }
  }

  //Recommended launch configuration (1D)
  cudaConfig::ExecutionPolicy execPol(cudaConfig::configure((*cuSerPtr)->cudaStatus(),
                                                    n*m, matAddKernel));
  //Explicitly set desired launch config (2D) based on the previous recommendation
  const unsigned blockSize= sqrt(execPol.getBlockSize().x);
  //Explicitly set blockSize, automatically demand adequate grid size to map the input
  execPol.setBlockSize({blockSize, blockSize}).autoGrid({n,m});

  //Semi-manually launch GPU tasks
  for(int thread=0; thread<threadN; thread++){
    futVec[thread]= (*cuSerPtr)->cudaLaunchManaged(execPol, matAddKernel, m, n,
                                                  A[thread],B[thread],C[thread]);
  }
  cout<<"Launch config:\nBlock="<<execPol.getBlockSize().x<<", "<<execPol.getBlockSize().y;
  cout<<"\nGrid="<<execPol.getGridSize().x<<", "<<execPol.getGridSize().y<<"\n\n";
  //...
  for(auto&& elt: futVec) {
    elt.get();
  }

  for(int thread=0; thread<threadN; thread++){
    //Assert matrix addition correctness
    for (int i=0; i<n*m; i++)
      if (abs(A[thread][i]+B[thread][i]-C[thread][i]) > TOLERANCEadd){
        CPPUNIT_ASSERT_DOUBLES_EQUAL(A[thread][i]+B[thread][i],
                                     C[thread][i], TOLERANCEadd);
      }
    cudaFree(A[thread]); cudaFree(B[thread]); cudaFree(C[thread]);
  }
}
void TestCudaService::CudaPointerAutolaunchTest(){
  cout<<"Starting *CudaPointer* autolaunch test...\n";
  const int n= 10000000, times= 1000;
  cudaPointer<float> in (n);
  cudaPointer<float> out(n);
  for(int i=0; i<n; i++) in.p[i]= 10*cos(3.141592/100*i);

  cout<<"Launching auto...\n";
  // Auto launch config
  cudaConfig::ExecutionPolicy execPol(cudaConfig::configure((*cuSerPtr)->cudaStatus(),
                                                    n, longKernel));
  (*cuSerPtr)->cudaLaunchManaged(execPol, longKernel, n,times,in,out).get();
  for(int i=0; i<n; i++) if (abs(times*in.p[i]-out.p[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in.p[i], out.p[i], TOLERANCEmul);
  }

  cout<<"Launching manual...\n";
  // Manual launch config
  execPol= cudaConfig::ExecutionPolicy(320, (n-1+320)/320);
  (*cuSerPtr)->cudaLaunchManaged(execPol, longKernel, n,times,in,out).get();
  for(int i=0; i<n; i++) if (abs(times*in.p[i]-out.p[i])>TOLERANCEmul){
    cout<<"ERROR: i="<<i<<'\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(times*in.p[i], out.p[i], TOLERANCEmul);
  }
}
void TestCudaService::timeBenchmarkTask(){
  cout << "Starting quick task launch && completion time benchmark...\n";
  long N= 10000;
  auto start= chrono::steady_clock::now();
  auto end = start;
  auto diff= start-start;
  future<void> fut;
  int threadN= std::thread::hardware_concurrency();

  vector<future<void>> futVec(threadN);
  diff= start-start;
  for (int i = 0; i <= N; ++i)
  {
    //Assign [threadN] tasks and wait for results
    start = chrono::steady_clock::now();
    for(register int thr=0; thr<threadN; thr++)
      futVec[thr]= (*cuSerPtr)->schedule([] (){
        for (register short k=0; k<5; k++)
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
    //Assign [threadN] tasks and wait for results
    start = chrono::steady_clock::now();
    for(register int thr=0; thr<threadN; thr++)
      futVec[thr]= (*cuSerPtr)->schedule([] (){
        for (register short k=0; k<5; k++)
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
}/*
void TestCudaService::timeBenchmarkKernel()
{
  cout << "Starting quick task launch && completion time benchmark...\n";
  long N= 10000;
  auto start= chrono::steady_clock::now();
  auto end = start;
  auto diff= start-start;
  future<void> fut;
  int threadN= std::thread::hardware_concurrency();

  vector<future<void>> futVec(threadN);
  diff= start-start;
  for (int i = 0; i <= N; ++i)
  {
    //Assign [threadN] tasks and wait for results
    start = chrono::steady_clock::now();
    for(register int thr=0; thr<threadN; thr++)
      futVec[thr]= (*cuSerPtr)->cudaLaunchManaged();
    for(auto&& elt: futVec) {
      elt.get();
    }
    end = chrono::steady_clock::now();

    diff += (i>0)? end-start: start-start;
  }
  cout << "CudaService at \"natural\" task burden (tasks = threads): "<< chrono::duration <double, nano> (diff).count()/N << " ns" << endl;

  const int heavyBurden= 10;
  threadN*= heavyBurden;
  futVec.resize(threadN);
  diff= start-start;
  for (int i = 0; i <= N; ++i)
  {
    //Assign [threadN] tasks and wait for results
    start = chrono::steady_clock::now();
    for(register int thr=0; thr<threadN; thr++)
      futVec[thr]= (*cuSerPtr)->cudaLaunchManaged();
    for(auto&& elt: futVec) {
      elt.get();
    }
    end = chrono::steady_clock::now();

    diff += (i>0)? end-start: start-start;
  }
  cout << "CudaService at \"heavy\" task burden (tasks = "<<heavyBurden<<" x threads): "<< chrono::duration <double, nano> (diff).count()/N << " ns" << endl;
  cout << "Divided by extra burden: "<< chrono::duration <double, nano> (diff).count()/N/heavyBurden << " ns" << endl;
}*/

#include <random>
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
    for(unsigned i=0; i<meanExp; i++) cls.p[i]= randFl(mt); });
  futVec[1]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++) clx.p[i]= randFl(mt); });
  futVec[2]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++) cly.p[i]= randFl(mt); });
  for(auto&& fut: futVec) fut.get();

  //Calculate results on CPU
  vector<float> cpuCls(meanExp), cpuClx(meanExp), cpuCly(meanExp);
  for (unsigned i= 0; i < meanExp; i++)
  {
    if (cls.p[i] != 0) {
      cpuClx[i]= clx.p[i]/cls.p[i];
      cpuCly[i]= cly.p[i]/cls.p[i];
    }
    cpuCls[i]= 0;
  }
  //Calculate results on GPU
  auto execPol= cudaConfig::configure((*cuSerPtr)->cudaStatus(),
                                      meanExp, originalKernel);
  auto result= (*cuSerPtr)->cudaLaunchManaged(execPol, originalKernel, meanExp,
                                           cls, clx, cly);
  result.get();

  futVec[0]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++)
      CPPUNIT_ASSERT_DOUBLES_EQUAL(cpuCls[i], cls.p[i], TOLERANCEorig);
  });
  futVec[1]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++)
      CPPUNIT_ASSERT_DOUBLES_EQUAL(cpuCls[i], cls.p[i], TOLERANCEorig);
  });
  futVec[2]= (*cuSerPtr)->schedule([&] {
    for(unsigned i=0; i<meanExp; i++)
      CPPUNIT_ASSERT_DOUBLES_EQUAL(cpuCls[i], cls.p[i], TOLERANCEorig);
  });
  for(auto&& fut: futVec) fut.get();  
  cout <<clx.p[100];
}

/*$$$---TESTS END---$$$*/

void TestCudaService::setUp(){
  static atomic_flag notFirstTime= ATOMIC_FLAG_INIT;
  if (!notFirstTime.test_and_set()){
    // Init modelled after "FWCore/Catalog/test/FileLocator_t.cpp"
    // Make the services.
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }
  //Alternative way to setup Services
  /*ParameterSet pSet;
  pSet.addParameter("@service_type", string("CudaService"));
  pSet.addParameter("thread_num", 2*std::thread::hardware_concurrency());
  vector<ParameterSet> vec;
  vec.push_back(pSet);*/
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
__global__ void matAddKernel(int m, int n, const float* __restrict__ A, 
                              const float* __restrict__ B, float* __restrict__ C)
{
  int x= blockIdx.x*blockDim.x + threadIdx.x;
  int y= blockIdx.y*blockDim.y + threadIdx.y;

  // ### Difference between manual and automatic kernel grid:
  if (x<n && y<m)
    C[y*n+x]= A[y*n+x]+B[y*n+x];
  //if (y*n+x < n*m)
    //C[y*n+x]= A[y*n+x]+B[y*n+x];
}
__global__ void originalKernel(unsigned meanExp, float* cls, float* clx, float* cly)
{
  int i= blockDim.x*blockIdx.x+threadIdx.x;
  if(i<meanExp){
    if (cls[i] != 0){
      clx[i] /= cls[i];
      cly[i] /= cls[i];
    }
    cls[i]= 0;
  }
}

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
void TestCudaService::cudaTask(int n, int i, const float* din, int times){
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
