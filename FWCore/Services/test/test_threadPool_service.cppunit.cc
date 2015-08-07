// Interface to test
#include "FWCore/Services/plugins/thread_poolUNI.hxx"
#include "FWCore/Services/interface/PrintLoadingPlugins.h"

// std
#include <iostream>
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>

#include <thread>
#include <chrono>

// CMSSW
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
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
  //CPPUNIT_TEST(CUDATest);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){
    ready= false;
    sum= 0;
  }
  void tearDown(){};
  void basicUseTest();
  void CUDATest();
private:
  void print_id(int id);
  void go();
  //--$--//
  std::mutex mtx;
  std::condition_variable cv;
  bool ready;
  long sum;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestThreadPoolService);


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

void TestThreadPoolService::basicUseTest()
{
  // Init modelled after "FWCore/Catalog/test/FileLocator_t.cpp"
  // Make the services.
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  edm::ServiceToken tempToken = edm::ServiceRegistry::createServicesFromConfig(
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('testThreadPoolService')\n"
      "process.ThreadPoolService = cms.Service('ThreadPoolService')\n"
      "process.PrintLoadingPlugins = cms.Service('PrintLoadingPlugins')\n"
      );
  //make the services available
  edm::ServiceRegistry::Operate operate(tempToken);
  //edm::service::ThreadPoolService pool(paramset, activityreg);
  edm::Service<edm::service::ThreadPoolService> pool;
  std::cout<<"\nStarting test...\n";
  std::this_thread::sleep_for(std::chrono::seconds(1));
  std::cout<<"...Started\n";
  pool->enqueue([]() {std::cout<<"Empty task\n";});

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

void TestThreadPoolService::CUDATest()
{
  
}

