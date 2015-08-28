#include "RecoLocalTracker/SubCollectionProducers/interface/ClusterSummaryProducer.h"

#include <iostream>
#include <vector>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Services/interface/cudaService_TBBQueueBlocking.h"

#define TOLERANCEorig 1e-5
#define CPPUNIT_ASSERT_DOUBLES_EQUAL(expected,actual,delta)   \
    if (abs((expected)-(actual)) > (delta)){                  \
      std::cout << "ASSERTION failed\n"                       \
                << "Expected: "<<(expected)<<"\t"             \
                << "Actual: "<<(actual)<<"\t"                 \
                << "Delta: "<<(delta)<<"\n\n";                \
      break;                                                  \
    }


void simpleTask_auto(unsigned& launchSize, unsigned meanExp,
                            float* cls, float* clx, float* cly);
void simpleTask_man(const cuda::ExecutionPolicy& execPol,
                          unsigned meanExp, float* cls, float* clx, float* cly);
__global__ void simpleTask_kernel(unsigned meanExp, float* cls,
                                         float* clx, float* cly);

ClusterSummaryProducer::ClusterSummaryProducer(const edm::ParameterSet& iConfig)
  : doStrips(iConfig.getParameter<bool>("doStrips")),
    doPixels(iConfig.getParameter<bool>("doPixels")),
    verbose(iConfig.getParameter<bool>("verbose"))
{
 
  pixelClusters_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"));
  stripClusters_ = consumes<edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("stripClusters"));

  ClusterSummary::CMSTracker maxEnum = ClusterSummary::STRIP;

  std::vector<std::string> wantedsubdets = iConfig.getParameter<std::vector<std::string> >("wantedSubDets");
  for(const auto& iS : wantedsubdets){

    ClusterSummary::CMSTracker subdet = ClusterSummary::NVALIDENUMS;
    for(int iN = 0; iN < ClusterSummary::NVALIDENUMS; ++iN)
      if(ClusterSummary::subDetNames[iN] == iS)
        subdet = ClusterSummary::CMSTracker(iN);
    if(subdet == ClusterSummary::NVALIDENUMS) throw cms::Exception( "No standard selection: ") << iS;

    selectors.push_back(ModuleSelection(DetIdSelector(ClusterSummary::subDetSelections[subdet]),subdet));
    if(subdet > maxEnum) maxEnum = subdet;
    if(verbose)moduleNames.push_back(ClusterSummary::subDetNames[subdet]);
  }


  std::vector<edm::ParameterSet> wantedusersubdets_ps = iConfig.getParameter<std::vector<edm::ParameterSet> >("wantedUserSubDets");
  for(const auto& iS : wantedusersubdets_ps){
    ClusterSummary::CMSTracker subdet    = (ClusterSummary::CMSTracker)iS.getParameter<unsigned int>("detSelection");
    std::string                detname   = iS.getParameter<std::string>("detLabel");
    std::vector<std::string>   selection = iS.getParameter<std::vector<std::string> >("selection");

    if(subdet <=  ClusterSummary::NVALIDENUMS) throw cms::Exception( "Already predefined selection: ") << subdet;
    if(subdet >=  ClusterSummary::NTRACKERENUMS) throw cms::Exception( "Selection is out of range: ") << subdet;

    selectors.push_back(ModuleSelection(DetIdSelector(selection),subdet));
    if(subdet > maxEnum) maxEnum = subdet;
    if(verbose)moduleNames.push_back(detname);
  }

  cCluster = ClusterSummary(maxEnum + 1);
  produces<ClusterSummary>().setBranchAlias("trackerClusterSummary");
}

void
ClusterSummaryProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   std::cout << "[ClSumProd]: Entered produce\n";
   using namespace edm;
   cCluster.reset();
   std::vector<bool> selectedVector(selectors.size(),false);
   
   auto getSelections   =  [&] (const uint32_t detid ){
     for(unsigned int iS = 0; iS < selectors.size(); ++iS)
       selectedVector[iS] = selectors[iS].first.isSelected(detid);
   };
   auto fillSelections =  [&] (const int clusterSize, const float clusterCharge ){
     for(unsigned int iS = 0; iS < selectors.size(); ++iS){
       if(!selectedVector[iS]) continue;
       const ClusterSummary::CMSTracker  module   = selectors[iS].second;
       cCluster.addNClusByIndex     (module, 1 );
       cCluster.addClusSizeByIndex  (module, clusterSize );
       cCluster.addClusChargeByIndex(module, clusterCharge );
     }
   };

   //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   // Test code transplanted from: FWCore/Services/test/test_cudaService.cppunit.cu
   // Initial CPU code from: RecoLocalTracker/SubCollectionProducers/src/JetCoreClusterSplitter.cc
   // JetCoreClusterSplitter::fittingSplit
    std::cout<< "[ClSumProd]: Starting integration test...\n\n";
    edm::Service<edm::service::CudaService> cudaService;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> randFl(0, 1000);
    std::vector<std::future<void>> futVec(3);
    unsigned meanExp= 1000;
    cudaPointer<float> cls(meanExp), clx(meanExp),
                       cly(meanExp);
    //Initialize
    std::cout<< "[ClSumProd]: Initializing data...\n";
    futVec[0]= cudaService->schedule([&] {
      for(unsigned i=0; i<meanExp; i++) cls.p[i]= randFl(mt); });
    futVec[1]= cudaService->schedule([&] {
      for(unsigned i=0; i<meanExp; i++) clx.p[i]= randFl(mt); });
    futVec[2]= cudaService->schedule([&] {
      for(unsigned i=0; i<meanExp; i++) cly.p[i]= randFl(mt); });
    for(auto&& fut: futVec) fut.get();

    //Calculate results on CPU
    std::vector<float> cpuCls(meanExp), cpuClx(meanExp), cpuCly(meanExp);
    for (unsigned i= 0; i < meanExp; i++)
    {
      if (cls.p[i] != 0) {
        cpuClx[i]= clx.p[i]/cls.p[i];
        cpuCly[i]= cly.p[i]/cls.p[i];
      }
      cpuCls[i]= 0;
    }
    //Calculate results on GPU
    std::cout<< "[ClSumProd]: Launching on GPU...\n";
    //#define AUTO_TEST
    #ifdef AUTO_TEST
    auto GPUResult= cudaService->cudaLaunch(meanExp, simpleTask_auto, meanExp,
                                                cls, clx, cly);
    #else
    auto execPol= cuda::AutoConfig()(meanExp,(void*)simpleTask_kernel);
    auto GPUResult= cudaService->cudaLaunch(execPol, simpleTask_man, meanExp,
                                               cls, clx, cly);
    #endif
    GPUResult.get();

    std::atomic<unsigned> successN; successN= 0;
    futVec[0]= cudaService->schedule([&] {
      for(unsigned i=0; i<meanExp; i++, successN++)
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cpuCls[i], cls.p[i], TOLERANCEorig);
    });
    futVec[1]= cudaService->schedule([&] {
      for(unsigned i=0; i<meanExp; i++, successN++)
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cpuCls[i], cls.p[i], TOLERANCEorig);
    });
    futVec[2]= cudaService->schedule([&] {
      for(unsigned i=0; i<meanExp; i++, successN++)
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cpuCls[i], cls.p[i], TOLERANCEorig);
    });
    for(auto&& fut: futVec) fut.get();
    if (successN == 3*meanExp) std::cout<< "\n[ClSumProd]: --> PASS Integration test PASS!!!\n\n\n";
    else std::cout<< "\n[ClSumProd]: --> FAIL Integration test FAIL!!!\n\n\n";
   //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   //===================++++++++++++========================
   //                   For SiStrips
   //===================++++++++++++========================
   if (doStrips){
     edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
     iEvent.getByToken(stripClusters_, stripClusters);
     edmNew::DetSetVector<SiStripCluster>::const_iterator itClusters=stripClusters->begin();
     for(;itClusters!=stripClusters->end();++itClusters){
       getSelections(itClusters->id());
       for(edmNew::DetSet<SiStripCluster>::const_iterator cluster=itClusters->begin(); cluster!=itClusters->end();++cluster){
         const ClusterVariables Summaryinfo(*cluster);
         fillSelections(Summaryinfo.clusterSize(),Summaryinfo.charge());
       }
     }
   }
   
   //===================++++++++++++========================
   //                   For SiPixels
   //===================++++++++++++========================
   if (doPixels){
     edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
     iEvent.getByToken(pixelClusters_, pixelClusters);
     edmNew::DetSetVector<SiPixelCluster>::const_iterator itClusters=pixelClusters->begin();
     for(;itClusters!=pixelClusters->end();++itClusters){
       getSelections(itClusters->id());
       for(edmNew::DetSet<SiPixelCluster>::const_iterator cluster=itClusters->begin(); cluster!=itClusters->end();++cluster){
         fillSelections(cluster->size(),float(cluster->charge())/1000.);
       }
     }
   }

   //===================+++++++++++++========================
   //                   Fill Producer
   //===================+++++++++++++========================
   if(verbose){
     for(const auto& iS : selectors){
       const ClusterSummary::CMSTracker  module   = iS.second;
       edm::LogInfo("ClusterSummaryProducer") << "n" << moduleNames[module]   <<", avg size, avg charge = "
           << cCluster.getNClusByIndex     (module ) << ", "
           << cCluster.getClusSizeByIndex  (module )/cCluster.getNClusByIndex(module ) << ", "
           << cCluster.getClusChargeByIndex(module )/cCluster.getNClusByIndex(module)
           << std::endl;
     }
     std::cout << "-------------------------------------------------------" << std::endl;
   }

   //Put the filled class into the producer
   std::auto_ptr<ClusterSummary> result(new ClusterSummary (0) );
   //Cleanup empty selections
   result->copyNonEmpty(cCluster);
   iEvent.put( result );
}


void 
ClusterSummaryProducer::beginStream(edm::StreamID)
{
  if(!verbose) return;
  edm::LogInfo("ClusterSummaryProducer") << "+++++++++++++++++++++++++++++++ "  << std::endl << "Getting info on " ;
    for (const auto& iS : moduleNames ) { edm::LogInfo("ClusterSummaryProducer") << iS<< " " ;}
    edm::LogInfo("ClusterSummaryProducer")  << std::endl;
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusterSummaryProducer);


