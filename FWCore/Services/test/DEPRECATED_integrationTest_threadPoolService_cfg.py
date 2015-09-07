import FWCore.ParameterSet.Config as cms
process = cms.Process('integrationTestCudaService')

#process.CudaService = cms.Service('CudaService')
process.load("FWCore.Services.CudaService_cfi")

#===============================================================================
# Event data
process.source = cms.Source("EmptySource",
   numberEventsInRun = cms.untracked.uint32(1), # do not change!
   firstRun = cms.untracked.uint32(1)
)

#===============================================================================
# Condition Data
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_CONDITIONS"
process.GlobalTag.globaltag = "GR_E_V48"

# process.options= cms.untracked.PSet(
# 	 SkipEvent = cms.untracked.vstring('ProductNotFound'))
# from RecoLocalTracker.SubCollectionProducers.jetCoreClusterSplitter_cfi import jetCoreClusterSplitter
# process.test= jetCoreClusterSplitter

from RecoLocalTracker.SubCollectionProducers.clustersummaryproducer_cfi import clusterSummaryProducer
process.test= clusterSummaryProducer
process.path1 = cms.Path(process.test)

# /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.32/x86_64-slc6-gcc49-opt/root
