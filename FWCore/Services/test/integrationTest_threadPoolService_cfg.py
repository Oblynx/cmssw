import FWCore.ParameterSet.Config as cms
process = cms.Process('integrationTestCudaService')

#process.CudaService = cms.Service('CudaService')
process.load("FWCore.Services.CudaService_cfi")

process.source = cms.Source("EmptySource")
#    fileNames      = cms.untracked.vstring('file:/home/ksamaras/ws/example_rootfile.root')
#    fileNames      = cms.untracked.vstring('file:/afs/cern.ch/work/a/astacchi/public/ntuples/DATA/ZTreeProducer_tree_RecoSkimmed.root')
#)
# process.load('RecoLocalTracker.SubCollectionProducers.jetCoreClusterSplitter_cfi')
from RecoLocalTracker.SubCollectionProducers.jetCoreClusterSplitter_cfi import jetCoreClusterSplitter
process.jetTest= jetCoreClusterSplitter
process.path1 = cms.Path(process.jetTest)

# /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.32/x86_64-slc6-gcc49-opt/root
