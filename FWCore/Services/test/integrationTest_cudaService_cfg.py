## @package CudaServiceIntegrationTest
#  Configuration script for cmsRun that sets up an integration test for CudaService
# 
# This test shows an example use of CudaService. It is actually a modified version of
# RecoPixelVertexing/PixelTriplets/test/trip_cfg.py
# 
# For understanding how to use the service, first read the corresponding TWiki page.
# For a clear example, read this cfg script and the following files: 
# - RecoLocalTracker/SiPixelRecHits/plugins/SiPixelRecHitConverter_CudaServiceIntegrationTest.cc
# - RecoLocalTracker/SiPixelRecHits/src/cudaService_integrationTest_kernel.cu
# - RecoLocalTracker/SiPixelRecHits/BuildFile.xml

import FWCore.ParameterSet.Config as cms
process = cms.Process("CudaServiceIntegrationTest")

import FWCore.ParameterSet.Config as cms
process = cms.Process("TKSEEDING")

#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                   ignoreTotal=cms.untracked.int32(1),
                                   oncePerEventMode=cms.untracked.bool(False)
)
## Load CudaService to be tested
# Use the (trivial) cfi script or simply write
# process.CudaService = cms.Service("CudaService")
process.load("FWCore.Services.CudaService_cfi")

## Source *transplanted* from RecoLocalTracker/SiPixelRecHits/test/readRecHits_cfg.py
process.source = cms.Source("PoolSource",
   fileNames =  cms.untracked.vstring(
    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/rechits/rechits2_postls171.root'
   )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('*'),
#    destinations = cms.untracked.vstring('cout'),
#    cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
#)

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cfi import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *
process.triplets = cms.EDAnalyzer("HitTripletProducer",
  OrderedHitsFactoryPSet = cms.PSet(
    ComponentName = cms.string("StandardHitTripletGenerator"),
    SeedingLayers = cms.InputTag("PixelLayerTriplets"),
    GeneratorPSet = cms.PSet( PixelTripletHLTGenerator )
#    GeneratorPSet = cms.PSet( PixelTripletLargeTipGenerator )
  ),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
  )

)
#process.triplets.OrderedHitsFactoryPSet.GeneratorPSet.useFixedPreFiltering = cms.bool(True)
#process.triplets.RegionFactoryPSet.RegionPSet.ptMin = cms.double(1000.00)
#process.triplets.RegionFactoryPSet.RegionPSet.originRadius = cms.double(0.001)
#process.triplets.RegionFactoryPSet.RegionPSet.originHalfLength = cms.double(0.0001)

#######~~~~~ The cms.Path elements are defined in these locations: ~~~~~#######
## siPixelRecHits[EDProducer]: RecoLocalTracker/SiPixelRecHits/interface/SiPixelRecHitConverter.h
## PixelLayerTriplets[EDProducer]: RecoTracker/TkSeedingLayers/plugins/SeedingLayersEDProducer.cc
## triplets[EDAnalyzer]: RecoPixelVertexing/PixelTriplets/test/HitTripletProducer.cc

process.p = cms.Path(process.siPixelRecHits+process.PixelLayerTriplets+process.triplets)
#process.p = cms.Path(process.PixelLayerTriplets+process.triplets)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from:
# SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

