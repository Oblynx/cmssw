//! Initialize static members of class `cuda::GPUPresenceStatic`
#include "FWCore/Services/interface/utils/GPU_presence_static.h"

bool cuda::GPUPresenceStatic::status_= false;
bool cuda::GPUPresenceStatic::alreadySet_= false;
