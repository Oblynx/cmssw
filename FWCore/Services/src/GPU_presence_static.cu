#include "FWCore/Services/interface/utils/GPU_presence_static.h"

//Initialize static members
bool cuda::GPUPresenceStatic::status_= false;
bool cuda::GPUPresenceStatic::alreadySet_= false;
