#ifndef GPU_Presence_Static_H
#define GPU_Presence_Static_H

#include <cuda_runtime_api.h>
#include <type_traits>

//Forward declaration
class cudaPtrBase;
namespace edm{namespace service{class CudaService;}}

namespace cuda{
  struct AutoConfig;

  // Private static cuda status_
  // Set only by CudaService constructor (therefore once, by a single thread)
  // Read only by cudaPointer instances, cuda::AutoConfig
  class GPUPresenceStatic{
  	static bool status_;
  public:
    // SET access: only CudaService
  	template<typename T, typename std::enable_if< std::is_same< edm::service::CudaService,
      typename std::remove_reference<typename std::remove_cv<T>::type>::type >
        ::value, int >::type= 0>
  	static void setStatus(const T*, bool newStatus){ status_= newStatus; }
  	// GET access: only cudaPointer
  	template<typename T, typename std::enable_if< std::is_base_of< cudaPtrBase,
      typename std::remove_reference<typename std::remove_cv<T>::type>::type >::value
      || std::is_same< AutoConfig, typename std::remove_reference<typename
      std::remove_cv<T>::type>::type >::value, int >::type= 0>
  	static bool getStatus(const T*){ return status_; }
  };
} //namespace cuda

#endif  // GPU_Presence_Static_H
