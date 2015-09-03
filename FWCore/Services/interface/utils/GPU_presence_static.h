#ifndef GPU_Presence_Static_H
#define GPU_Presence_Static_H

#include <cuda_runtime_api.h>
#include <type_traits>
#include <stdexcept>

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
    //!< @brief Checks that only 1 CudaService object is constructed
    static bool alreadySet_;
  public:
    // SET access: only CudaService
  	template<typename T, typename std::enable_if< std::is_same< edm::service::CudaService,
      typename std::remove_reference<typename std::remove_cv<T>::type>::type >
        ::value, int >::type= 0>
    //!< @brief Set GPU presence. In release versions, throw if multiple services are constructed
  	static void setStatus(const T*, bool newStatus){
      if(!alreadySet_) status_= newStatus, alreadySet_= true;
      // else throw new std::runtime_error("More than 1 CudaService object constructed!");
    }
  	// GET access: only cudaPointer
  	template<typename T, typename std::enable_if< std::is_base_of< cudaPtrBase,
      typename std::remove_reference<typename std::remove_cv<T>::type>::type >::value
      || std::is_same< AutoConfig, typename std::remove_reference<typename
      std::remove_cv<T>::type>::type >::value, int >::type= 0>
  	static bool getStatus(const T*){ return status_; }
  };
} //namespace cuda

#endif  // GPU_Presence_Static_H
