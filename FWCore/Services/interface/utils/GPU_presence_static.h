//! Defines `GPUPresenceStatic` class, __Never__ include this file.
#ifndef GPU_Presence_Static_H
#define GPU_Presence_Static_H

#include <type_traits>
#include <stdexcept>
#include <cuda_runtime_api.h>

//Forward declarations
class cudaPtrBase;
namespace edm{namespace service{class CudaService;}}
namespace cuda{struct AutoConfig;}

namespace cuda{

  /*! Broadcasts the GPU presence to all parts of the system that need to know
    It is set only by `CudaService` and read only by `cudaPointer` and `cuda::AutoConfig`
  */
  class GPUPresenceStatic{
    /*! Stores GPU presence.
      Set only by CudaService constructor (therefore once, by a single thread)
      Read only by cudaPointer instances and cuda::AutoConfig
    */
  	static bool status_;
    //! Checks that only 1 CudaService object has been constructed
    static bool alreadySet_;
  public:
    /*! Allows to __set__ GPU presence only by `CudaService` (in its constructor).
      If another object attempts to call this, there will be a compiler error.
      [release]->throw if multiple services are constructed
    */
  	template<typename T, typename std::enable_if< std::is_same< edm::service::CudaService,
      typename std::remove_reference<typename std::remove_cv<T>::type>::type >
        ::value, int >::type= 0>
  	static void setStatus(const T*, bool newStatus){
      if(!alreadySet_) status_= newStatus, alreadySet_= true;
      // else throw new std::runtime_error("More than 1 CudaService object constructed!");
    }
  	/*! Provides __get__ access only to `cudaPointer` and `cuda::AutoConfig` objects.
      If another object attempts to call this, there will be a compiler error.
    */
  	template<typename T, typename std::enable_if< std::is_base_of< cudaPtrBase,
      typename std::remove_reference<typename std::remove_cv<T>::type>::type >::value
      || std::is_same< AutoConfig, typename std::remove_reference<typename
      std::remove_cv<T>::type>::type >::value, int >::type= 0>
  	static bool getStatus(const T*){ return status_; }
  };
} //namespace cuda

#endif  // GPU_Presence_Static_H
