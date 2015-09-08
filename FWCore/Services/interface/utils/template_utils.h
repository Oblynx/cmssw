//! Template utilities for internal use by the CudaService
//! Included with cuda_service.h
#include "cuda_pointer.h"

/**$$$~~~~~ Kernel argument wrapper templates ~~~~~$$$**/
namespace edm{namespace service{
  struct utils{
    /*! @defgroup passKernelArg Template wrappers discriminating `cudaPointer`s from
      other data when passed to kernels. Used like `std::forward`
      @{
    */
    //! Change `cudaPointer` stream attachment and pass its managed data (expect lvalue ref)
    template<typename T, typename std::enable_if< std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
              ::type >::value, int >::type= 0>
    static inline auto passKernelArg(typename std::remove_reference<T>::type& cudaPtr)
        noexcept -> decltype(cudaPtr.p){
      cudaPtr.attachStream();
      return cudaPtr.p;
    }
    //! Perfectly forward non-cudaPointer args (based on std::forward)
    template<typename T, typename std::enable_if< !std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
              ::type >::value, int >::type= 0>
    static inline T&& passKernelArg(typename std::remove_reference<T>::type& valueArg)
        noexcept{      
      return static_cast<T&&>(valueArg);
    }
    template<typename T, typename std::enable_if< !std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
              ::type >::value, int >::type= 0>
    static inline T&& passKernelArg(typename std::remove_reference<T>::type&& valueArg)
        noexcept{
      static_assert(!std::is_lvalue_reference<T>::value,
                    "Can not forward an rvalue as an lvalue.");
      return static_cast<T&&>(valueArg);
    }
    //!@}
    /*! @defgroup releaseKernelArg Template wrappers discriminating `cudaPointer`s,
      which have to release stream attachment when released from the GPU.
      @{
    */
    //! Release stream attachment of `cudaPointer`
    template<typename T, typename std::enable_if< std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
              ::type >::value, int >::type= 0>
    static inline T&& releaseKernelArg(typename std::remove_reference<T>::type& cudaPtr)
        noexcept{
      cudaPtr.releaseStream();
      return static_cast<T&&>(cudaPtr);
    }
    //! If `T != cudaPointer`, do nothing
    template<typename T, typename std::enable_if< !std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
              ::type >::value, int >::type= 0>
    static inline T&& releaseKernelArg(typename std::remove_reference<T>::type& valueArg)
        noexcept{
      return static_cast<T&&>(valueArg);
    }
    template<typename T, typename std::enable_if< !std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
              ::type >::value, int >::type= 0>
    static inline T&& releaseKernelArg(typename std::remove_reference<T>::type&& valueArg)
        noexcept{
      return static_cast<T&&>(valueArg);
    }
    //!@}
    //! Enables to perform an operation on each element of a variadic template param pack
    template<typename... Args>
    static inline void operateParamPack(Args&&...) {}
  };
}} //namespace edm::service
