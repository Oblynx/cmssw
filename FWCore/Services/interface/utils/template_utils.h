#include "cuda_pointer.h"

/**$$$~~~~~ Kernel argument wrapper templates ~~~~~$$$**/
namespace edm{namespace service{namespace utils{

  //!< @brief Manipulate cudaPtr and pass the included pointer (expect lvalue ref)
  template<typename T, typename std::enable_if< std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
                  ::type >::value, int >::type= 0>
  inline auto passKernelArg(typename std::remove_reference<T>::type& cudaPtr) noexcept
    -> decltype(cudaPtr.p)
  {
    //std::cout<<"[passKernelArg]: Managed arg!\n";
    cudaPtr.attachStream();
    return cudaPtr.p;
  }

  //!< @brief Perfectly forward non-cudaPointer args (based on std::forward)
  template<typename T, typename std::enable_if< !std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
                  ::type >::value, int >::type= 0>
  inline T&& passKernelArg(typename std::remove_reference<T>::type& valueArg) noexcept{
    //std::cout<<"[passKernelArg]: value arg\n";
    return static_cast<T&&>(valueArg);
  }
  template<typename T, typename std::enable_if< !std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
                  ::type >::value, int >::type= 0>
  inline T&& passKernelArg(typename std::remove_reference<T>::type&& valueArg) noexcept{
    static_assert(!std::is_lvalue_reference<T>::value,
                  "Can not forward an rvalue as an lvalue.");
    //std::cout<<"[passKernelArg]: value arg (rvalue ref)\n";
    return static_cast<T&&>(valueArg);
  }
  //!< @brief Release stream attachment of cudaPtr
  template<typename T, typename std::enable_if< std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
                  ::type >::value, int >::type= 0>
  inline T&& releaseKernelArg(typename std::remove_reference<T>::type& cudaPtr) noexcept{
    cudaPtr.releaseStream();
    return static_cast<T&&>(cudaPtr);
  }
  //!< @brief If T != cudaPointer, do nothing
  template<typename T, typename std::enable_if< !std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
                  ::type >::value, int >::type= 0>
  inline T&& releaseKernelArg(typename std::remove_reference<T>::type& valueArg) noexcept{
    return static_cast<T&&>(valueArg);
  }
  template<typename T, typename std::enable_if< !std::is_base_of< cudaPtrBase,
              typename std::remove_reference<typename std::remove_cv<T>::type>
                  ::type >::value, int >::type= 0>
  inline T&& releaseKernelArg(typename std::remove_reference<T>::type&& valueArg) noexcept{
    return static_cast<T&&>(valueArg);
  }

  template<typename... Args>
  inline void operateParamPack(Args&&...) {}
}}} //namespace edm::service::utils
