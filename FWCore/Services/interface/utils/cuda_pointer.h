//! Defines `cudaPointer` smart pointers for CUDA data_ arrays
//! Included with cuda_service.h
#ifndef CUDA_POINTER_H
#define CUDA_POINTER_H

//DEBUG
#include <iostream>
//<Debug

//System
#include <type_traits>
#include <vector>
#include <cuda_runtime_api.h>
//Local components
#include "GPU_presence_static.h"

// Forward declaration
namespace edm{namespace service{struct utils;}}

//! Contains everything not related to cudaPointer's template parameter
class cudaPtrBase{
public:
  enum Attachment{
    global= cudaMemAttachGlobal,
    host= cudaMemAttachHost,
    single= cudaMemAttachSingle,
    noAlloc= 999
  };
  bool GPUpresent() const { return cuda::GPUPresenceStatic::getStatus(this); }
  cudaError_t getErrorState() const { return errorState_; }
protected:
  cudaPtrBase(size_t size, Attachment attach= host, cudaError_t errorState= cudaSuccess,
              bool ownershipReleased= false):
      sizeOnDevice_(size), attachment_(attach), errorState_(errorState),
      ownershipReleased_(ownershipReleased) {}
  //! Signals that the memory can be accessed by the CPU again
  void releaseStream() {attachment_= host;}
  size_t sizeOnDevice_;
  Attachment attachment_;
  cudaError_t errorState_;
  bool ownershipReleased_;
};
// Needed for SFINAE type checking
template<typename, typename =void> class cudaPointer;
/*! `std::unique_ptr`-like managed CUDA smart pointer for arrays of type T.
  @param T non-const, non-ref arithmetic type
  
  - Declared as cudaPointer<T[]>, where T={int, float, double, ...}
  - Uses Unified Memory (`cudaMallocManaged`)
  - __NOT__ thread safe.
*/
template<typename T>
class cudaPointer<T[], typename std::enable_if<std::is_arithmetic<T>::value>::type>: cudaPtrBase{
public:
  /*! Default constructor for creating arrays of type T elements
    @param elementN_ Number of elements in array. If 0, no allocation occurs.
    @param flag Leave at default (`host`) for normal use cases.
    See also CUDA C Programming Guide J.2.2.6

    If used without specifying element number, a call to reset() is required for
    memory allocation.
    Allocation failures result in bad_alloc exception.
  */
  cudaPointer(unsigned elementN= 0, Attachment flag=host):
      cudaPtrBase(elementN*sizeof(T), flag), elementN_(elementN) {
    allocate();
  }
  //!@{ Move semantics
  cudaPointer(cudaPointer&& other) noexcept: cudaPtrBase(other.sizeOnDevice_,
      other.attachment_, other.errorState_, other.ownershipReleased_),
      data_(other.data_), elementN_(other.elementN_) { other.data_= nullptr; }
  cudaPointer& operator=(cudaPointer&& other) noexcept{
    data_= other.data_, other.data_= nullptr, sizeOnDevice_= other.sizeOnDevice_,
    attachment_= other.attachment_, errorState_= other.errorState_,
    ownershipReleased_= other.ownershipReleased_, elementN_= other.elementN_;
    return *this;
  }
  //!@} @{ Delete copy semantics to enforce unique memory ownership
  cudaPointer(const cudaPointer&) =delete;
  cudaPointer& operator=(const cudaPointer&) =delete;
  //!@}
  //! Vector-copy constructor. If vector is empty, throws.
  cudaPointer(const std::vector<T>& vec, Attachment flag=host);
  //! Free memory. If CUDA runtime returns an error code here, it will be lost (destructor doesn't throw).
  ~cudaPointer() noexcept{
    if (!ownershipReleased_) deallocate();
  }
  //! Allocate the requested memory after freeing any previous allocation
  void reset(int elementN){
    if (attachment_ == noAlloc){
      attachment_= host;
      elementN_= elementN; sizeOnDevice_= elementN_*sizeof(T);
      allocate();
    } else if (!ownershipReleased_){
      deallocate();
      elementN_= elementN;
      allocate();
    }
  }
  //! Release ownership and return contained pointer
  T* release(){
    ownershipReleased_= true; T* tmpData= data_; data_= nullptr;
    return tmpData;
  }
  //! Get contained pointer -- __unsafe__
  T* get() const { return data_; }
  void releaseAndFree() { 
    if (!ownershipReleased_){
      ownershipReleased_= true; deallocate();
    }
  }
  //! Construct an `std::vector` out of the contained data
  std::vector<T> getVec() const;
  //! Accesses array element, but throws if data_ is currently being used by the GPU
  T& operator[](int idx) const{
    if(attachment_== host) return data_[idx];
    else throw new std::runtime_error("Illegal cudaPointer access from CPU, "
                                      "while currently in use by the GPU.\n");
  }
  // OVERLOAD for device?
  __device__ T& at(int idx) const{
    return data_[idx];
  }
  int size() { return elementN_; }
  // OVERLOAD for device?
  __device__ int size(bool) { return elementN_; }
private:
  //! Attaches the memory to the provided CUDA stream
  void attachStream(cudaStream_t stream= cudaStreamPerThread){
    attachment_= single;
    if (GPUpresent()) cudaStreamAttachMemAsync(stream, data_, 0, attachment_);
  }
  //! If GPU present, uses `cudaMallocManaged()`, otherwise `new` or `new[]`.
  //! Throws `std::bad_alloc` on allocation failure.
  void allocate();
  //! If GPU present, `cudaFree()`, otherwise `delete` or `delete[]`.
  void deallocate() noexcept;
      //---$$$---//
  //! The contained data_.
  T* data_;
  unsigned elementN_;
  friend struct edm::service::utils;
};
/*! `std::unique_ptr`-like managed CUDA smart pointer for single objects.
  @param T non-const, non-ref arithmetic type
  
  - Uses Unified Memory (`cudaMallocManaged`)
  - __NOT__ thread safe.
  - TODO: future support for STL data_ types, like vectors.
  - TODO: future support for GPU constant && texture memory
*/
template<typename T>
class cudaPointer<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>: cudaPtrBase{
public:
  /*! Default constructor for creating single objects of type T
    @param flag Leave at default (`host`) for normal use cases.
    See also CUDA C Programming Guide J.2.2.6

    Allocation failures result in bad_alloc exception.
  */
  cudaPointer(Attachment flag= host):
      cudaPtrBase(sizeof(T), flag) { allocate(); }
  //!@{ Move semantics
  cudaPointer(cudaPointer&& other) noexcept: cudaPtrBase(other.sizeOnDevice_,
      other.attachment_, other.errorState_, other.ownershipReleased_),
      data_(other.data_) { other.data_= nullptr; }
  cudaPointer& operator=(cudaPointer&& other) noexcept{
    data_= other.data_, other.data_= nullptr, sizeOnDevice_= other.sizeOnDevice_,
    attachment_= other.attachment_, errorState_= other.errorState_,
    ownershipReleased_= other.ownershipReleased_;
    return *this;
  }
  //!@} @{ Delete copy semantics to enforce unique memory ownership
  cudaPointer(const cudaPointer&) =delete;
  cudaPointer& operator=(const cudaPointer&) =delete;
  //!@}
  //! Free memory
  ~cudaPointer() noexcept{
    if (!ownershipReleased_) deallocate();
  }
  //! Release ownership and return contained pointer
  T* release(){
    ownershipReleased_= true;
    T* tmpData= data_;
    data_= nullptr;
    return tmpData;
  }
  //! Get contained pointer -- __unsafe__
  T* get() const { return data_; }
  T* operator->() const { return data_; }
  void freeMem() { 
    if (!ownershipReleased_){
      ownershipReleased_= true; deallocate();
    }
  }
private:
  //! Attaches the memory to the provided CUDA stream
  void attachStream(cudaStream_t stream= cudaStreamPerThread){
    attachment_= single;
    if (GPUpresent()) cudaStreamAttachMemAsync(stream, data_, 0, attachment_);
  }
  //! If GPU present, uses `cudaMallocManaged()`, otherwise `new` or `new[]`.
  //! Throws `std::bad_alloc` on allocation failure.
  void allocate();
  //! If GPU present, `cudaFree()`, otherwise `delete` or `delete[]`.
  void deallocate() noexcept;
      //---$$$---//
  //! The contained data_.
  T* data_;
  friend struct edm::service::utils;
};

/**$$$~~~~~ cudaPointer<T[]> template method definitions ~~~~~$$$**/
//Constructor vector copy
template<typename T>
cudaPointer<T[], typename std::enable_if<std::is_arithmetic<T>::value>::type>::
  cudaPointer(const std::vector<T>& vec, Attachment flag):
    cudaPtrBase(vec.size()*sizeof(T), flag), elementN_(vec.size())
{
  allocate(); // might throw bad_alloc
  for(unsigned i=0; i<elementN_; i++)
    data_[i]= vec[i];
}
template<typename T>
std::vector<T> cudaPointer<T[], typename std::enable_if<std::is_arithmetic<T>::
  value>::type>::getVec() const
{
  std::vector<T> vec;
  vec.reserve(elementN_);
  for(unsigned i=0; i<elementN_; i++)
    vec.push_back(std::move(data_[i]));
  return vec;
}
template<typename T>
void cudaPointer<T[], typename std::enable_if<std::is_arithmetic<T>::value>::type>::allocate(){
  if (!elementN_){
    attachment_= noAlloc;
    return;
  }
  if (GPUpresent()){
    static_assert(!std::is_const<T>::value && !std::is_reference<T>::value,
                  "\nCannot allocate cuda managed memory for const-qualified "
                  "or reference types.\nConsult the CUDA C Programming Guide "
                  "section E.2.2.2. __managed__ Qualifier.\n");
    errorState_= cudaMallocManaged((void**)&data_, sizeOnDevice_, attachment_);
    if (errorState_ != cudaSuccess) throw new std::bad_alloc();
  }
  else data_= new T[elementN_];
}
template<typename T>
void cudaPointer<T[], typename std::enable_if<std::is_arithmetic<T>::value>::type>::deallocate()
  noexcept{
  if (GPUpresent()) errorState_= cudaFree(data_);
  else delete[] data_;
}

/**$$$~~~~~ cudaPointer<T> template method definitions ~~~~~$$$**/
template<typename T>
void cudaPointer<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>::allocate(){
  if (GPUpresent()){
    static_assert(!std::is_const<T>::value && !std::is_reference<T>::value,
                  "\nCannot allocate cuda managed memory for const-qualified "
                  "or reference types.\nConsult the CUDA C Programming Guide "
                  "section E.2.2.2. __managed__ Qualifier.\n");
    errorState_= cudaMallocManaged((void**)&data_, sizeOnDevice_, attachment_);
    if (errorState_ != cudaSuccess) throw new std::bad_alloc();
  }
  else data_= new T;
}
template<typename T>
void cudaPointer<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>::deallocate() noexcept{
  if (GPUpresent()) errorState_= cudaFree(data_);
  else delete data_;
}

#endif	// CUDA_POINTER_H
