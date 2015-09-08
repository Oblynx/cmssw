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

//! Useful for type identification.
class cudaPtrBase {};

/*! `std::unique_ptr`-like managed CUDA smart pointer for single objects.
  @param T non-const, non-ref arithmetic type
  
  - Uses Unified Memory (`cudaMallocManaged`)
  - __NOT__ thread safe.
  - TODO: future support for STL data_ types, like vectors.
  - TODO: future support for GPU constant && texture memory
*/
template<typename T, typename std::enable_if<std::is_arithmetic<T>::value, int>= 0>
class cudaPointer: cudaPtrBase{
public:
	enum Attachment{
		global= cudaMemAttachGlobal,
		host= cudaMemAttachHost,
		single= cudaMemAttachSingle
	};
	/*! Default constructor for creating arrays of type T elements
    @param elementN Number of elements in array
    @param Attachment Leave at default (`host`) for normal use cases.
    See also CUDA C Programming Guide J.2.2.6

    - TODO: Handle (or forward) __malloc failures__ !
  */
	cudaPointer(unsigned elementN=1, Attachment flag=host):
      sizeOnDevice(elementN*sizeof(T)), attachment(flag), ownershipReleased_(false),
      elementN(elementN) { allocate(&data_); }
	//!@{ Move semantics
	cudaPointer(cudaPointer&& other) noexcept: data_(other.data_), sizeOnDevice(other.sizeOnDevice),
      attachment(other.attachment), errorState_(other.errorState_),
			ownershipReleased_(other.ownershipReleased_), elementN(other.elementN) { other.data_= nullptr; }
	cudaPointer& operator=(cudaPointer&& other) noexcept;
  //!@} @{ Delete copy semantics to enforce unique memory ownership
	cudaPointer(const cudaPointer&) =delete;
	cudaPointer& operator=(const cudaPointer&) =delete;
  //!@}
  //! Vector-copy constructor
	cudaPointer(const std::vector<T>& vec, Attachment flag=host);
	//! Free memory. If ownership has not been retained, the `ownershipReleased_`
  //! flag _must have been set._
	~cudaPointer(){
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
	cudaError_t getErrorState() const { return errorState_; }
	//! Construct an `std::vector` out of the contained data
	std::vector<T> getVec() const;
	bool GPUpresent() const { return cuda::GPUPresenceStatic::getStatus(this); }

  //! Accesses array element, but throws if data_ is currently being used by the GPU.
  T& operator[](int idx) const {
    if (attachment==host) return data_[idx];
    else throw new std::runtime_error("Illegal cudaPointer access from CPU, "
                                      "while currently in use by the GPU.");
  }
private:
  //! Attaches the memory to the provided CUDA stream
  void attachStream(cudaStream_t stream= cudaStreamPerThread){
    attachment= single;
    if (GPUpresent()) cudaStreamAttachMemAsync(stream, data_, 0, attachment);
  }
  //! Signals that the memory can be accessed by the CPU again
  void releaseStream() {attachment= host;}
  //! If GPU present, uses `cudaMallocManaged()`, otherwise `new` or `new[]`.
  //! Throws `std::bad_alloc` on allocation failure.
  void allocate();
  //! If GPU present, `cudaFree()`, otherwise `delete` or `delete[]`.
  void deallocate() noexcept;
      //---$$$---//
  //! The contained data_.
  T* data_;
	size_t sizeOnDevice;
	Attachment attachment;
	cudaError_t errorState_;
	bool ownershipReleased_= false;
	unsigned elementN;

  friend struct edm::service::utils;
};

/////////////////////////////////////////////
/*! `std::unique_ptr`-like managed CUDA smart pointer for arrays of type T.
  @param T arithmetic __array__ type
  
  - Uses Unified Memory (`cudaMallocManaged`)
  - __NOT__ thread safe.
*/
template<typename T, typename std::enable_if<std::is_arithmetic<T>::value, int>= 0>
class cudaPointer<T[]>: cudaPtrBase{
public:
  enum Attachment{
    global= cudaMemAttachGlobal,
    host= cudaMemAttachHost,
    single= cudaMemAttachSingle
  };
  /*! Default constructor for creating arrays of type T elements
    @param elementN Number of elements in array
    @param Attachment Leave at default (`host`) for normal use cases.
    See also CUDA C Programming Guide J.2.2.6

    - TODO: Handle (or forward) __malloc failures__ !
  */
  cudaPointer(unsigned elementN=1, Attachment flag=host):
      sizeOnDevice(elementN*sizeof(T)), attachment(flag), ownershipReleased_(false),
      elementN(elementN) { allocate(&data_); }
  //!@{ Move semantics
  cudaPointer(cudaPointer&& other) noexcept: data_(other.data_), sizeOnDevice(other.sizeOnDevice),
      attachment(other.attachment), errorState_(other.errorState_),
      ownershipReleased_(other.ownershipReleased_), elementN(other.elementN) { other.data_= nullptr; }
  cudaPointer& operator=(cudaPointer&& other) noexcept;
  //!@} @{ Delete copy semantics to enforce unique memory ownership
  cudaPointer(const cudaPointer&) =delete;
  cudaPointer& operator=(const cudaPointer&) =delete;
  //!@}
  //! Vector-copy constructor
  cudaPointer(const std::vector<T>& vec, Attachment flag=host);
  //! Free memory. If ownership has not been retained, the `ownershipReleased_`
  //! flag _must have been set._
  ~cudaPointer(){
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
  cudaError_t getErrorState() const { return errorState_; }
  //! Construct an `std::vector` out of the contained data
  std::vector<T> getVec() const;
  bool GPUpresent() const { return cuda::GPUPresenceStatic::getStatus(this); }

  //! Accesses array element, but throws if data_ is currently being used by the GPU.
  T& operator[](int idx) const {
    if (attachment==host) return data_[idx];
    else throw new std::runtime_error("Illegal cudaPointer access from CPU, "
                                      "while currently in use by the GPU.");
  }
private:
  //! Attaches the memory to the provided CUDA stream
  void attachStream(cudaStream_t stream= cudaStreamPerThread){
    attachment= single;
    if (GPUpresent()) cudaStreamAttachMemAsync(stream, data_, 0, attachment);
  }
  //! Signals that the memory can be accessed by the CPU again
  void releaseStream() {attachment= host;}
  //! If GPU present, uses `cudaMallocManaged()`, otherwise `new` or `new[]`.
  //! Throws `std::bad_alloc` on allocation failure.
  void allocate();
  //! If GPU present, `cudaFree()`, otherwise `delete` or `delete[]`.
  void deallocate() noexcept;
      //---$$$---//
  //! The contained data_.
  T* data_;
  size_t sizeOnDevice;
  Attachment attachment;
  cudaError_t errorState_;
  bool ownershipReleased_= false;
  unsigned elementN;

  friend struct edm::service::utils;
};


/////////////////////////////////////////////

/**$$$~~~~~ cudaPointer template method definitions ~~~~~$$$**/
//Move assignment
template<typename T>
cudaPointer<T>& cudaPointer<T>::operator=(cudaPointer&& other) noexcept{
  data_= other.data_; other.data_= nullptr;
  sizeOnDevice= other.sizeOnDevice;
  attachment= other.attachment;
  errorState_= other.errorState_;
  ownershipReleased_= other.ownershipReleased_;
  elementN= other.elementN;
  return *this;
}
//Constructor vector copy
template<typename T>
cudaPointer<T>::cudaPointer(const std::vector<T>& vec, Attachment flag):
    attachment(flag), ownershipReleased_(false), elementN(vec.size())
{
  sizeOnDevice= elementN*sizeof(T);
  allocate(); // might throw bad_alloc
  for(unsigned i=0; i<elementN; i++)
    data_[i]= vec[i];
}
template<typename T>
std::vector<T> cudaPointer<T>::getVec() const{
  std::vector<T> vec;
  vec.reserve(elementN);
  for(unsigned i=0; i<elementN; i++)
    vec.push_back(std::move(data_[i]));
  return vec;
}
template<typename T>
void cudaPointer<T>::allocate(){
  if (GPUpresent()){
    static_assert(!std::is_const<T>::value && !std::is_reference<T>::value,
                  "\nCannot allocate cuda managed memory for const-qualified "
                  "or reference types.\nConsult the CUDA C Programming Guide "
                  "section E.2.2.2. __managed__ Qualifier.\n");
    errorState_= cudaMallocManaged((void**)&data_, sizeOnDevice, attachment);
    if (errorState_ != cudaSuccess) throw new std::bad_alloc();
  }
  else if (elementN==1)
    data_= new T;
  else
    data_= new T[elementN];
}
template<typename T>
void cudaPointer<T>::deallocate() noexcept{
  if (GPUpresent()) errorState_= cudaFree(data_);
  else if (elementN==1)
    delete data_;
  else
    delete[] data_;
}

#endif	// CUDA_POINTER_H
