//! Defines `cudaPointer` smart pointers for CUDA data arrays
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

//! Useful for type identification.
class cudaPtrBase {};
/*! `std::unique_ptr`-like managed CUDA smart pointer.
  @param T non-const, non-ref POD data type
  
  - Creates C-like arrays of T type elements
  - Uses Unified Memory (`cudaMallocManaged`)
  - __NOT__ thread safe.
  - TODO: future support for STL data types, like vectors.
  - TODO: future support for GPU constant && texture memory
*/
template<typename T>
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
      elementN(elementN) { allocate(); }
	//!@{
  //! Move semantics
	cudaPointer(cudaPointer&& other) noexcept: p(other.p), sizeOnDevice(other.sizeOnDevice),
      attachment(other.attachment), errorState_(other.errorState_),
			ownershipReleased_(other.ownershipReleased_), elementN(other.elementN) { other.p= NULL; }
	cudaPointer& operator=(cudaPointer&& other) noexcept;
  //!@}
  //!@{
	//!Delete copy semantics to enforce unique memory ownership
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
	//! Attaches the memory to the provided CUDA stream
	void attachStream(cudaStream_t stream= cudaStreamPerThread){
	  attachment= single;
	  if (GPUpresent()) cudaStreamAttachMemAsync(stream, p, 0, attachment);
	}
  //! Signals that the memory can be accessed by the CPU again
  void releaseStream() {attachment= host;}
	cudaError_t getErrorState() const { return errorState_; }
	/*! Construct an `std::vector` out of the contained data.
    @param release Explicitly signal that the ownership of the data should be released
    (releases the data and sets the internal `ownershipReleased_` flag)
  */
	std::vector<T> getVec(bool release= false);
	bool GPUpresent() { return cuda::GPUPresenceStatic::getStatus(this); }

	//public!
  /*! The contained data.

    - TODO: Make private and provide array API overloads (like []), that check
    whether the memory is accessible from the CPU.
  */
	T* p;
private:
	void allocate();
	void deallocate();
	size_t sizeOnDevice;
	Attachment attachment;
	cudaError_t errorState_;
	bool ownershipReleased_= false;
	unsigned elementN;
};

/**$$$~~~~~ cudaPointer template method definitions ~~~~~$$$**/
//Move assignment
template<typename T>
cudaPointer<T>& cudaPointer<T>::operator=(cudaPointer&& other) noexcept{
  p= other.p; other.p= NULL;
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
  std::cout<<"[cudaPtr]: VecCopy, elts="<<elementN<<'\n';
  allocate();
  if (errorState_==cudaSuccess)
    for(unsigned i=0; i<elementN; i++)
      p[i]= vec[i];
}
template<typename T>
std::vector<T> cudaPointer<T>::getVec(bool release){
  std::vector<T> vec;
  vec.reserve(elementN);
  for(unsigned i=0; i<elementN; i++)
    vec.push_back(std::move(p[i]));
  if(release){
    ownershipReleased_= true;
    deallocate();
  }
  return vec;
}
template<typename T>
void cudaPointer<T>::allocate(){
  if (GPUpresent()){
    static_assert(!std::is_const<T>::value && !std::is_reference<T>::value,
                  "\nCannot allocate cuda managed memory for const-qualified "
                  "or reference types.\nConsult the CUDA C Programming Guide "
                  "section E.2.2.2. __managed__ Qualifier.\n");
    errorState_= cudaMallocManaged((void**)&p, sizeOnDevice, attachment);
  }
  else if (elementN==1)
    p= new T;
  else
    p= new T[elementN];
}
template<typename T>
void cudaPointer<T>::deallocate(){
  if (GPUpresent()) errorState_= cudaFree(p);
  else if (elementN==1)
    delete p;
  else
    delete[] p;
}

#endif	// CUDA_POINTER_H
