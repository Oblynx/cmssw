#ifndef CUDA_POINTER_H
#define CUDA_POINTER_H

#include <iostream>

#include <cuda_runtime_api.h>
#include <type_traits>
#include <vector>

#include "GPU_presence_static.h"

//!< @class Useful for type identification
class cudaPtrBase {};
//!< @class <unique_ptr>-like managed cuda smart pointer. T: non-const, non-ref
template<typename T>
class cudaPointer: cudaPtrBase{
public:
	enum Attachment{
		global= cudaMemAttachGlobal,
		host= cudaMemAttachHost,
		single= cudaMemAttachSingle
	};
	//flag default= host: see CUDA C Programming Guide J.2.2.6
	// TODO: What if malloc fails???
	cudaPointer(unsigned elementN=1, Attachment flag=host):
      sizeOnDevice(elementN*sizeof(T)), attachment(flag), freeFlag(false),
      elementN(elementN) { allocate(); }
	//Move constructor && assignment
	cudaPointer(cudaPointer&& other) noexcept: p(other.p), 
			sizeOnDevice(other.sizeOnDevice), attachment(other.attachment),
      errorState_(other.errorState_), freeFlag(other.freeFlag),
      elementN(other.elementN) { other.p= NULL; }
	cudaPointer& operator=(cudaPointer&& other) noexcept;
	//Delete copy constructor and assignment
	cudaPointer(const cudaPointer&) =delete;
	cudaPointer& operator=(const cudaPointer&) =delete;
	cudaPointer(const std::vector<T>& vec, Attachment flag=host);
	//p must retain ownership
	~cudaPointer(){
		if (!freeFlag) deallocate();
	}
	//Only call default if on a new thread
	void attachStream(cudaStream_t stream= cudaStreamPerThread){
	  attachment= single;
	  if (GPUpresent()) cudaStreamAttachMemAsync(stream, p, 0, attachment);
	}
	cudaError_t getErrorState() const { return errorState_; }
	std::vector<T> getVec(bool release= false);
	bool GPUpresent() { return cuda::GPUPresenceStatic::getStatus(this); }

	//public!
	T* p;
private:
	void allocate();
	void deallocate();
	size_t sizeOnDevice;
	Attachment attachment;
	cudaError_t errorState_;
	bool freeFlag= false;
	unsigned elementN;
};

template<typename CPU>
struct KernelWrapper{
  KernelWrapper(const void* kernel, CPU fallback): kernel(kernel), fallback(fallback),
      fallbackPresent(true) {}
  KernelWrapper(const void* kernel): kernel(kernel), fallback(nullptr),
      fallbackPresent(false) {}
  const void* kernel;
  const CPU fallback;
  const bool fallbackPresent;
};/*
template<>
struct KernelWrapper<void>{
  KernelWrapper(const void* kernel): kernel(kernel), fallbackPresent(false) {}
  const void* kernel;
  const bool fallbackPresent;
};*/
template<typename CPU>
KernelWrapper<CPU> make_wrapper(const void* kernel, CPU fallback){
  return KernelWrapper<CPU>(kernel, fallback);
}
KernelWrapper<void*> make_wrapper(const void* kernel){
  return KernelWrapper<void*>(kernel);
}
//cudaConst??

/**$$$~~~~~ CudaPointer method definitions ~~~~~$$$**/
//Move assignment
template<typename T>
cudaPointer<T>& cudaPointer<T>::operator=(cudaPointer&& other) noexcept{
  p= other.p; other.p= NULL;
  sizeOnDevice= other.sizeOnDevice;
  attachment= other.attachment;
  errorState_= other.errorState_;
  freeFlag= other.freeFlag;
  elementN= other.elementN;
  return *this;
}
//Constructor vector copy
template<typename T>
cudaPointer<T>::cudaPointer(const std::vector<T>& vec, Attachment flag):
    attachment(flag), freeFlag(false), elementN(vec.size())
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
    freeFlag= true;
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
}}} //namespace edm::service::utils


#endif	// CUDA_POINTER_H
