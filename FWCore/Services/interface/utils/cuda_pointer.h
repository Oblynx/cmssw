#ifndef CUDA_POINTER_H
#define CUDA_POINTER_H

#include <cuda_runtime_api.h>
#include <type_traits>

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
	cudaPointer(int elementN=1, Attachment flag=host): p(new T),
									attachment(flag), sizeOnDevice(elementN*sizeof(T)){
		static_assert(!std::is_const<T>::value && !std::is_reference<T>::value,
		              "\nCannot allocate cuda managed memory for const-qualified "
		              "or reference types.\nConsult the CUDA C Programming Guide "
		              "section E.2.2.2. __managed__ Qualifier.\n");
		status= cudaMallocManaged(&p, elementN*sizeof(T), flag);
	}
	//Delete copy constructor and assignment
	cudaPointer(const cudaPointer&) =delete;
	cudaPointer& operator=(const cudaPointer&) =delete;

	//Move constructor && assignment
	cudaPointer(cudaPointer&& other) noexcept: p(other.p), sizeOnDevice(other.sizeOnDevice),
				attachment(other.attachment), status(other.status) { other.p= NULL; }
	cudaPointer& operator=(cudaPointer&& other) noexcept{
		p= other.p; other.p= NULL;
		sizeOnDevice= other.sizeOnDevice;
		attachment= other.attachment;
		status= other.status;
	}
	//p must retain ownership
	~cudaPointer(){
		cudaFree(p);
	}
	//Only call default if on a new thread
	void attachStream(cudaStream_t stream= cudaStreamPerThread){
		attachment= single;
		cudaStreamAttachMemAsync(stream, p, 0, attachment);
	}
	cudaError_t getStatus() { return status; }

	//public!
	T* p;
private:
	size_t sizeOnDevice;
	Attachment attachment;
	cudaError_t status;
};

//cudaConst??

namespace edm{
namespace service{
namespace utils{

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

}}}	//namespace edm::service::utils

#endif	// CUDA_POINTER_H


// //!< @class Useful for type identification
// class cudaPtrBase {};
// //!< @class <unique_ptr>-like managed cuda smart pointer
// template<typename T>
// class cudaPointer: cudaPtrBase{
// public:
// 	//flag: cudaMemAttachGlobal | cudaMemAttachHost
// 	cudaPointer(int elementN, unsigned flag=cudaMemAttachGlobal): p(new T), attachment(flag){
// 		cudaMallocManaged(&p.get(), elementN*sizeof(T), flag);
// 	}
// 	//Delete copy constructor and assignment
// 	cudaPointer(const cudaPointer&) =delete;
// 	cudaPointer& operator=(const cudaPointer&) =delete;

// 	//p must retain ownership
// 	~cudaPointer(){
// 		cudaFree(p.get());
// 	}
// 	//Only call default if on a new thread
// 	void attachStream(cudaStream_t stream= cudaStreamPerThread){
// 		attachment= cudaMemAttachSingle;
// 		cudaStreamAttachMemAsync(stream, p.get(), 0, attachment);
// 	}

// 	// To remove!!!
// 	operator T*(){ return p.get(); }
// 	//public!
// 	std::unique_ptr<T> p;
// private:
// 	unsigned attachment;
// };
