#ifndef CUDA_POINTER_H
#define CUDA_POINTER_H

#include <cuda_runtime_api.h>
#include <type_traits>
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace edm{namespace service{
class CudaService;
}}
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
	cudaPointer(edm::Service<edm::service::CudaService>& service, unsigned elementN=1, Attachment flag=host);
	//Move constructor && assignment
	cudaPointer(cudaPointer&& other) noexcept: p(other.p), attachment(other.attachment),
			sizeOnDevice(other.sizeOnDevice), status(other.status),
			freeFlag(other.freeFlag), service_(other.service) { other.p= NULL; }
	cudaPointer& operator=(cudaPointer&& other) noexcept;
	//Delete copy constructor and assignment
	cudaPointer(const cudaPointer&) =delete;
	cudaPointer& operator=(const cudaPointer&) =delete;
	cudaPointer(edm::Service<edm::service::CudaService>& service, const std::vector<T>& vec, Attachment flag=host);
	//p must retain ownership
	~cudaPointer(){
		if (!freeFlag) deallocate();
	}
	//Only call default if on a new thread
	void attachStream(cudaStream_t stream= cudaStreamPerThread);
	cudaError_t getStatus() const { return status; }
	
	std::vector<T> getVec(bool release= false);

	//public!
	T* p;
private:
	void allocate();
	void deallocate();
	size_t sizeOnDevice;
	Attachment attachment;
	cudaError_t status;
	bool freeFlag= false;
	unsigned elementN;
	edm::Service<edm::service::CudaService>& service_;
};

//cudaConst??


#endif	// CUDA_POINTER_H
