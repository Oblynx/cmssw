#include <cuda_runtime_api.h>

// template<typename T>
// class cudaPointer{
// public:
// 	//flag: cudaMemAttachGlobal | cudaMemAttachHost
// 	cudaPointer(unsigned flag=cudaMemAttachGlobal): p(new T){
// 		cudaMallocManaged(&p.get(), sizeof(T), flag);
// 	}
// 	//p must retain ownership until here!
// 	~cudaPointer(){
// 		cudaFree(p.get());
// 	}
// 	T* passKernel() const{
// 		return p.get();
// 	}

// 	//public!
// 	std::unique_ptr<T> p;
// };

class cudaPtrBase {};

template<typename T>
class cudaPointer: cudaPtrBase{
public:
	//flag: cudaMemAttachGlobal | cudaMemAttachHost
	cudaPointer(int elementN, unsigned flag=cudaMemAttachGlobal): p(new T), attachment(flag){
		cudaMallocManaged(&p, elementN*sizeof(T), flag);
	}
	//p must retain ownership until here!
	~cudaPointer(){
		cudaFree(p);
	}
	//Only call default if on a new thread
	void attachStream(cudaStream_t stream= cudaStreamPerThread){
		attachment= cudaMemAttachSingle;
		cudaStreamAttachMemAsync(stream, p, 0, attachment);
	}

	operator T*(){ return p; }
	//public!
	T* p;
private:
	unsigned attachment;
};
