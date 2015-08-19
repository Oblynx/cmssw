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

template<typename T>
class cudaPointer{
public:
	//flag: cudaMemAttachGlobal | cudaMemAttachHost
	cudaPointer(int elementN, unsigned flag=cudaMemAttachGlobal): p(new T){
		cudaMallocManaged(&p, elementN*sizeof(T), flag);
	}
	//p must retain ownership until here!
	~cudaPointer(){
		cudaFree(p);
	}

	//public!
	T* p;
};
