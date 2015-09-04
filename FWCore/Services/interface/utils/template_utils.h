namespace edm{namespace service{namespace utils{
	
	template<unsigned Idx>
	inline void unwrapToList(void** list) {}

	//! If First arg is pointer, store its value
	template<unsigned Idx, typename First, typename... Tail, typename std::enable_if< 
			std::is_pointer<First>::value, int >::type =0>
	inline void unwrapToList(void** list, First&& arg, Tail&&... tail){
		list[Idx]= &arg;
		unwrapToList<Idx+1>(list, tail...);
	}
	//! If First arg is NOT pointer, store its address
	template<unsigned Idx, typename First, typename... Tail, typename std::enable_if< 
			!std::is_pointer<First>::value, int >::type =0>
	inline void unwrapToList(void** list, First&& arg, Tail&&... tail){
		list[Idx]= &arg;
		unwrapToList<Idx+1>(list, tail...);
	}

	//! Wrapper to enable operating on each arg of a parameter pack
	template<typename... Args>
	inline void operateOnParamPacks(Args&&... args){}

}}}	//namespace edm::service::utils
