namespace edm{
namespace service{
namespace utils{




template<int...> struct Seq {};
template<int N, int... S> struct GenSeq: GenSeq<N-1, N-1, S...> {};
template<int... S> struct GenSeq<0, S...> {
  typedef Seq<S...> type;
};
template<typename... Args> struct KernelArgs{
  KernelArgs(Args&&... args): args_(std::forward<Args>(args)...) {}
  std::tuple<Args...> args_;
};
template<typename... Args> struct ManagedArgs: KernelArgs<Args...>{
  ManagedArgs(Args&&... args): KernelArgs<Args...>(std::forward<Args>(args)...)
  {}

};
template<typename... Args> struct NonManagedArgs: KernelArgs<Args...>{
  NonManagedArgs(Args&&... args): KernelArgs<Args...>(std::forward<Args>(args)...)
  {}
};
/*
// visit_tuple
template<typename Callable, typename Head>
Callable visit_tuple(Callable f, Head&& aTuple)
{
   const size_t size = std::tuple_size<typename std::remove_reference<Head>::type>::value-1;
   visit_tuple_ws<size>::visit(f, aTuple);
   return f;
}
 
// support struct to iterate over the tuple
template<size_t size>
struct visit_tuple_ws
{
   template<typename Callable, typename Head>
   static void visit(Callable& f, Head& aTuple)
   {
      visit_tuple_ws<size-1>::visit(f, aTuple);
      f(std::get<size>(aTuple));
   }
};
 
// stop recursion here
template<>
struct visit_tuple_ws<0u>
{
   template<typename Callable, typename Head>
   static void visit(Callable& f, Head& aTuple)
   {
      f(std::get<0>(aTuple));
   }
};
*/
}	//namespace utils
}	//namespace service
}	//namespace edm
