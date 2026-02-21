#ifndef _NBL_CORE_ALGORITHM_UTILITY_H_INCLUDED_
#define _NBL_CORE_ALGORITHM_UTILITY_H_INCLUDED_

#include <tuple>
#include <variant>

namespace nbl::core
{
// I only thought if I could, not if I should
template<typename... T>
struct type_list
{
};
template<typename TypeList>
struct type_list_size;
template<typename... T>
struct type_list_size<type_list<T...>> : std::integral_constant<size_t,sizeof...(T)> { };
template<typename TypeList>
inline constexpr size_t type_list_size_v = type_list_size<TypeList>::value;

template<template<class...> class ListLikeOutT, template<class> class X, typename ListLike>
struct list_transform
{
	private:
		template<template<class...> class ListLikeInT, typename... T>
		static ListLikeOutT<X<T>...> _impl(const ListLikeInT<T...>&);
		
	public:
		using type = decltype(_impl(std::declval<ListLike>()));
};
template<template<class...> class ListLikeOutT, template<class> class X, typename ListLike>
using list_transform_t = list_transform<ListLikeOutT,X,ListLike>::type;

template<template<class> class X, typename ListLike>
using tuple_transform_t = list_transform_t<std::tuple,X,ListLike>;

template<template<class> class X, typename ListLike>
using variant_transform_t = list_transform_t<std::variant,X,ListLike>;


template<typename Tuple, typename F>
constexpr void for_each_in_tuple(Tuple& t, F&& f) noexcept
{
	std::apply([&f](auto& ...x){(..., f(x));},t);
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

template<typename T, class func_t>
static inline void visit_token_terminated_array(const T* array, const T& endToken, func_t&& func)
{
    if (array)
    for (auto it=array; *it!=endToken && func(*it); it++)
    {
    }
}

}

#endif //__NBL_CORE_ALGORITHM_UTILITY_H_INCLUDED__
