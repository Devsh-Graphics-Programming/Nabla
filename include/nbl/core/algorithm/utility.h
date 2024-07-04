#ifndef _NBL_CORE_ALGORITHM_UTILITY_H_INCLUDED_
#define _NBL_CORE_ALGORITHM_UTILITY_H_INCLUDED_

#include <tuple>
#include <variant>

namespace nbl::core
{

// I only thought if I could, not if I should
template<template<class> class X, typename TupleOrVariant>
struct tuple_transform
{
	private:
		template<typename... T>
		static std::tuple<X<T>...> _impl(const std::tuple<T...>&);
		template<typename... T>
		static std::tuple<X<T>...> _impl(const std::variant<T...>&);

	public:
		using type = decltype(_impl(std::declval<TupleOrVariant>()));
};
template<template<class> class X, typename TupleOrVariant>
using tuple_transform_t = tuple_transform<X,TupleOrVariant>::type;

template<template<class> class X, typename TupleOrVariant>
struct variant_transform
{
	private:
		template<typename... T>
		static std::variant<X<T>...> _impl(const std::tuple<T...>&);
		template<typename... T>
		static std::variant<X<T>...> _impl(const std::variant<T...>&);

	public:
		using type = decltype(_impl(std::declval<TupleOrVariant>()));
};
template<template<class> class X, typename TupleOrVariant>
using variant_transform_t = variant_transform<X, TupleOrVariant>::type;

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
