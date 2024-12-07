#ifndef _NBL_BUILTIN_HLSL_DOT_PRODUCTS_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_DOT_PRODUCTS_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/array_accessors.hlsl>

namespace nbl
{
namespace hlsl
{
template<typename T>
typename vector_traits<T>::ScalarType dot(T lhs, T rhs)
{
	using ScalarType = typename vector_traits<T>::ScalarType;

	static array_get<T, ScalarType> getter;
	ScalarType retval = getter(lhs, 0) * getter(rhs, 0);

	static const uint32_t ArrayDim = sizeof(T) / sizeof(ScalarType);
	for (uint32_t i = 1; i < ArrayDim; ++i)
		retval = retval + getter(lhs, i) * getter(rhs, i);

	return retval;
}
}
}

#endif