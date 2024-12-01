#ifndef _NBL_BUILTIN_HLSL_DOT_PRODUCTS_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_DOT_PRODUCTS_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/scalar_of.hlsl>
#include <nbl/builtin/hlsl/array_accessors.hlsl>

namespace nbl
{
namespace hlsl
{
template<typename T>
scalar_of_t<T> dot(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
{
	static array_get<T, scalar_of_t<T> > getter;
	scalar_type_t<T> retval = getter(lhs, 0) * getter(rhs, 0);
	static const uint16_t ArrayDim = sizeof(T) / sizeof(scalar_type_t<T>);
	for (uint16_t i = 1; i < ArrayDim; ++i)
		retval = retval + getter(lhs, i) * getter(rhs, i);

	return retval;
}
}
}

#endif