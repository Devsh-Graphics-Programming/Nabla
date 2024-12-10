#ifndef _NBL_BUILTIN_HLSL_PORTABLE_DOT_PRODUCT_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_PORTABLE_DOT_PRODUCT_HLSL_INCLUDED_

// The whole purpose of this file is to enable the creation of partial specializations of the dot function for 
// custom types without introducing circular dependencies. (this file needs to be included in nbl/builtin/hlsl/cpp_compat/intrinsics.h, 
#include <nbl/builtin/hlsl/cpp_compat/basic.h>

namespace nbl
{
namespace hlsl
{

namespace impl
{
template<typename VecT, typename OutputType = void>
struct dot_product_helper
{
	static inline OutputType dot(NBL_CONST_REF_ARG(VecT) lhs, NBL_CONST_REF_ARG(VecT) rhs)
	{
		// TODO:
		//_static_assert(false, "Default template of the dot_product_helper struch should be never called!");
	}
};
}

template<typename VecT, typename OutputType = void>
OutputType dot(NBL_CONST_REF_ARG(VecT) a, NBL_CONST_REF_ARG(VecT) b)
{
	return impl::dot_product_helper<VecT>::dot(a, b);
}

}
}

#endif