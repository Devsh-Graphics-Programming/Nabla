#ifndef _NBL_BUILTIN_HLSL_APPROX_ORIENTATION_INCLUDED_
#define _NBL_BUILTIN_HLSL_APPROX_ORIENTATION_INCLUDED_

#include <nbl/builtin/hlsl/approx/abs_rel.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace approx
{
namespace impl
{

template<typename FloatingPointVector NBL_PRIMARY_REQUIRES(concepts::FloatingPointVectorial<FloatingPointVector>)
struct OrientationEqualHelper
{
    static bool __call(NBL_CONST_REF_ARG(FloatingPointVector) lhs, NBL_CONST_REF_ARG(FloatingPointVector) rhs, const typename vector_traits<FloatingPointVector>::scalar_type maxRelativeDifference)
    {
        using scalar_t = typename vector_traits<FloatingPointVector>::scalar_type;

        const scalar_t dotLR = hlsl::abs(hlsl::dot(lhs, rhs));
        const scalar_t dotLL = hlsl::dot(lhs,lhs);
        const scalar_t dotRR = hlsl::dot(rhs,rhs);
        if (dotLL < numeric_limits<scalar_t>::min || dotRR < numeric_limits<scalar_t>::min)
            return false;

        const scalar_t scale = hlsl::sqrt(dotLL * dotRR);
        return absRelEqual<scalar_t>(dotLR, scale, scalar_t(0), maxRelativeDifference);
    }
};

}

// true if `lhs` and `rhs` point along the same line (either direction), ignoring their lengths
template<typename T>
bool orientationEqual(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, const typename vector_traits<T>::scalar_type maxRelativeDifference)
{
	return impl::OrientationEqualHelper<T>::__call(lhs, rhs, maxRelativeDifference);
}

}
}
}

#endif
