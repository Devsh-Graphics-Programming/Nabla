#ifndef _NBL_BUILTIN_HLSL_TESTING_ORIENTATION_COMPARE_INCLUDED_
#define _NBL_BUILTIN_HLSL_TESTING_ORIENTATION_COMPARE_INCLUDED_

#include <nbl/builtin/hlsl/testing/relative_approx_compare.hlsl>

namespace nbl 
{
namespace hlsl
{
namespace testing
{
namespace impl
{

template<typename FloatingPointVector NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<FloatingPointVector>)
struct OrientationCompareHelper
{
    static bool __call(NBL_CONST_REF_ARG(FloatingPointVector) lhs, NBL_CONST_REF_ARG(FloatingPointVector) rhs, const float64_t maxAllowedDifference)
    {
        using traits = nbl::hlsl::vector_traits<FloatingPointVector>;
        using scalar_t = typename traits::scalar_type;

        const scalar_t dotLR = hlsl::dot(lhs, rhs);
        if (dotLR < scalar_t(0.0))
            return false;

        const scalar_t scale = hlsl::sqrt(hlsl::dot(lhs,lhs) * hlsl::dot(rhs,rhs));
        return relativeApproxCompare<scalar_t>(dotLR, scale, maxAllowedDifference);
    }
};

}

template<typename T>
bool orientationCompare(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, const float64_t maxAllowedDifference)
{
	return impl::OrientationCompareHelper<T>::__call(lhs, rhs, maxAllowedDifference);
}

}
}
}

#endif