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

        const scalar_t dotLR = hlsl::abs(hlsl::dot(lhs, rhs));
        const scalar_t dotLL = hlsl::dot(lhs,lhs);
        const scalar_t dotRR = hlsl::dot(rhs,rhs);
        if (dotLL < numeric_limits<scalar_t>::min || dotRR < numeric_limits<scalar_t>::min)
            return false;

        const scalar_t scale = hlsl::sqrt(dotLL * dotRR);
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