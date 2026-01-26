#ifndef _NBL_BUILTIN_HLSL_TESTING_VECTOR_LENGTH_COMPARE_INCLUDED_
#define _NBL_BUILTIN_HLSL_TESTING_VECTOR_LENGTH_COMPARE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>

namespace nbl 
{
namespace hlsl
{
namespace testing
{
namespace impl
{

template<typename FloatingPointVector NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<FloatingPointVector>)
struct LengthCompareHelper
{
    static bool __call(NBL_CONST_REF_ARG(FloatingPointVector) lhs, NBL_CONST_REF_ARG(FloatingPointVector) rhs, const float64_t maxAbsoluteDifference, const float64_t maxRelativeDifference)
    {
        using traits = nbl::hlsl::vector_traits<FloatingPointVector>;
        using scalar_t = typename traits::scalar_type;

        const scalar_t dotLL = hlsl::dot(lhs,lhs);
        const scalar_t dotRR = hlsl::dot(rhs,rhs);
        const scalar_t diff = hlsl::abs(dotLL-dotRR);
        const scalar_t sc = hlsl::max(dotLL,dotRR);
        return diff <= maxAbsoluteDifference || diff <= maxRelativeDifference*sc;
    }
};

}

template<typename T>
bool vectorLengthCompare(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, const float64_t maxAbsoluteDifference, const float64_t maxRelativeDifference)
{
	return impl::LengthCompareHelper<T>::__call(lhs, rhs, maxAbsoluteDifference, maxRelativeDifference);
}

}
}
}

#endif