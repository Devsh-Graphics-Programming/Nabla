#ifndef _NBL_BUILTIN_HLSL_TESTING_APPROX_COMPARE_INCLUDED_
#define _NBL_BUILTIN_HLSL_TESTING_APPROX_COMPARE_INCLUDED_

#include <nbl/builtin/hlsl/testing/relative_approx_compare.hlsl>

namespace nbl
{
namespace hlsl
{
namespace testing
{
namespace impl
{

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct AbsoluteAndRelativeApproxCompareHelper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeScalar<FloatingPoint>)
struct AbsoluteAndRelativeApproxCompareHelper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeScalar<FloatingPoint>) >
{
    static bool __call(NBL_CONST_REF_ARG(FloatingPoint) lhs, NBL_CONST_REF_ARG(FloatingPoint) rhs, const float64_t maxAbsoluteDifference, const float64_t maxRelativeDifference)
    {
        // Absolute check first: catches small-magnitude values where relative comparison breaks down
        if (hlsl::abs(float64_t(lhs) - float64_t(rhs)) <= maxAbsoluteDifference)
            return true;

        // Fall back to relative comparison for larger values
        return RelativeApproxCompareHelper<FloatingPoint>::__call(lhs, rhs, maxRelativeDifference);
    }
};

template<typename FloatingPointVector>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<FloatingPointVector>)
struct AbsoluteAndRelativeApproxCompareHelper<FloatingPointVector NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<FloatingPointVector>) >
{
    static bool __call(NBL_CONST_REF_ARG(FloatingPointVector) lhs, NBL_CONST_REF_ARG(FloatingPointVector) rhs, const float64_t maxAbsoluteDifference, const float64_t maxRelativeDifference)
    {
        using traits = nbl::hlsl::vector_traits<FloatingPointVector>;
        for (uint32_t i = 0; i < traits::Dimension; ++i)
        {
            if (!AbsoluteAndRelativeApproxCompareHelper<typename traits::scalar_type>::__call(lhs[i], rhs[i], maxAbsoluteDifference, maxRelativeDifference))
                return false;
        }

        return true;
    }
};

template<typename FloatingPointMatrix>
NBL_PARTIAL_REQ_TOP(concepts::Matricial<FloatingPointMatrix> && concepts::FloatingPointLikeScalar<typename nbl::hlsl::matrix_traits<FloatingPointMatrix>::scalar_type>)
struct AbsoluteAndRelativeApproxCompareHelper<FloatingPointMatrix NBL_PARTIAL_REQ_BOT(concepts::Matricial<FloatingPointMatrix> && concepts::FloatingPointLikeScalar<typename nbl::hlsl::matrix_traits<FloatingPointMatrix>::scalar_type>) >
{
    static bool __call(NBL_CONST_REF_ARG(FloatingPointMatrix) lhs, NBL_CONST_REF_ARG(FloatingPointMatrix) rhs, const float64_t maxAbsoluteDifference, const float64_t maxRelativeDifference)
    {
        using traits = nbl::hlsl::matrix_traits<FloatingPointMatrix>;
        for (uint32_t i = 0; i < traits::RowCount; ++i)
        {
            if (!AbsoluteAndRelativeApproxCompareHelper<typename traits::row_type>::__call(lhs[i], rhs[i], maxAbsoluteDifference, maxRelativeDifference))
                return false;
        }

        return true;
    }
};

}

// Composite comparator that builds on top of relativeApproxCompare.
// Checks absolute difference first (handles small-magnitude values where
// relative comparison breaks down), then falls back to relative comparison.
template<typename T>
bool approxCompare(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, const float64_t maxAbsoluteDifference, const float64_t maxRelativeDifference)
{
	return impl::AbsoluteAndRelativeApproxCompareHelper<T>::__call(lhs, rhs, maxAbsoluteDifference, maxRelativeDifference);
}

}
}
}

#endif
