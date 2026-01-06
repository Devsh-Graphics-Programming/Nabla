#ifndef _NBL_BUILTIN_HLSL_TESTING_RELATIVE_APPROX_COMPARE_INCLUDED_
#define _NBL_BUILTIN_HLSL_TESTING_RELATIVE_APPROX_COMPARE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>

namespace nbl 
{
namespace hlsl
{
namespace testing
{
namespace impl
{

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct RelativeApproxCompareHelper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeScalar<FloatingPoint>)
struct RelativeApproxCompareHelper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeScalar<FloatingPoint>) >
{
    static bool __call(NBL_CONST_REF_ARG(FloatingPoint) lhs, NBL_CONST_REF_ARG(FloatingPoint) rhs, const float64_t maxAllowedDifference)
    {
        const bool bothAreNaN = nbl::hlsl::isnan(lhs) && nbl::hlsl::isnan(rhs);
        const bool bothAreInf = nbl::hlsl::isinf(lhs) && nbl::hlsl::isinf(rhs);
        const bool bothHaveSameSign = nbl::hlsl::ieee754::extractSign(lhs) == nbl::hlsl::ieee754::extractSign(rhs);
        const bool lhsIsSubnormalOrZero = ieee754::isSubnormal(lhs) || ieee754::isZero(lhs);
        const bool rhsIsSubnormalOrZero = ieee754::isSubnormal(rhs) || ieee754::isZero(rhs);

        if (bothAreNaN)
            return true;
        if (bothAreInf && bothHaveSameSign)
            return true;
        if (lhsIsSubnormalOrZero && rhsIsSubnormalOrZero)
            return true;
        if (!lhsIsSubnormalOrZero && rhsIsSubnormalOrZero)
            return false;
        if (lhsIsSubnormalOrZero && !rhsIsSubnormalOrZero)
            return false;

        return hlsl::max(hlsl::abs(lhs / rhs), hlsl::abs(rhs / lhs)) <= 1.f + maxAllowedDifference;
    }
};

template<typename FloatingPointVector>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<FloatingPointVector>)
struct RelativeApproxCompareHelper<FloatingPointVector NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<FloatingPointVector>) >
{
    static bool __call(NBL_CONST_REF_ARG(FloatingPointVector) lhs, NBL_CONST_REF_ARG(FloatingPointVector) rhs, const float64_t maxAllowedDifference)
    {
        using traits = nbl::hlsl::vector_traits<FloatingPointVector>;
        for (uint32_t i = 0; i < traits::Dimension; ++i)
        {
            if (!RelativeApproxCompareHelper<typename traits::scalar_type>::__call(lhs[i], rhs[i], maxAllowedDifference))
                return false;
        }

        return true;
    }
};

template<typename FloatingPointMatrix>
NBL_PARTIAL_REQ_TOP(concepts::Matricial<FloatingPointMatrix> && concepts::FloatingPointLikeScalar<typename nbl::hlsl::matrix_traits<FloatingPointMatrix>::scalar_type>)
struct RelativeApproxCompareHelper<FloatingPointMatrix NBL_PARTIAL_REQ_BOT(concepts::Matricial<FloatingPointMatrix> && concepts::FloatingPointLikeScalar<typename nbl::hlsl::matrix_traits<FloatingPointMatrix>::scalar_type>) >
{
    static bool __call(NBL_CONST_REF_ARG(FloatingPointMatrix) lhs, NBL_CONST_REF_ARG(FloatingPointMatrix) rhs, const float64_t maxAllowedDifference)
    {
        using traits = nbl::hlsl::matrix_traits<FloatingPointMatrix>;
        for (uint32_t i = 0; i < traits::RowCount; ++i)
        {
            if (!RelativeApproxCompareHelper<typename traits::row_type>::__call(lhs[i], rhs[i], maxAllowedDifference))
                return false;
        }

        return true;
    }
};

}

template<typename T>
bool relativeApproxCompare(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, const float64_t maxAllowedDifference)
{
	return impl::RelativeApproxCompareHelper<T>::__call(lhs, rhs, maxAllowedDifference);
}

}
}
}

#endif