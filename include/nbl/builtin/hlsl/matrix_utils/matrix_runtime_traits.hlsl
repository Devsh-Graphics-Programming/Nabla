// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_RUNTIME_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_RUNTIME_TRAITS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/testing/relative_approx_compare.hlsl"
#include "nbl/builtin/hlsl/concepts/matrix.hlsl"
#include "nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{
namespace linalg
{

template<typename T NBL_PRIMARY_REQUIRES(concepts::Matricial<T> && matrix_traits<T>::Square)
struct RuntimeTraits
{
    using matrix_t = T;
    using scalar_t = typename matrix_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t N = matrix_traits<T>::RowCount;

    static RuntimeTraits<matrix_t> create(const matrix_t m)
    {
        RuntimeTraits<matrix_t> retval;
        retval.invertible = !testing::relativeApproxCompare(hlsl::determinant(m), scalar_t(0.0), 1e-5);
        {
            bool orthogonal = true;
            NBL_UNROLL for (uint16_t i = 0; i < N; i++)
                orthogonal = orthogonal && testing::relativeApproxCompare(hlsl::dot(m[i], m[(i+1)%N]), scalar_t(0.0), 1e-4);
            retval.orthogonal = orthogonal;
        }
        {
            const matrix_t m_T = hlsl::transpose(m);
            scalar_t uniformScaleSq = hlsl::dot(m_T[0], m_T[0]);
            NBL_UNROLL for (uint16_t i = 1; i < N; i++)
            {
                if (!testing::relativeApproxCompare(hlsl::dot(m_T[i], m_T[i]), uniformScaleSq, 1e-4))
                {
                    uniformScaleSq = bit_cast<scalar_t>(numeric_limits<scalar_t>::quiet_NaN);
                    break;
                }
            }

            retval.uniformScaleSq = uniformScaleSq;
            retval.orthonormal = retval.orthogonal && testing::relativeApproxCompare(uniformScaleSq, scalar_t(1.0), 1e-5);
        }
        return retval;
    }
    
    bool invertible;
    bool orthogonal;
    scalar_t uniformScaleSq; // TODO: rename to `uniformColumnSqNorm`
    bool orthonormal;
};

}
}
}
}

#endif
