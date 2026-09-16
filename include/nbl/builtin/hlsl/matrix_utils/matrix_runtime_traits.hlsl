// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_RUNTIME_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_RUNTIME_TRAITS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/approx/abs_rel.hlsl"
#include "nbl/builtin/hlsl/approx/vector.hlsl"
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
    using row_t = typename matrix_traits<T>::row_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t N = matrix_traits<T>::RowCount;

    static RuntimeTraits<matrix_t> create(const matrix_t m)
    {
        RuntimeTraits<matrix_t> retval;
        retval.invertible = !approx::absRelEqual<scalar_t>(hlsl::determinant(m), scalar_t(0.0), scalar_t(1e-5), scalar_t(1e-5));
        {
            bool orthogonal = true;
            NBL_UNROLL for (uint16_t i = 0; i < N; i++)
            {
                // |cos(theta)| <= 1e-5 allows ~5.73e-4 degrees of deviation from 90deg, independent of the row lengths
                // so uniformly scaled matrices pass too (`quaternion::create` relies on that), see `approx::isPerpendicular`
                const scalar_t cosThetaEpsilon = scalar_t(1e-5);
                orthogonal = orthogonal && approx::isPerpendicular<row_t>(m[i], m[(i+1)%N], cosThetaEpsilon);
            }
            retval.orthogonal = orthogonal;
        }
        {
            const matrix_t m_T = hlsl::transpose(m);
            scalar_t uniformColumnSqNorm = hlsl::dot(m_T[0], m_T[0]);
            NBL_UNROLL for (uint16_t i = 1; i < N; i++)
            {
                if (!approx::absRelEqual<scalar_t>(hlsl::dot(m_T[i], m_T[i]), uniformColumnSqNorm, scalar_t(1e-4), scalar_t(1e-4)))
                {
                    uniformColumnSqNorm = bit_cast<scalar_t>(numeric_limits<scalar_t>::quiet_NaN);
                    break;
                }
            }

            retval.uniformColumnSqNorm = uniformColumnSqNorm;
            retval.orthonormal = retval.orthogonal && approx::absRelEqual<scalar_t>(uniformColumnSqNorm, scalar_t(1.0), scalar_t(1e-5), scalar_t(1e-5));
        }
        return retval;
    }
    
    bool invertible;
    bool orthogonal;
    scalar_t uniformColumnSqNorm;
    bool orthonormal;
};

}
}
}
}

#endif
