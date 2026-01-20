// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_LINALG_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_LINALG_TRANSFORM_INCLUDED_

#include <nbl/builtin/hlsl/mpl.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>

namespace nbl
{
namespace hlsl
{
namespace math
{
namespace linalg
{

/// Builds a rotation 3 * 3 matrix created from an axis vector and an angle.
///
/// @param angle Rotation angle expressed in radians.
/// @param axis Rotation axis, must be normalized.
///
/// @tparam T A floating-point scalar type
template <typename T>
matrix<T, 3, 3> rotation_mat(T angle, const vector<T, 3> axis)
{
    const T a = angle;
    const T c = cos(a);
    const T s = sin(a);

    vector<T, 3> temp = hlsl::promote<vector<T, 3> >((T(1.0) - c) * axis);

    matrix<T, 3, 3> rotation;
    rotation[0][0] = c + temp[0] * axis[0];
    rotation[0][1] = temp[1] * axis[0] - s * axis[2];
    rotation[0][2] = temp[2] * axis[0] + s * axis[1];

    rotation[1][0] = temp[0] * axis[1] + s * axis[2];
    rotation[1][1] = c + temp[1] * axis[1];
    rotation[1][2] = temp[2] * axis[1] - s * axis[0];

    rotation[2][0] = temp[0] * axis[2] - s * axis[1];
    rotation[2][1] = temp[1] * axis[2] + s * axis[0];
    rotation[2][2] = c + temp[2] * axis[2];

    return rotation;
}

namespace impl
{
template<uint16_t MOut, uint16_t MIn, typename T>
struct zero_expand_helper
{
    static vector<T, MOut> __call(const vector<T, MIn> inVec)
    {
        return vector<T, MOut>(inVec, vector<T, MOut - MIn>(0));
    }
};
template<uint16_t M, typename T>
struct zero_expand_helper<M,M,T>
{
    static vector<T, M> __call(const vector<T, M> inVec)
    {
        return inVec;
    }
};
}

template<uint16_t MOut, uint16_t MIn, typename T NBL_FUNC_REQUIRES(MOut >= MIn)
vector<T, MOut> zero_expand(vector<T, MIn> inVec)
{
    return impl::zero_expand_helper<MOut, MIn, T>::__call(inVec);
}

template <uint16_t NOut, uint16_t MOut, uint16_t NIn, uint16_t MIn, typename T NBL_FUNC_REQUIRES(NOut >= NIn && MOut >= MIn)
matrix<T, NOut, MOut> promote_affine(const matrix<T, NIn, MIn> inMatrix)
{
    matrix<T, NOut, MOut> retval;

    using out_row_t = hlsl::vector<T, MOut>;

    NBL_UNROLL for (uint32_t row_i = 0; row_i < NIn; row_i++)
    {
        retval[row_i] = zero_expand<MOut, MIn>(inMatrix[row_i]);
    }
    NBL_UNROLL for (uint32_t row_i = NIn; row_i < NOut; row_i++)
    {
        retval[row_i] = promote<out_row_t>(0.0);
        if (row_i < MOut)
            retval[row_i][row_i] = T(1.0);
    }
    return retval;
}

// /Arek: glm:: for normalize till dot product is fixed (ambiguity with glm namespace + linker issues)
template<typename T>
inline matrix<T, 3, 4> lhLookAt(
    const vector<T, 3>& position,
    const vector<T, 3>& target,
    const vector<T, 3>& upVector)
{
    const vector<T, 3> zaxis = hlsl::normalize(target - position);
    const vector<T, 3> xaxis = hlsl::normalize(hlsl::cross(upVector, zaxis));
    const vector<T, 3> yaxis = hlsl::cross(zaxis, xaxis);

    matrix<T, 3, 4> r;
    r[0] = vector<T, 4>(xaxis, -hlsl::dot(xaxis, position));
    r[1] = vector<T, 4>(yaxis, -hlsl::dot(yaxis, position));
    r[2] = vector<T, 4>(zaxis, -hlsl::dot(zaxis, position));

    return r;
}

template<typename T>
inline matrix<T, 3, 4> rhLookAt(
    const vector<T, 3>& position,
    const vector<T, 3>& target,
    const vector<T, 3>& upVector)
{
    const vector<T, 3> zaxis = hlsl::normalize(position - target);
    const vector<T, 3> xaxis = hlsl::normalize(hlsl::cross(upVector, zaxis));
    const vector<T, 3> yaxis = hlsl::cross(zaxis, xaxis);

    matrix<T, 3, 4> r;
    r[0] = vector<T, 4>(xaxis, -hlsl::dot(xaxis, position));
    r[1] = vector<T, 4>(yaxis, -hlsl::dot(yaxis, position));
    r[2] = vector<T, 4>(zaxis, -hlsl::dot(zaxis, position));

    return r;
}

template<typename T, int16_t N, int16_t M, int16_t VecN>
inline void setTranslation(NBL_REF_ARG(matrix<T, N, M>) outMat, NBL_CONST_REF_ARG(vector<T, VecN>) translation)
{
    // TODO: not sure if it will be compatible with hlsl
    static_assert(M > 0 && N > 0);
    static_assert(M >= VecN);

    NBL_CONSTEXPR int16_t indexOfTheLastRowComponent = M - 1;

    for(int i = 0; i < VecN; ++i)
        outMat[i][indexOfTheLastRowComponent] = translation[i];
}

}
}
}
}
#endif
