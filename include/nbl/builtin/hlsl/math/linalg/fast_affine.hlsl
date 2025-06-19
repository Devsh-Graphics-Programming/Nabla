// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_LINALG_FAST_AFFINE_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_LINALG_FAST_AFFINE_INCLUDED_


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
// TODO: move to macros
#ifdef __HLSL_VERSION
#define NBL_UNROLL [[unroll]]
#else
#define NBL_UNROLL
#endif

// Multiply matrices as-if extended to be filled with identity elements
template<typename T, int N, int M, int P, int Q> 
matrix<T,N,M> promoted_mul(NBL_CONST_REF_ARG(matrix<T,N,P>) lhs, NBL_CONST_REF_ARG(matrix<T,Q,M>) rhs)
{
    matrix<T,N,M> retval;
    // NxM = NxR RxM
    // out[i][j] == dot(row[i],col[j])
    // out[i][j] == lhs[i][0]*col[j][0]+...+lhs[i][3]*col[j][3]
    // col[a][b] == (rhs^T)[b][a]
    // out[i][j] == lhs[i][0]*rhs[0][j]+...+lhs[i][3]*rhs[3][j]
    // out[i] == lhs[i][0]*rhs[0]+...+lhs[i][3]*rhs[3]
    NBL_UNROLL for (uint32_t i=0; i<N; i++)
    {
        vector<T,M> acc = rhs[i];
        // multiply if not outside of `lhs` matrix
        // otherwise the diagonal element is just unity
        if (i<P)
            acc *= lhs[i][i];
        // other elements are 0 if outside the LHS matrix
        NBL_UNROLL for (uint32_t j=0; j<P; j++)
        if (j!=i)
        {
            // inside the RHS matrix
            if (j<Q)
                acc += rhs[j]*lhs[i][j];
            else // outside we have an implicit e_j valued row
                acc[j] += lhs[i][j];
        }
        retval[i] = acc;
    }
    return retval;
}

// Multiply matrix and vector as-if extended to be filled with 1 in diagonal for matrix and last for vector
template<typename T, int N, int M, int P> 
vector<T,N> promoted_mul(NBL_CONST_REF_ARG(matrix<T,N,M>) lhs, const vector<T,P> v)
{
    vector<T,N> retval;
    // Nx1 = NxM Mx1
    {
        matrix<T,M,1> rhs;
        // one can safely discard elements of `v[i]` where `i<P && i>=M`, because to contribute `lhs` would need to have `M>=P`
        NBL_UNROLL for (uint32_t i=0; i<M; i++)
        {
            if (i<P)
                rhs[i] = v[i];
            else
                rhs[i] = i!=(M-1) ? T(0):T(1);
        }
        matrix<T,N,1> tmp = promoted_mul<T,N,1,M,M>(lhs,rhs);
        NBL_UNROLL for (uint32_t i=0; i<N; i++)
            retval[i] = tmp[i];
    }
    return retval;
}
#undef NBL_UNROLL

// useful for fast computation of a Normal Matrix
template<typename T, int N>
struct cofactors_base;

template<typename T>
struct cofactors_base<T,3>
{
    using matrix_t = matrix<T,3,3>;
    using vector_t = vector<T,3>;

    static inline cofactors_base<T,3> create(NBL_CONST_REF_ARG(matrix_t) val)
    {
        cofactors_base<T,3> retval;

        retval.transposed = matrix_t(
            hlsl::cross<vector_t>(val[1],val[2]),
            hlsl::cross<vector_t>(val[2],val[0]),
            hlsl::cross<vector_t>(val[0],val[1])
        );

        return retval;
    }

    //
    inline matrix_t get() NBL_CONST_MEMBER_FUNC
    {
        return hlsl::transpose<matrix_t>(transposed);
    }
    
    //
    inline vector_t normalTransform(const vector_t n) NBL_CONST_MEMBER_FUNC
    {
        const vector_t tmp = hlsl::mul<matrix_t,vector_t>(transposed,n);
        return hlsl::normalize<vector_t>(tmp);
    }

    matrix_t transposed;
};

// variant that cares about flipped/mirrored transforms
template<typename T, int N>
struct cofactors
{
    using pseudo_base_t = cofactors_base<T,N>;
    using matrix_t = typename pseudo_base_t::matrix_t;
    using vector_t = typename pseudo_base_t::vector_t;
    using mask_t = unsigned_integer_of_size_t<sizeof(T)>;

    static inline cofactors<T,3> create(NBL_CONST_REF_ARG(matrix_t) val)
    {
        cofactors<T,3> retval;
        retval.composed = pseudo_base_t::create(val);

        const T det = hlsl::dot<vector_t>(val[0],retval.composed.transposed[0]);

        const mask_t SignBit = 1;
        SignBit = SignBit<<(sizeof(mask_t)*8-1);
        retval.signFlipMask = bit_cast<mask_t>(det) & SignBit;

        return retval;
    }
    
    //
    inline vector_t normalTransform(const vector_t n) NBL_CONST_MEMBER_FUNC
    {
        const vector_t tmp = hlsl::mul<matrix_t,vector_t>(composed.transposed,n);
        const T rcpLen = hlsl::rsqrt<T>(hlsl::dot<vector_t>(tmp,tmp));
        return tmp*bit_cast<T>(bit_cast<mask_t>(rcpLen)^determinantSignMask);
    }

    cofactors_base<T,N> composed;
    mask_t determinantSignMask;
};

//
template<typename Mat3x4 NBL_FUNC_REQUIRES(is_matrix_v<Mat3x4>) // TODO: allow any matrix type AND our emulated ones
Mat3x4 pseudoInverse3x4(NBL_CONST_REF_ARG(Mat3x4) tform, NBL_CONST_REF_ARG(matrix<scalar_type_t<Mat3x4>,3,3>) sub3x3Inv)
{
    Mat3x4 retval;
    retval[0] = sub3x3Inv[0];
    retval[1] = sub3x3Inv[1];
    retval[2] = sub3x3Inv[2];
    retval[3] = -hlsl::mul(sub3x3Inv,tform[3]);
    return retval;
}
template<typename Mat3x4 NBL_FUNC_REQUIRES(is_matrix_v<Mat3x4>) // TODO: allow any matrix type AND our emulated ones
Mat3x4 pseudoInverse3x4(NBL_CONST_REF_ARG(Mat3x4) tform)
{
    return pseudoInverse3x4(tform,inverse(matrix<scalar_type_t<Mat3x4>,3,3>(tform)));
}


}
}
}
}
#endif