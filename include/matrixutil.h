#ifndef  __IRR_MATRIX_UTIL_H_INCLUDED__
#define __IRR_MATRIX_UTIL_H_INCLUDED__

#include "matrix4SIMD.h"
#include "matrix3x4SIMD.h"

namespace irr
{
namespace core
{


//! TODO: OPTIMIZE THIS, DON'T PROMOTE THE MATRIX IF DON'T HAVE TO
inline matrix4SIMD concatenateBFollowedByA(const matrix4SIMD& _a, const matrix3x4SIMD& _b)
{
    return concatenateBFollowedByA(_a, matrix4SIMD(_b));
}
/*
inline matrix4SIMD concatenateBFollowedByAPrecisely(const matrix4SIMD& _a, const matrix3x4SIMD& _b)
{
    return concatenateBFollowedByAPrecisely(_a, matrix4SIMD(_b));
}
*/

// TODO: Kill this when killing matrix4x3
inline matrix4SIMD concatenateBFollowedByA(const matrix4SIMD& other_a, const matrix4x3& other_b )
{
    matrix3x4SIMD rowBasedMatrixB;
    rowBasedMatrixB.set(other_b);
    return concatenateBFollowedByA(other_a,matrix4SIMD(rowBasedMatrixB));
}


}
}

#endif // __IRR_MATRIX_UTIL_H_INCLUDED__
