#include "matrix4SIMD.h"
#include "matrix3x4SIMD.h""

namespace irr { namespace core
{

namespace impl
{
matrix4SIMD toMat4(const matrix3x4SIMD& _mtx)
{
    matrix4SIMD m4;
    m4[0] = _mtx[0];
    m4[1] = _mtx[1];
    m4[2] = _mtx[2];
    m4[3] = vectorSIMDf(0.f, 0.f, 0.f, 1.f);

    return m4;
}
}

inline matrix4SIMD concatenateBFollowedByA(const matrix4SIMD& _a, const matrix3x4SIMD& _b)
{
    return concatenateBFollowedByA(_a, impl::toMat4(_b));
}
inline matrix4SIMD concatenateBFollowedByAPrecisely(const matrix4SIMD& _a, const matrix3x4SIMD& _b)
{
    return concatenateBFollowedByAPrecisely(_a, impl::toMat4(_b));
}

inline matrix4SIMD concatenateBFollowedByA(const matrix3x4SIMD& _a, const matrix4SIMD& _b)
{
    return concatenateBFollowedByA(impl::toMat4(_a), _b);
}
inline matrix4SIMD concatenateBFollowedByAPrecisely(const matrix3x4SIMD& _a, const matrix4SIMD& _b)
{
    concatenateBFollowedByAPrecisely(impl::toMat4(_a), _b);
}

}}