// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_QUADRATURE_ADAPTIVE_SIMPSON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_QUADRATURE_ADAPTIVE_SIMPSON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace math
{
namespace quadrature
{

namespace impl
{
template<class F, typename float_t, uint32_t Depth> // F has function __call(x)
struct integrate_helper
{
    static float_t __call(float_t a, float_t b, float_t c, float_t fa, float_t fb, float_t fc, float_t I, float_t eps, NBL_REF_ARG(int) count)
    {
        float_t d = float_t(0.5) * (a + b);
        float_t e = float_t(0.5) * (b + c);
        float_t fd = F::__call(d);
        float_t fe = F::__call(e);

        float_t h = c - a;
        float_t I0 = (float_t(1.0) / float_t(12.0)) * h * (fa + float_t(4.0) * fd + fb);
        float_t I1 = (float_t(1.0) / float_t(12.0)) * h * (fb + float_t(4.0) * fe + fc);
        float_t Ip = I0 + I1;
        count++;

        if (hlsl::abs(Ip - I) < float_t(15.0) * eps)
            return Ip + (float_t(1.0) / float_t(15.0)) * (Ip - I);

        return integrate_helper<F, float_t, Depth-1>::__call(a, d, b, fa, fd, fb, I0, float_t(0.5) * eps, count) +
                integrate_helper<F, float_t, Depth-1>::__call(b, e, c, fb, fe, fc, I1, float_t(0.5) * eps, count);
    }
};

template<class F, typename float_t>
struct integrate_helper<F, float_t, 0>
{
    static float_t __call(float_t a, float_t b, float_t c, float_t fa, float_t fb, float_t fc, float_t I, float_t eps, NBL_REF_ARG(int) count)
    {
        float_t d = float_t(0.5) * (a + b);
        float_t e = float_t(0.5) * (b + c);
        float_t fd = F::__call(d);
        float_t fe = F::__call(e);

        float_t h = c - a;
        float_t I0 = (float_t(1.0) / float_t(12.0)) * h * (fa + float_t(4.0) * fd + fb);
        float_t I1 = (float_t(1.0) / float_t(12.0)) * h * (fb + float_t(4.0) * fe + fc);
        float_t Ip = I0 + I1;
        count++;

        return Ip + (float_t(1.0) / float_t(15.0)) * (Ip - I);
    }
};
}

template<class F, typename float_t, uint32_t Depth=6> // F has function __call(x)
struct AdaptiveSimpson
{
    static float_t __call(float_t x0, float_t x1, float_t eps = 1e-6)
    {
        int count = 0;
        float_t a = x0;
        float_t b = float_t(0.5) * (x0 + x1);
        float_t c = x1;
        float_t fa = F::__call(a);
        float_t fb = F::__call(b);
        float_t fc = F::__call(c);
        float_t I = (c - a) * (float_t(1.0) / float_t(6.0)) * (fa + float_t(4.0) * fb + fc);
        return impl::integrate_helper<F, float_t, Depth>::__call(a, b, c, fa, fb, fc, I, eps, count);
    }
};

}
}
}
}

#endif
