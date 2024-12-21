// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_QUADRATURE_GAUSS_LEGENDRE_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_QUADRATURE_GAUSS_LEGENDRE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>


namespace nbl
{
namespace hlsl
{
namespace math
{
namespace quadrature
{

template<uint16_t Order, typename float_t>
struct GaussLegendreValues;

template<int Order, typename float_t, class IntegrandFunc>
struct GaussLegendreIntegration
{
    static float_t calculateIntegral(NBL_CONST_REF_ARG(IntegrandFunc) func, float_t start, float_t end)
    {
        float_t integral = 0.0;
        for (uint32_t i = 0u; i < Order; ++i)
        {
            const float_t xi = GaussLegendreValues<Order, float_t>::xi(i) * ((end - start) / 2.0) + ((end + start) / 2.0);
            integral += GaussLegendreValues<Order, float_t>::wi(i) * func(xi);
        }
        return ((end - start) / 2.0) * integral;
    }
};

#define float_t float32_t
#define TYPED_NUMBER(N) NBL_CONCATENATE(N, f) // to add f after floating point numbers and avoid casting warnings and emitting ShaderFloat64 Caps
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/impl.hlsl>
#undef TYPED_NUMBER
#undef float_t

#define float_t float64_t
#define TYPED_NUMBER(N) N
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/impl.hlsl>
#undef TYPED_NUMBER
#undef float_t

} // quadrature
} // math
} // hlsl
} // nbl

#endif
