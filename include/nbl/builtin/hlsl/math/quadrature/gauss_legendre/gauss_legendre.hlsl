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
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/impl.hlsl>

// rename NBL_GLSL_FEATURE_SHADER_FLOAT64 to NBL_LIMIT_FLOAT64 after merge
#if defined(NBL_GLSL_FEATURE_SHADER_FLOAT64) || !defined(__HLSL_VERSION)
#define float_t float64_t
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/impl.hlsl>
#endif
#undef float_t


#define DEF_GAUSS_LEGENDRE_VALS(N)                                                                              \
template <typename float_t>                                                                                     \
struct GaussLegendreValues<N, float_t>                                                                          \
{                                                                                                               \
    static float_t xi(uint32_t idx) { return nbl::hlsl::math::quadrature::impl::xi_##N[idx]; }                  \
    static float_t wi(uint32_t idx) { return nbl::hlsl::math::quadrature::impl::wi_##N[idx]; }                  \
}

        DEF_GAUSS_LEGENDRE_VALS(2);
        DEF_GAUSS_LEGENDRE_VALS(3);
        DEF_GAUSS_LEGENDRE_VALS(4);
        DEF_GAUSS_LEGENDRE_VALS(5);
        DEF_GAUSS_LEGENDRE_VALS(6);
        DEF_GAUSS_LEGENDRE_VALS(7);
        DEF_GAUSS_LEGENDRE_VALS(8);
        DEF_GAUSS_LEGENDRE_VALS(9);
        DEF_GAUSS_LEGENDRE_VALS(10);
        DEF_GAUSS_LEGENDRE_VALS(11);
        DEF_GAUSS_LEGENDRE_VALS(12);
        DEF_GAUSS_LEGENDRE_VALS(13);
        DEF_GAUSS_LEGENDRE_VALS(14);
        DEF_GAUSS_LEGENDRE_VALS(15);
} // quadrature
} // math
} // hlsl
} // nbl

#endif
