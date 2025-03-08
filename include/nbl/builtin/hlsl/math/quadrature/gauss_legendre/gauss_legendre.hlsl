// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_QUADRATURE_GAUSS_LEGENDRE_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_QUADRATURE_GAUSS_LEGENDRE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/emulated/float64_t.hlsl>

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
        float_t integral = _static_cast<float_t>(0ull);
        for (uint32_t i = 0u; i < Order; ++i)
        {
            const float_t xi = GaussLegendreValues<Order, float_t>::xi(i) * ((end - start) / 2.0f) + ((end + start) / 2.0f);
            integral = integral + GaussLegendreValues<Order, float_t>::wi(i) * func(xi);
        }

        return ((end - start) / 2.0) * integral;
    }
};

#define float_t float32_t
#define float_t_namespace impl_float32_t
#define TYPED_NUMBER(N) NBL_CONCATENATE(N, f) // to add f after floating point numbers and avoid casting warnings and emitting ShaderFloat64 Caps
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/impl.hlsl>
#undef TYPED_NUMBER
#undef float_t_namespace
#undef float_t

#define float_t float64_t
#define float_t_namespace impl_float64_t
#define TYPED_NUMBER(N) N
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/impl.hlsl>
#undef TYPED_NUMBER
#undef float_t_namespace
#undef float_t

#define float_t emulated_float64_t<true, true>
#define float_t_namespace impl_emulated_float64_t_true_true
#define TYPED_NUMBER(N) emulated_float64_t<true, true>::create(N)
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/impl.hlsl>
#undef TYPED_NUMBER
#undef float_t_namespace
#undef float_t

#define float_t emulated_float64_t<false, false>
#define float_t_namespace impl_emulated_float64_t_false_false
#define TYPED_NUMBER(N) emulated_float64_t<false, false>::create(N)
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/impl.hlsl>
#undef TYPED_NUMBER
#undef float_t_namespace
#undef float_t

#define float_t emulated_float64_t<false, true>
#define float_t_namespace impl_emulated_float64_t_false_true
#define TYPED_NUMBER(N) emulated_float64_t<false, true>::create(N)
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/impl.hlsl>
#undef TYPED_NUMBER
#undef float_t_namespace
#undef float_t

#define float_t emulated_float64_t<true, false>
#define float_t_namespace impl_emulated_float64_t_true_false
#define TYPED_NUMBER(N) emulated_float64_t<true, false>::create(N)
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/impl.hlsl>
#undef TYPED_NUMBER
#undef float_t_namespace
#undef float_t

} // quadrature
} // math
} // hlsl
} // nbl

#endif
