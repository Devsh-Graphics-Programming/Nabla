// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_ANGLE_ADDING_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_ANGLE_ADDING_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"
#include "nbl/builtin/hlsl/concepts/vector.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"
#include "nbl/builtin/hlsl/ieee754.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{

namespace impl
{
struct sincos_accumulator
{
    using this_t = sincos_accumulator;

    static this_t create()
    {
        this_t retval;
        retval.tmp0 = 0;
        retval.tmp1 = 0;
        retval.tmp2 = 0;
        retval.tmp3 = 0;
        retval.tmp4 = 0;
        retval.tmp5 = 0;
        return retval;
    }

    static this_t create(float cosA, float cosB, float cosC, float sinA, float sinB, float sinC)
    {
        this_t retval;
        retval.tmp0 = cosA;
        retval.tmp1 = cosB;
        retval.tmp2 = cosC;
        retval.tmp3 = sinA;
        retval.tmp4 = sinB;
        retval.tmp5 = sinC;
        return retval;
    }

    float getArccosSumofABC_minus_PI()
    {
        const bool AltminusB = tmp0 < (-tmp1);
        const float cosSumAB = tmp0 * tmp1 - tmp3 * tmp4;
        const bool ABltminusC = cosSumAB < (-tmp2);
        const bool ABltC = cosSumAB < tmp2;
        // apply triple angle formula
        const float absArccosSumABC = acos<float>(clamp<float>(cosSumAB * tmp2 - (tmp0 * tmp4 + tmp3 * tmp1) * tmp5, -1.f, 1.f));
        return ((AltminusB ? ABltC : ABltminusC) ? (-absArccosSumABC) : absArccosSumABC) + ((AltminusB || ABltminusC) ? numbers::pi<float> : (-numbers::pi<float>));
    }

    static void combineCosForSumOfAcos(float cosA, float cosB, float biasA, float biasB, NBL_REF_ARG(float) out0, NBL_REF_ARG(float) out1)
    {
        const float bias = biasA + biasB;
        const float a = cosA;
        const float b = cosB;
        const bool reverse = abs<float>(min<float>(a, b)) > max<float>(a, b);
        const float c = a * b - sqrt<float>((1.0f - a * a) * (1.0f - b * b));

        if (reverse)
        {
            out0 = -c;
            out1 = bias + numbers::pi<float>;
        }
        else
        {
            out0 = c;
            out1 = bias;
        }
    }

    float tmp0;
    float tmp1;
    float tmp2;
    float tmp3;
    float tmp4;
    float tmp5;
};
}

float getArccosSumofABC_minus_PI(float cosA, float cosB, float cosC, float sinA, float sinB, float sinC)
{
    impl::sincos_accumulator acc = impl::sincos_accumulator::create(cosA, cosB, cosC, sinA, sinB, sinC);
    return acc.getArccosSumofABC_minus_PI();
}

void combineCosForSumOfAcos(float cosA, float cosB, float biasA, float biasB, NBL_REF_ARG(float) out0, NBL_REF_ARG(float) out1)
{
    impl::sincos_accumulator acc = impl::sincos_accumulator::create();
    impl::sincos_accumulator::combineCosForSumOfAcos(cosA, cosB, biasA, biasB, acc.tmp0, acc.tmp1);
    out0 = acc.tmp0;
    out1 = acc.tmp1;
}

// returns acos(a) + acos(b)
float getSumofArccosAB(float cosA, float cosB)
{
    impl::sincos_accumulator acc = impl::sincos_accumulator::create();
    impl::sincos_accumulator::combineCosForSumOfAcos(cosA, cosB, 0.0f, 0.0f, acc.tmp0, acc.tmp1);
    return acos<float>(acc.tmp0) + acc.tmp1;
}

// returns acos(a) + acos(b) + acos(c) + acos(d)
float getSumofArccosABCD(float cosA, float cosB, float cosC, float cosD)
{
    impl::sincos_accumulator acc = impl::sincos_accumulator::create();
    impl::sincos_accumulator::combineCosForSumOfAcos(cosA, cosB, 0.0f, 0.0f, acc.tmp0, acc.tmp1);
    impl::sincos_accumulator::combineCosForSumOfAcos(cosC, cosD, 0.0f, 0.0f, acc.tmp2, acc.tmp3);
    impl::sincos_accumulator::combineCosForSumOfAcos(acc.tmp0, acc.tmp2, acc.tmp1, acc.tmp3, acc.tmp4, acc.tmp5);
    return acos<float>(acc.tmp4) + acc.tmp5;
}

}
}
}

#endif
