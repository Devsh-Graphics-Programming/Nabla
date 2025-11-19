// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_ANGLE_ADDING_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_ANGLE_ADDING_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/complex.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{

template <typename T>
struct sincos_accumulator
{
    using this_t = sincos_accumulator<T>;

    static this_t create()
    {
        this_t retval;
        retval.runningSum = complex_t<T>::create(T(1.0), T(0.0));
        return retval;
    }

    static this_t create(T cosA)
    {
        this_t retval;
        retval.runningSum = complex_t<T>::create(cosA, T(0));
        return retval;
    }

    void addCosine(T cosA, T biasA)
    {
        const T bias = biasA + runningSum.imag();
        const T a = cosA;
        const T b = runningSum.real();
        const bool reverse = abs<T>(min<T>(a, b)) > max<T>(a, b);
        const T c = a * b - sqrt<T>((T(1.0) - a * a) * (T(1.0) - b * b));

        runningSum.real(ieee754::flipSign<T>(c, reverse));
        runningSum.imag(hlsl::mix(bias, bias + numbers::pi<T>, reverse));
    }
    void addCosine(T cosA)
    {
        addCosine(cosA, T(0.0));
    }

    T getSumofArccos()
    {
        return acos<T>(runningSum.real()) + runningSum.imag();
    }

    static T getArccosSumofABC_minus_PI(T cosA, T cosB, T cosC, T sinA, T sinB, T sinC)
    {
        const bool AltminusB = cosA < (-cosB);
        const T cosSumAB = cosA * cosB - sinA * sinB;
        const bool ABltminusC = cosSumAB < (-cosC);
        const bool ABltC = cosSumAB < cosC;
        // apply triple angle formula
        const T absArccosSumABC = acos<T>(clamp<T>(cosSumAB * cosC - (cosA * sinB + sinA * cosB) * sinC, T(-1.0), T(1.0)));
        return ((AltminusB ? ABltC : ABltminusC) ? (-absArccosSumABC) : absArccosSumABC) + ((AltminusB || ABltminusC) ? numbers::pi<T> : (-numbers::pi<T>));
    }

    complex_t<T> runningSum;
};

}
}
}

#endif
