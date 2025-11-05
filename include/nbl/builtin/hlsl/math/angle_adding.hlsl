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

    static this_t create(T cosA)
    {
        return create(cosA, sqrt<T>(T(1.0) - cosA * cosA));
    }

    static this_t create(T cosA, T sinA)
    {
        this_t retval;
        retval.runningSum = complex_t<T>::create(cosA, sinA);
        // retval.runningSum.real(cosA);
        // retval.runningSum.imag(sinA);
        retval.wraparound = 0u;
        return retval;
    }

    void addCosine(T cosA, T sinA)
    {
        const T a = cosA;
        const T cosB = runningSum.real();
        const T sinB = runningSum.imag();
        const bool reverse = abs<T>(min<T>(a, cosB)) > max<T>(a, cosB);
        const T c = a * cosB - sinA * sinB;
        const T d = sinA * cosB + a * sinB;

        runningSum.real(ieee754::flipSign<T>(c, reverse));
        runningSum.imag(ieee754::flipSign<T>(d, reverse));

        wraparound += hlsl::mix(0u, 1u, reverse);
    }
    void addCosine(T cosA)
    {
        addCosine(cosA, sqrt<T>(T(1.0) - cosA * cosA));
    }

    T getSumofArccos()
    {
        return acos<T>(runningSum.real()) + wraparound * numbers::pi<T>;
    }

    complex_t<T> runningSum;
    uint16_t wraparound;
};

}
}
}

#endif
