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
        assert(sinA >= T(0.0));
        this_t retval;
        retval.runningSum = complex_t<T>::create(cosA, sinA);
        retval.wraparound = T(0.0);
        return retval;
    }

    // This utility expects that all input angles being added are [0,PI], i.e. runningSum.y and sinA are always positive
    // We make use of sine and cosine angle sum formulas to accumulate a running sum of angles
    // and any "overflow" (when sum goes over PI) is stored in wraparound
    // This way, we can accumulate the sines and cosines of a set of angles, and only have to do one acos() call to retrieve the final sum
    void addAngle(T cosA, T sinA)
    {
        const T cosB = runningSum.real();
        const T sinB = runningSum.imag();
        // TODO: prove if we infer overflow from sign of `d` instead
        const bool overflow = abs<T>(min<T>(cosA, cosB)) > max<T>(cosA, cosB);
        const T c = cosA * cosB - sinA * sinB;
        const T d = sinA * cosB + cosA * sinB;

        // TODO: figure out if we can skip flipping signs until `getSumOfArccos()`, this would mean that on odd overflows the formula is
        // const T c = sinA * sinB - cosA * cosB;
        // const T d = - sinA * cosB - cosA * sinB;
        // but both signs get inverted so the `abs()` of both terms gets preserved, so the equation can stay as is.
        // And in `getSumOfArccos()` we could do either of:
        // return acos<T>((intWraparound<<signBitOffset<T>)^runningSum.real()) + intWraparound * numbers::pi<T>;
        // return (intWraparound+(intWraparound&0x1u)) * numbers::pi<T> + (intWraparound<<signBitOffset<T>)^acos<T>(runningSum.real());
        runningSum.real(ieee754::flipSign<T>(c, overflow));
        runningSum.imag(ieee754::flipSign<T>(d, overflow));

        if (overflow)
            wraparound += numbers::pi<T>;
    }
    void addCosine(T cosA)
    {
        addAngle(cosA, sqrt<T>(T(1.0) - cosA * cosA));
    }

    // TODO: rename to `getSumOfArccos`
    T getSumofArccos()
    {
        return acos<T>(runningSum.real()) + wraparound;
    }

    complex_t<T> runningSum;
    T wraparound;    // counts in pi (half revolutions)
};

}
}
}

#endif
