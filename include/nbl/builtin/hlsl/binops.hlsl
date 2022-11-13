// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MACROS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MACROS_INCLUDED_
// TODO (PentaKon): Should this file be renamed to binops.hlsl ?
// TODO (PentaKon): Should we move isNPoT from algorithm.hlsl to this file ?
namespace nbl
{
namespace hlsl
{
// TODO (PentaKon): NBL_GLSL_EVAL? Keep as macro or don't use at all?
namespace binops
{
template<typename T>
struct and
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs&rhs;
    }
};

template<typename T>
struct or
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs|rhs;
    }
};

template<typename T>
struct xor
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs^rhs;
    }
};

template<typename T>
struct mul
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs*rhs;
    }
};

template<typename T>
struct min
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs>rhs ? rhs : lhs;
    }
};

template<typename T>
struct max
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs>rhs ? lhs : rhs;
    }
};

}
}
}
#endif