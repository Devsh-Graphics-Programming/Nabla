// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_FUNCTIONAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_FUNCTIONAL_INCLUDED_

namespace nbl
{
namespace hlsl
{
#ifndef __HLSL_VERSION // CPP

template<class T> using bit_and = std::bit_and;
template<class T> using bit_or = std::bit_or;
template<class T> using bit_xor = std::bit_xor;
template<class T> using plus = std::plus;
template<class T> using multiplies = std::multiplies;
template<class T> using greater = std::greater;
template<class T> using less = std::less;
template<class T> using greater_equal = std::greater_equal;
template<class T> using less_equal = std::less_equal;

#else // HLSL
    
template<typename T>
struct bit_and
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs & rhs;
    }
    
    static T identity()
    {
        return ~0;
    }
};

template<typename T>
struct bit_or
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs | rhs;
    }
    
    static T identity()
    {
        return 0;
    }
};

template<typename T>
struct bit_xor
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs ^ rhs;
    }
    
    static T identity()
    {
        return 0;
    }
};

template<typename T>
struct plus
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs + rhs;
    }
    
    static T identity()
    {
        return 0;
    }
};

template<typename T>
struct multiplies
{
    T operator()(const T lhs, const T rhs)
    {
        return lhs * rhs;
    }
    
    static T identity()
    {
        return 1;
    }
};

template<typename T>
struct greater
{
    bool operator()(const T lhs, const T rhs)
    {
        return lhs > rhs;
    }
};

template<typename T>
struct less
{
    bool operator()(const T lhs, const T rhs)
    {
        return lhs < rhs;
    }
};

template<typename T>
struct greater_equal
{
    bool operator()(const T lhs, const T rhs)
    {
        return lhs >= rhs;
    }
};

template<typename T>
struct less_equal
{
    bool operator()(const T lhs, const T rhs)
    {
        return lhs <= rhs;
    }
};

#endif

// Min and Max are outside of the HLSL/C++ directives because we want these to be available in both contexts
// TODO: implement as mix(rhs<lhs,lhs,rhs)

template<typename T>
struct minimum
{
    T operator()(const T lhs, const T rhs)
    {
        return (rhs < lhs) ? rhs : lhs;
    }

    static T identity()
    {
        return ~0;
    }
};

template<typename T>
struct maximum
{
    T operator()(const T lhs, const T rhs)
    {
        return (lhs < rhs) ? rhs : lhs;
    }

    static T identity()
    {
        return 0;
    }
};

}
}

#endif