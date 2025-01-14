// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_RANDOM_PCG_HLSL_INCLUDED_
#define _NBL_BUILTIN_GLSL_RANDOM_PCG_HLSL_INCLUDED_

namespace nbl
{
namespace hlsl
{

namespace impl
{

uint32_t pcg_hash(uint32_t v)
{
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

template<typename T>
struct PCGHash;

template<>
struct PCGHash<uint32_t>
{
    static uint32_t __call(uint32_t v)
    {
        return pcg_hash(v);
    }
};

template<uint16_t N>
struct PCGHash<vector<uint32_t, N>>
{
    static vector<uint32_t, N> __call(vector<uint32_t, N> v)
    {
        vector<uint32_t, N> retval;
        for (int i = 0; i < N; i++)
            retval[i] = pcg_hash(v[i]);
        return retval;
    }
};
}

template<typename T>
T pcg32(T v)
{
    return impl::PCGHash<T>::__call(v);
}

uint32_t2 pcg32x2(uint32_t v)
{
    return impl::PCGHash<uint32_t2>::__call(uint32_t2(v, v+1));
}

}
}

#endif
