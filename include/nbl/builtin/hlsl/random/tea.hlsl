// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_RANDOM_TEA_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RANDOM_TEA_HLSL_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace random
{

struct Tea
{
    static Tea create()
	{
        Tea tea;
        return tea;
    }
	
    static uint32_t2 __call(const uint32_t stream, const uint32_t sequence, const uint32_t roundCount)
	{
        uint32_t sum = 0;
        uint32_t v0 = stream;
        uint32_t v1 = sequence;
        for (uint32_t n = 0; n < roundCount; n++)
        {
            sum += 0x9e3779b9;
            v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1 >> 5) + 0xc8013ea4);
            v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0 >> 5) + 0x7e95761e);
        }

		return uint32_t2(v0, v1);
    }
	
    uint32_t2 operator()(const uint32_t stream, const uint32_t sequence, const uint32_t roundCount) NBL_CONST_MEMBER_FUNC
	{
        return __call(stream,sequence,roundCount);
    }
};

}
}
}
#endif
