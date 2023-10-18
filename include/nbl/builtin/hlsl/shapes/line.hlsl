// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_LINE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_LINE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{
    template<typename float_t>
    struct Line
    {
        using scalar_t = float_t;
        using float_t2 = vector<float_t, 2>;
        using float_t3 = vector<float_t, 3>;
        
        float_t2 P0;
        float_t2 P1;
        
        struct ArcLengthCalculator
        {
            float_t len;

            static ArcLengthCalculator construct(NBL_CONST_REF_ARG(Line<float_t>) segment)
            {
                ArcLengthCalculator ret = { segment.getLength() };
                return ret;
            }

            float_t calcArcLen(float_t t)
            {
                return len * t;
            }
            
            float_t calcArcLenInverse(NBL_CONST_REF_ARG(Line<float_t>) segment, float_t arcLen, float_t accuracyThreshold, float_t hint)
            {
                return arcLen / len;
            }
        };
        
        static Line construct(NBL_CONST_REF_ARG(float_t2) P0, NBL_CONST_REF_ARG(float_t2) P1)
        {            
            Line ret = { P0, P1 };
            return ret;
        }

        float_t2 evaluate(float_t t)
        {
            return t * (P1 - P0) + P0;
        }

        float_t getLength()
        {
            return length(P1 - P0);
        }
        
        NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxCandidates = 1u;
        using Candidates = vector<float_t, MaxCandidates>;

        Candidates getClosestCandidates(NBL_CONST_REF_ARG(float_t2) pos)
        {
            float_t2 p0p1 = P1 - P0;
            float_t2 posp0 = pos - P0;
            float_t t = dot(posp0, p0p1) / dot(p0p1, p0p1);
            return t;
        }
    };
}
}
}

#endif