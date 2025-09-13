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
    using float_t2x2 = matrix<float_t, 2, 2>;
        
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
            
        float_t calcArcLenInverse(NBL_CONST_REF_ARG(Line<float_t>) segment, float_t min, float_t max, float_t arcLen, float_t accuracyThreshold, float_t hint)
        {
            return arcLen / len;
        }
    };
        
    static Line construct(NBL_CONST_REF_ARG(float_t2) P0, NBL_CONST_REF_ARG(float_t2) P1)
    {            
        Line ret = { P0, P1 };
        return ret;
    }

    float_t2 evaluate(float_t t) NBL_CONST_MEMBER_FUNC
    {
        return t * (P1 - P0) + P0;
    }

    float_t getLength() NBL_CONST_MEMBER_FUNC
    {
        return length(P1 - P0);
    }
        
    NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxCandidates = 1u;
    using Candidates = vector<float_t, MaxCandidates>;

    Candidates getClosestCandidates(NBL_CONST_REF_ARG(float_t2) pos) NBL_CONST_MEMBER_FUNC
    {
        Candidates ret;
        float_t2 p0p1 = P1 - P0;
        float_t2 posp0 = pos - P0;
        ret[0] = dot(posp0, p0p1) / dot(p0p1, p0p1);
        return ret;
    }

    float_t2x2 getLocalCoordinateSpace(float_t t) NBL_CONST_MEMBER_FUNC
    {
        float_t2 d = normalize(P1 - P0);
        return float_t2x2(d.x, d.y, -d.y, d.x);
    }
};

namespace util
{
template<typename float_t>
static vector<float_t, 2> LineLineIntersection(NBL_CONST_REF_ARG(vector<float_t, 2>) P1, NBL_CONST_REF_ARG(vector<float_t, 2>) V1, NBL_CONST_REF_ARG(vector<float_t, 2>) P2, NBL_CONST_REF_ARG(vector<float_t, 2>) V2)
{
    typedef vector<float_t, 2> float_t2;

    float_t denominator = V1.y * V2.x - V1.x * V2.y;
    vector<float_t, 2> diff = P1 - P2;
    float_t numerator = dot(float_t2(V2.y, -V2.x), float_t2(diff.x, diff.y));

    if (abs(denominator) < 1e-15 && abs(numerator) < 1e-15)
    {
        // are parallel and the same
        return (P1 + P2) / 2.0;
    }

    float_t t = numerator / denominator;
    float_t2 intersectionPoint = P1 + t * V1;
    return intersectionPoint;
}
}

}
}
}

#endif