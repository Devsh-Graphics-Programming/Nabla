// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_LINE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_LINE_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{

template<typename float_t, uint32_t Dim NBL_PRIMARY_REQUIRES(concepts::FloatingPointScalar<float_t> && (Dim == 2 || Dim == 3))
struct Line
{
    using scalar_t = float_t;
    using vec_t = vector<float_t, Dim>;
    using mat_t = matrix<float_t, Dim, Dim>;
        
    vec_t P0;
    vec_t P1;
        
    struct ArcLengthCalculator
    {
        float_t len;

        static ArcLengthCalculator construct(NBL_CONST_REF_ARG(Line<float_t, Dim>) segment)
        {
            ArcLengthCalculator ret = { segment.getLength() };
            return ret;
        }

        float_t calcArcLen(float_t t)
        {
            return len * t;
        }
            
        float_t calcArcLenInverse(NBL_CONST_REF_ARG(Line<float_t, Dim>) segment, float_t min, float_t max, float_t arcLen, float_t accuracyThreshold, float_t hint)
        {
            return arcLen / len;
        }
    };
        
    static Line construct(NBL_CONST_REF_ARG(vec_t) P0, NBL_CONST_REF_ARG(vec_t) P1)
    {            
        Line ret;
        ret.P0 = P0;
        ret.P1 = P1;
        return ret;
    }

    vec_t evaluate(float_t t) NBL_CONST_MEMBER_FUNC
    {
        return t * (P1 - P0) + P0;
    }

    float_t getLength() NBL_CONST_MEMBER_FUNC
    {
        return length(P1 - P0);
    }
        
    NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxCandidates = 1u;
    using Candidates = vector<float_t, MaxCandidates>;

    Candidates getClosestCandidates(NBL_CONST_REF_ARG(vec_t) pos) NBL_CONST_MEMBER_FUNC
    {
        Candidates ret;
        vec_t p0p1 = P1 - P0;
        vec_t posp0 = pos - P0;
        ret[0] = dot(posp0, p0p1) / dot(p0p1, p0p1);
        return ret;
    }

    // TODO: this supports only 2D lines for now
    matrix<float_t, 2, 2> getLocalCoordinateSpace(float_t t) NBL_CONST_MEMBER_FUNC
    {
        vec_t d = normalize(P1 - P0);
        return matrix<float_t, 2, 2>(d.x, d.y, -d.y, d.x);
    }
};

template<typename float_t>
using Line2D = Line<float_t, 2>;
template<typename float_t>
using Line3D = Line<float_t, 3>;

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