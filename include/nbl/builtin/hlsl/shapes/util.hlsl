// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_UTIL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_UTIL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace util
{

 template<typename float_t>
 static float64_t2 LineLineIntersection(vector<float_t, 2> P1, vector<float_t, 2> V1, vector<float_t, 2> P2, vector<float_t, 2> V2)
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

//static float32_t2 LineLineIntersection(float32_t2 p1, float32_t2 p2, float32_t2 v1, float32_t2 v2)
//{
//    // Here we're doing part of a matrix calculation because we're interested in only the intersection point not both t values
//    /*
//        float det = v1.y * v2.x - v1.x * v2.y;
//        float2x2 inv = float2x2(v2.y, -v2.x, v1.y, -v1.x) / det;
//        float2 t = mul(inv, p1 - p2);
//        return p2 + mul(v2, t.y);
//    */
//    float denominator = v1.y * v2.x - v1.x * v2.y;
//    float numerator = dot(float32_t2(v2.y, -v2.x), p1 - p2); 

//    float t = numerator / denominator;
//    float32_t2 intersectionPoint = p1 + t * v1;

//    return intersectionPoint;
//}

}
}
}

#endif