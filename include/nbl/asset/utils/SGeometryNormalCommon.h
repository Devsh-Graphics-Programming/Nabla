// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_NORMAL_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_NORMAL_COMMON_H_INCLUDED_


#include "nbl/builtin/hlsl/tgmath.hlsl"


namespace nbl::asset
{

// Shared normal helpers used by loaders and geometry utilities for simple face-normal generation.
class SGeometryNormalCommon
{
    public:
        static_assert(sizeof(hlsl::float32_t3) == sizeof(float[3]));
        static_assert(alignof(hlsl::float32_t3) == alignof(float));

        static inline hlsl::float32_t3 normalizeOrZero(const hlsl::float32_t3& v, const float epsilon = 0.f)
        {
            const float len2 = hlsl::dot(v, v);
            const float epsilon2 = epsilon * epsilon;
            if (len2 <= epsilon2)
                return hlsl::float32_t3(0.f, 0.f, 0.f);
            return hlsl::normalize(v);
        }

        static inline hlsl::float32_t3 computeFaceNormal(const hlsl::float32_t3& a, const hlsl::float32_t3& b, const hlsl::float32_t3& c, const float epsilon = 0.000001f)
        {
            return normalizeOrZero(hlsl::cross(b - a, c - a), epsilon);
        }

        static inline void computeFaceNormal(const float a[3], const float b[3], const float c[3], float normal[3], const float epsilon = 0.000001f)
        {
            *(hlsl::float32_t3*)normal = computeFaceNormal(*(const hlsl::float32_t3*)a, *(const hlsl::float32_t3*)b, *(const hlsl::float32_t3*)c, epsilon);
        }
};

}


#endif
