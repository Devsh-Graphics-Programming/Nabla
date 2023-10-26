// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_AABB_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_AABB_INCLUDED_

#include <nbl/builtin/hlsl/format/decode.hlsl>


namespace nbl
{
namespace hlsl
{
namespace shapes
{

struct AABB_t
{
    //
    void addPoint(const float32_t3 pt)
    {
        minVx = min(pt, minVx);
        maxVx = max(pt, maxVx);
    }
    //
    float32_t3 getExtent()
    {
        return maxVx - minVx;
    }

    //
    float getVolume()
    {
        const float32_t3 extent = AABB_t::getExtent();
        return extent.x * extent.y * extent.z;
    }

    // returns the corner of the AABB which has the most positive dot product
    float32_t3 getFarthestPointInFront(const float32_t3 plane)
    {
        return lerp(maxVx, minVx, plane<float3(0.f,0.f,0.f));
    }

    float32_t3 minVx;
    float32_t3 maxVx;
};

struct nbl_glsl_shapes_CompressedAABB_t
{
    //
    AABB_t decompress()
    {
        AABB_t retval;
        retval.minVx = decodeRGB18E7S3(minVx18E7S3);
        retval.maxVx = decodeRGB18E7S3(maxVx18E7S3);
        return retval;
    }

    uint32_t2 minVx18E7S3;
    uint32_t2 maxVx18E7S3;
};


}
}
}

#endif
