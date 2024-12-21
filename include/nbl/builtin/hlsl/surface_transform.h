// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SURFACE_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_SURFACE_TRANSFORM_INCLUDED_
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

namespace nbl
{
namespace hlsl
{
namespace SurfaceTransform
{

enum class FLAG_BITS : uint16_t
{
    NONE = 0x0,
    IDENTITY_BIT = 0x0001,
    ROTATE_90_BIT = 0x0002,
    ROTATE_180_BIT = 0x0004,
    ROTATE_270_BIT = 0x0008,
    HORIZONTAL_MIRROR_BIT = 0x0010,
    HORIZONTAL_MIRROR_ROTATE_90_BIT = 0x0020,
    HORIZONTAL_MIRROR_ROTATE_180_BIT = 0x0040,
    HORIZONTAL_MIRROR_ROTATE_270_BIT = 0x0080,
    INHERIT_BIT = 0x0100,
    ALL_BITS = 0x01FF
};

// define everything else in terms of this
inline float32_t2x2 transformMatrix(const FLAG_BITS transform)
{
    switch (transform)
    {
        case FLAG_BITS::IDENTITY_BIT:
            return float32_t2x2( 1.f, 0.f,
                                 0.f, 1.f);
        case FLAG_BITS::ROTATE_90_BIT:
            return float32_t2x2( 0.f, 1.f,
                                -1.f, 0.f);
        case FLAG_BITS::ROTATE_180_BIT:
            return float32_t2x2(-1.f, 0.f,
                                 0.f,-1.f);
        case FLAG_BITS::ROTATE_270_BIT:
            return float32_t2x2( 0.f,-1.f,
                                 1.f, 0.f);
        case FLAG_BITS::HORIZONTAL_MIRROR_BIT:
            return float32_t2x2(-1.f, 0.f,
                                 0.f, 1.f);
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_90_BIT:
            return float32_t2x2( 0.f, 1.f,
                                 1.f, 0.f);
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_180_BIT:
            return float32_t2x2( 1.f, 0.f,
                                 0.f,-1.f);
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_270_BIT:
            return float32_t2x2( 0.f,-1.f,
                                -1.f, 0.f);
        default:
            break;
    }
    const float _nan = numeric_limits<float>::signaling_NaN;
    return float32_t2x2(_nan,_nan,_nan,_nan);
}

//! [width,height] might switch to [height, width] in orientations such as 90°CW
//! Usecase: Find out how big the viewport has to be after or before a tranform is applied
inline uint16_t2 transformedExtents(const FLAG_BITS transform, const uint16_t2 screenSize)
{
    switch (transform)
    {
        case FLAG_BITS::IDENTITY_BIT:
        case FLAG_BITS::HORIZONTAL_MIRROR_BIT:
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_180_BIT:
        case FLAG_BITS::ROTATE_180_BIT:
            return screenSize;
        case FLAG_BITS::ROTATE_90_BIT:
        case FLAG_BITS::ROTATE_270_BIT:
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_90_BIT:
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_270_BIT:
            return screenSize.yx;
        default:
            break;
    }
    return uint16_t2(0,0);
}

inline float transformedAspectRatio(const FLAG_BITS transform, const uint16_t2 screenSize)
{
    const uint16_t2 newExtents = transformedExtents(transform,screenSize);
    return float(newExtents[1])/float(newExtents[0]);
}

//! Use this function to apply the INVERSE of swapchain tranformation to the screenspace coordinate `coord` 
//! For example when the device orientation is 90°CW then this transforms the point 90°CCW.
//! Usecase = [Gather]:
//!   Applications such as raytracing in shaders where you would want to generate rays from screen space coordinates. 
//! Warnings: 
//! - You don't need to consider this using in your raytracing shaders if you apply the forward transformation to your projection matrix.
//! - Be aware that almost always you'd want to do a single transform in your rendering pipeline.
inline uint16_t2 applyInverseToScreenSpaceCoordinate(const FLAG_BITS transform, const uint16_t2 coord, const uint16_t2 screenSize)
{
    // TODO: use inverse(transformMatrix(transform)) somehow
    const uint16_t2 lastTexel = screenSize - uint16_t2(1,1);
    switch (transform)
    {
        case FLAG_BITS::IDENTITY_BIT:
            return coord;
        case FLAG_BITS::ROTATE_90_BIT:
            return uint16_t2(lastTexel.y - coord.y, coord.x);
        case FLAG_BITS::ROTATE_180_BIT:
            return uint16_t2(lastTexel) - coord;
        case FLAG_BITS::ROTATE_270_BIT:
            return uint16_t2(coord.y, lastTexel.x - coord.x);
        case FLAG_BITS::HORIZONTAL_MIRROR_BIT:
            return uint16_t2(lastTexel.x - coord.x, coord.y);
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_90_BIT:
            return lastTexel - coord.yx;
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_180_BIT:
            return uint16_t2(coord.x, lastTexel.y - coord.y);
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_270_BIT:
            return coord.yx;
        default:
            break;
    }
    return uint16_t2(0,0);
}

//! Use this function to apply the swapchain tranformation to the screenspace coordinate `coord` 
//! Usecase = [Scatter]:
//!   When directly writing to your swapchain using `imageStore` in order to match the orientation of the device relative to it's natural orientation. 
//! Warning: Be aware that almost always you'd want to do a single transform in your rendering pipeline.
inline uint16_t2 applyToScreenSpaceCoordinate(const FLAG_BITS transform, const uint16_t2 coord, const uint16_t2 screenSize)
{
    // TODO: use transformMatrix(transform) somehow
    const uint16_t2 lastTexel = screenSize - uint16_t2(1, 1);
    switch (transform)
    {
        case FLAG_BITS::IDENTITY_BIT:
            return coord;
        case FLAG_BITS::ROTATE_90_BIT:
            return uint16_t2(coord.y, lastTexel.x - coord.x);
        case FLAG_BITS::ROTATE_180_BIT:
            return uint16_t2(lastTexel) - coord;
        case FLAG_BITS::ROTATE_270_BIT:
            return uint16_t2(lastTexel.y - coord.y, coord.x);
        case FLAG_BITS::HORIZONTAL_MIRROR_BIT:
            return uint16_t2(lastTexel.x - coord.x, coord.y);
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_90_BIT:
            return coord.yx;
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_180_BIT:
            return uint16_t2(coord.x, lastTexel.y - coord.y);
        case FLAG_BITS::HORIZONTAL_MIRROR_ROTATE_270_BIT:
            return lastTexel - coord.yx;
        default:
            break;
    }
    return uint16_t2(0,0);
}

//! Same as `applyToScreenSpaceCoordinate` but const NDC space
//! If rendering to the swapchain, you may use this function to transform the NDC coordinates directly
//! to be fed into gl_Position in vertex shading
//! Warning: Be aware that almost always you'd want to do a single transform in your rendering pipeline.
inline float32_t2 applyToNDC(const FLAG_BITS transform, const float32_t2 ndc)
{
    return mul(transformMatrix(transform),ndc);
}

// TODO: This is untested
// If rendering directly to the swapchain, dFdx/dFdy operations may be incorrect due to the swapchain
// transform. Use these helper functions to transform the dFdx or dFdy accordingly.
template<typename TwoColumns>
TwoColumns applyToDerivatives(const FLAG_BITS transform, TwoColumns dDx_dDy)
{
    using namespace glsl; // IN HLSL mode, C++ doens't need this to access `inverse`
    return mul(inverse(transformMatrix(transform)),dDx_dDy);
}

}
}
}
#endif