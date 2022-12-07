
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SURFACE_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_SURFACE_TRANSFORM_INCLUDED_

#include "nbl/builtin/hlsl/utils/surface_transform_e.h"


namespace nbl
{
namespace hlsl
{
namespace surface_transform
{


//! Use this function to apply the INVERSE of swapchain tranformation to the screenspace coordinate `coord` 
//! For example when the device orientation is 90°CW then this transforms the point 90°CCW.
//! Usecase = [Gather]:
//!   Applications such as raytracing in shaders where you would want to generate rays from screen space coordinates. 
//! Warnings: 
//! - You don't need to consider this using in your raytracing shaders if you apply the forward transformation to your projection matrix.
//! - Be aware that almost always you'd want to do a single transform in your rendering pipeline.
int2 applyInverseToScreenSpaceCoordinate(in uint swapchainTransform, in int2 coord, in int2 screenSize)
{
    int2 lastTexel = screenSize - (1).xx;
    switch (swapchainTransform) 
    {
    case IDENTITY:
        return coord;
    case ROTATE_90:
        return int2(lastTexel.y - coord.y, coord.x);
    case ROTATE_180:
        return int2(lastTexel) - coord;
    case ROTATE_270:
        return int2(coord.y, lastTexel.x - coord.x);
    case HORIZONTAL_MIRROR:
        return int2(lastTexel.x - coord.x, coord.y);
    case HORIZONTAL_MIRROR_ROTATE_90:
        return lastTexel - coord.yx;
    case HORIZONTAL_MIRROR_ROTATE_180:
        return int2(coord.x, lastTexel.y - coord.y);
    case HORIZONTAL_MIRROR_ROTATE_270:
        return coord.yx;
    default:
        return (0).xx;
    }
}

//! Use this function to apply the swapchain tranformation to the screenspace coordinate `coord` 
//! Usecase = [Scatter]:
//!   When directly writing to your swapchain using `imageStore` in order to match the orientation of the device relative to it's natural orientation. 
//! Warning: Be aware that almost always you'd want to do a single transform in your rendering pipeline.
int2 applyToScreenSpaceCoordinate(in uint swapchainTransform, in int2 coord, in int2 screenSize)
{
    int2 lastTexel = screenSize - (1).xx;
    switch (swapchainTransform) 
    {
    case IDENTITY:
        return coord;
    case ROTATE_90:
        return int2(coord.y, lastTexel.x - coord.x);
    case ROTATE_180:
        return int2(lastTexel) - coord;
    case ROTATE_270:
        return int2(lastTexel.y - coord.y, coord.x);
    case HORIZONTAL_MIRROR:
        return int2(lastTexel.x - coord.x, coord.y);
    case HORIZONTAL_MIRROR_ROTATE_90:
        return coord.yx;
    case HORIZONTAL_MIRROR_ROTATE_180:
        return int2(coord.x, lastTexel.y - coord.y);
    case HORIZONTAL_MIRROR_ROTATE_270:
        return lastTexel - coord.yx;
    default:
        return (0).xx;
    }
}

//! [width,height] might switch to [height, width] in orientations such as 90°CW
//! Usecase: Currently none in the shaders
int2 transformedExtents(in uint swapchainTransform, in int2 screenSize)
{
    switch (swapchainTransform) 
    {
    case IDENTITY:
    case HORIZONTAL_MIRROR:
    case HORIZONTAL_MIRROR_ROTATE_180:
    case ROTATE_180:
        return screenSize;
    case ROTATE_90:
    case ROTATE_270:
    case HORIZONTAL_MIRROR_ROTATE_90:
    case HORIZONTAL_MIRROR_ROTATE_270:
        return screenSize.yx;
    default:
        return (0).xx;
    }
}

// TODO: surface_transform::transformedDerivatives implementations are untested

// If rendering directly to the swapchain, dFdx/dFdy operations may be incorrect due to the swapchain
// transform. Use these helper functions to transform the dFdx or dFdy accordingly.

float2 transformedDerivatives(in uint swapchainTransform, in float2 ddxDdy)
{
    #define OUTPUT_TYPE float2
    #include "nbl/builtin/hlsl/utils/surface_transform_transformedDerivatives.hlsl"
    #undef OUTPUT_TYPE
}
float2x2 transformedDerivatives(in uint swapchainTransform, in float2x2 ddxDdy)
{
    #define OUTPUT_TYPE mat2
    #include "nbl/builtin/hlsl/utils/surface_transform_transformedDerivatives.hlsl"
    #undef OUTPUT_TYPE
}
float2x3 transformedDerivatives(in uint swapchainTransform, in float2x3 ddxDdy)
{
    #define OUTPUT_TYPE float2x3
    #include "nbl/builtin/hlsl/utils/surface_transform_transformedDerivatives.hlsl"
    #undef OUTPUT_TYPE
}
float2x4 transformedDerivatives(in uint swapchainTransform, in float2x4 ddxDdy)
{
    #define OUTPUT_TYPE float2x4
    #include "nbl/builtin/hlsl/utils/surface_transform_transformedDerivatives.hlsl"
    #undef OUTPUT_TYPE
}

//! Same as `surface_transform::applyToScreenSpaceCoordinate` but in NDC space
//! If rendering to the swapchain, you may use this function to transform the NDC coordinates directly
//! to be fed into gl_Position in vertex shading
//! Warning: Be aware that almost always you'd want to do a single transform in your rendering pipeline.
float2 applyToNDC(in uint swapchainTransform, in float2 ndc)
{
    const float sin90 = 1.0, cos90 = 0.0,
        sin180 = 0.0, cos180 = -1.0,
        sin270 = -1.0, cos270 = 0.0;
    switch (swapchainTransform) 
    {
    case ROTATE_90:
        return ndc * mat2(float2(cos90, -sin90),
                          float2(sin90, cos90));
    case ROTATE_180:
        return ndc * mat2(float2(cos180, -sin180),
                          float2(sin180, cos180));
    case ROTATE_270:
        return ndc * mat2(float2(cos270, -sin270),
                          float2(sin270, cos270));
    case HORIZONTAL_MIRROR:
        return ndc * mat2(float2(-1, 0),
                          float2(0, 1));
    case HORIZONTAL_MIRROR_ROTATE_90:
        return ndc * mat2(float2(-cos90, sin90),
                          float2(sin90, cos90));
    case HORIZONTAL_MIRROR_ROTATE_180:
        return ndc * mat2(float2(-cos180, sin180),
                          float2(sin180, cos180));
    case HORIZONTAL_MIRROR_ROTATE_270:
        return ndc * mat2(float2(-cos270, sin270),
                          float2(sin270, cos270));
    default:
        return ndc;
    }
}



}
}
}

#endif