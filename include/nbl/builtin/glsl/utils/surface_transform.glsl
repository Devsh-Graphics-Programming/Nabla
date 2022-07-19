#ifndef _NBL_BUILTIN_GLSL_SURFACE_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_GLSL_SURFACE_TRANSFORM_INCLUDED_

#include "nbl/builtin/glsl/utils/surface_transform_e.h"

//! Use this function to apply the INVERSE of swapchain tranformation to the screenspace coordinate `coord` 
//! For example when the device orientation is 90°CW then this transforms the point 90°CCW.
//! Usecase = [Gather]:
//!   Applications such as raytracing in shaders where you would want to generate rays from screen space coordinates. 
//! Warnings: 
//! - You don't need to consider this using in your raytracing shaders if you apply the forward transformation to your projection matrix.
//! - Be aware that almost always you'd want to do a single transform in your rendering pipeline.
ivec2 nbl_glsl_surface_transform_applyInverseToScreenSpaceCoordinate(in uint swapchainTransform, in ivec2 coord, in ivec2 screenSize) {
    ivec2 lastTexel = screenSize - ivec2(1);
    switch (swapchainTransform) 
    {
    case NBL_GLSL_SURFACE_TRANSFORM_E_IDENTITY:
        return coord;
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_90:
        return ivec2(lastTexel.y - coord.y, coord.x);
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_180:
        return ivec2(lastTexel) - coord;
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_270:
        return ivec2(coord.y, lastTexel.x - coord.x);
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR:
        return ivec2(lastTexel.x - coord.x, coord.y);
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90:
        return lastTexel - coord.yx;
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_180:
        return ivec2(coord.x, lastTexel.y - coord.y);
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270:
        return coord.yx;
    default:
        return ivec2(0);
    }
}

//! Use this function to apply the swapchain tranformation to the screenspace coordinate `coord` 
//! Usecase = [Scatter]:
//!   When directly writing to your swapchain using `imageStore` in order to match the orientation of the device relative to it's natural orientation. 
//! Warning: Be aware that almost always you'd want to do a single transform in your rendering pipeline.
ivec2 nbl_glsl_surface_transform_applyToScreenSpaceCoordinate(in uint swapchainTransform, in ivec2 coord, in ivec2 screenSize) {
    ivec2 lastTexel = screenSize - ivec2(1);
    switch (swapchainTransform) 
    {
    case NBL_GLSL_SURFACE_TRANSFORM_E_IDENTITY:
        return coord;
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_90:
        return ivec2(coord.y, lastTexel.x - coord.x);
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_180:
        return ivec2(lastTexel) - coord;
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_270:
        return ivec2(lastTexel.y - coord.y, coord.x);
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR:
        return ivec2(lastTexel.x - coord.x, coord.y);
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90:
        return coord.yx;
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_180:
        return ivec2(coord.x, lastTexel.y - coord.y);
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270:
        return lastTexel - coord.yx;
    default:
        return ivec2(0);
    }
}

//! [width,height] might switch to [height, width] in orientations such as 90°CW
//! Usecase: Currently none in the shaders
ivec2 nbl_glsl_surface_transform_transformedExtents(in uint swapchainTransform, in ivec2 screenSize) {
    switch (swapchainTransform) 
    {
    case NBL_GLSL_SURFACE_TRANSFORM_E_IDENTITY:
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR:
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_180:
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_180:
        return screenSize;
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_90:
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_270:
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90:
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270:
        return screenSize.yx;
    default:
        return ivec2(0);
    }
}

// TODO: nbl_glsl_surface_transform_transformedDerivatives implementations are untested

// If rendering directly to the swapchain, dFdx/dFdy operations may be incorrect due to the swapchain
// transform. Use these helper functions to transform the dFdx or dFdy accordingly.

vec2 nbl_glsl_surface_transform_transformedDerivatives(in uint swapchainTransform, in vec2 ddxDdy) {
    #define OUTPUT_TYPE vec2
    #include "nbl/builtin/glsl/utils/surface_transform_transformedDerivatives.glsl"
    #undef OUTPUT_TYPE
}
mat2 nbl_glsl_surface_transform_transformedDerivatives(in uint swapchainTransform, in mat2 ddxDdy) {
    #define OUTPUT_TYPE mat2
    #include "nbl/builtin/glsl/utils/surface_transform_transformedDerivatives.glsl"
    #undef OUTPUT_TYPE
}
mat2x3 nbl_glsl_surface_transform_transformedDerivatives(in uint swapchainTransform, in mat2x3 ddxDdy) {
    #define OUTPUT_TYPE mat2x3
    #include "nbl/builtin/glsl/utils/surface_transform_transformedDerivatives.glsl"
    #undef OUTPUT_TYPE
}
mat2x4 nbl_glsl_surface_transform_transformedDerivatives(in uint swapchainTransform, in mat2x4 ddxDdy) {
    #define OUTPUT_TYPE mat2x4
    #include "nbl/builtin/glsl/utils/surface_transform_transformedDerivatives.glsl"
    #undef OUTPUT_TYPE
}

//! Same as `nbl_glsl_surface_transform_applyToScreenSpaceCoordinate` but in NDC space
//! If rendering to the swapchain, you may use this function to transform the NDC coordinates directly
//! to be fed into gl_Position in vertex shading
//! Warning: Be aware that almost always you'd want to do a single transform in your rendering pipeline.
vec2 nbl_glsl_surface_transform_applyToNDC(in uint swapchainTransform, in vec2 ndc) {
    const float sin90 = 1.0, cos90 = 0.0,
        sin180 = 0.0, cos180 = -1.0,
        sin270 = -1.0, cos270 = 0.0;
    switch (swapchainTransform) 
    {
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_90:
        return ndc * mat2(vec2(cos90, -sin90),
                          vec2(sin90, cos90));
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_180:
        return ndc * mat2(vec2(cos180, -sin180),
                          vec2(sin180, cos180));
    case NBL_GLSL_SURFACE_TRANSFORM_E_ROTATE_270:
        return ndc * mat2(vec2(cos270, -sin270),
                          vec2(sin270, cos270));
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR:
        return ndc * mat2(vec2(-1, 0),
                          vec2(0, 1));
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90:
        return ndc * mat2(vec2(-cos90, sin90),
                          vec2(sin90, cos90));
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_180:
        return ndc * mat2(vec2(-cos180, sin180),
                          vec2(sin180, cos180));
    case NBL_GLSL_SURFACE_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270:
        return ndc * mat2(vec2(-cos270, sin270),
                          vec2(sin270, cos270));
    default:
        return ndc;
    }
}

#endif