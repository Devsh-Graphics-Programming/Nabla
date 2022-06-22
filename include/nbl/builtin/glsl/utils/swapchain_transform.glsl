#ifndef _NBL_BUILTIN_GLSL_UTILS_SWAPCHAIN_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_SWAPCHAIN_TRANSFORM_INCLUDED_

#include "nbl/builtin/glsl/utils/swapchain_transform_e.h"

// TODO test transforms outside of horizontal mirror rotate 180 (flip y)

ivec2 nbl_glsl_swapchain_transform_forward(in uint swapchainTransform, in ivec2 coord, in ivec2 lastTexel) {
    switch (swapchainTransform) 
    {
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_IDENTITY:
        return coord;
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_90:
        return ivec2(coord.y, lastTexel.x - coord.x);
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_180:
        return ivec2(lastTexel) - coord;
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_270:
        return ivec2(lastTexel.y - coord.y, coord.x);
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR:
        return ivec2(lastTexel.x - coord.x, coord.y);
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90:
        return coord.yx;
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_180:
        return ivec2(coord.x, lastTexel.y - coord.y);
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270:
        return lastTexel - coord.yx;
    default:
        return ivec2(0);
    }
}

ivec2 nbl_glsl_swapchain_transform_backward(in uint swapchainTransform, in ivec2 coord, in ivec2 lastTexel) {
    uint reverseTransform;
    switch (swapchainTransform) 
    {
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_90:
        reverseTransform = NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_270;
        break;
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_270:
        reverseTransform = NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_90;
        break;
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90:
        reverseTransform = NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270;
        break;
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270:
        reverseTransform = NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90;
        break;
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR:
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_180:
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_180:
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_IDENTITY:
        reverseTransform = swapchainTransform;
        break;
    default:
        return ivec2(0);
    }
    return nbl_glsl_swapchain_transform_forward(reverseTransform, coord, lastTexel);
}

ivec2 nbl_glsl_swapchain_transform_preTransform(in uint swapchainTransform, in ivec2 coord, in ivec2 screenSize) {
    return nbl_glsl_swapchain_transform_backward(swapchainTransform, coord, screenSize - 1);
}

ivec2 nbl_glsl_swapchain_transform_postTransform(in uint swapchainTransform, in ivec2 coord, in ivec2 screenSize) {
    return nbl_glsl_swapchain_transform_forward(swapchainTransform, coord, screenSize - 1);
}

ivec2 nbl_glsl_swapchain_transform_preTransformExtents(in uint swapchainTransform, in ivec2 screenSize) {
    switch (swapchainTransform) 
    {
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_IDENTITY:
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR:
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_180:
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_180:
        return screenSize;
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_90:
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_270:
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90:
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270:
        return screenSize.yx;
    default:
        return ivec2(0);
    }
}

mat2 nbl_glsl_swapchain_transform_postTransformMatrix(in uint swapchainTransform) {
    const float sin90 = 1.0, cos90 = 0.0,
        sin180 = 0.0, cos180 = -1.0,
        sin270 = -1.0, cos270 = 0.0;
    switch (swapchainTransform) 
    {
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_90:
        return mat2(vec2(cos90, -sin90),
                    vec2(sin90, cos90));
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_180:
        return mat2(vec2(cos180, -sin180),
                    vec2(sin180, cos180));
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_ROTATE_270:
        return mat2(vec2(cos270, -sin270),
                    vec2(sin270, cos270));
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR:
        return mat2(vec2(-1, 0),
                    vec2(0, 1));
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_90:
        return mat2(vec2(-cos90, sin90),
                    vec2(sin90, cos90));
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_180:
        return mat2(vec2(-cos180, sin180),
                    vec2(sin180, cos180));
    case NBL_GLSL_SWAPCHAIN_TRANSFORM_E_HORIZONTAL_MIRROR_ROTATE_270:
        return mat2(vec2(-cos270, sin270),
                    vec2(sin270, cos270));
    default:
        return mat2(vec2(1, 0),
                    vec2(0, 1));
    }
}

#endif