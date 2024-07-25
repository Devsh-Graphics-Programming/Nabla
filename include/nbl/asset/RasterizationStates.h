// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_RASTERIZATION_STATES_H_INCLUDED_
#define _NBL_ASSET_RASTERIZATION_STATES_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/IRenderpass.h"


namespace nbl::asset
{

struct SViewport
{
    float x = 0.f;
    float y = 0.f;
    float width, height;
    // Reverse-Z is our framework default
    float minDepth = 1.f;
    float maxDepth = 0.f;
};


enum E_PRIMITIVE_TOPOLOGY : uint16_t
{
    EPT_POINT_LIST = 0,
    EPT_LINE_LIST = 1,
    EPT_LINE_STRIP = 2,
    EPT_TRIANGLE_LIST = 3,
    EPT_TRIANGLE_STRIP = 4,
    EPT_TRIANGLE_FAN = 5,
    EPT_LINE_LIST_WITH_ADJACENCY = 6,
    EPT_LINE_STRIP_WITH_ADJACENCY = 7,
    EPT_TRIANGLE_LIST_WITH_ADJACENCY = 8,
    EPT_TRIANGLE_STRIP_WITH_ADJACENCY = 9,
    EPT_PATCH_LIST = 10
};


enum E_STENCIL_OP : uint8_t
{
    ESO_KEEP = 0,
    ESO_ZERO = 1,
    ESO_REPLACE = 2,
    ESO_INCREMENT_AND_CLAMP = 3,
    ESO_DECREMENT_AND_CLAMP = 4,
    ESO_INVERT = 5,
    ESO_INCREMENT_AND_WRAP = 6,
    ESO_DECREMENT_AND_WRAP = 7
};

enum E_COMPARE_OP : uint8_t
{
    ECO_NEVER = 0,
    ECO_LESS = 1,
    ECO_EQUAL = 2,
    ECO_LESS_OR_EQUAL = 3,
    ECO_GREATER = 4,
    ECO_NOT_EQUAL = 5,
    ECO_GREATER_OR_EQUAL = 6,
    ECO_ALWAYS = 7
};

struct SStencilOpParams
{
    inline auto operator<=>(const SStencilOpParams& other) const = default;

    inline bool needsStencilTestEnable(const bool depthWillAlwaysPass) const
    {
        switch (compareOp)
        {
            case ECO_NEVER:
                if (failOp==ESO_KEEP)
                    return false;
                break;
            case ECO_ALWAYS:
                if (passOp==ESO_KEEP && (depthWillAlwaysPass || depthFailOp==ESO_KEEP))
                    return false;
                break;
            default:
                if (failOp==ESO_KEEP && passOp==ESO_KEEP && depthFailOp==ESO_KEEP)
                    return false;
                break;
        }
        return true;
    }

    E_STENCIL_OP failOp : 3 = ESO_KEEP;
    E_STENCIL_OP passOp : 3 = ESO_KEEP;
    E_STENCIL_OP depthFailOp : 3 = ESO_KEEP;
    E_COMPARE_OP compareOp : 3 = ECO_ALWAYS;
};
static_assert(sizeof(SStencilOpParams)==2, "Unexpected size!");


enum E_POLYGON_MODE : uint8_t
{
    EPM_FILL = 0,
    EPM_LINE = 1,
    EPM_POINT = 2
};

enum E_FACE_CULL_MODE : uint8_t
{
    EFCM_NONE = 0,
    EFCM_FRONT_BIT = 1,
    EFCM_BACK_BIT = 2,
    EFCM_FRONT_AND_BACK = 3
};

struct SRasterizationParams
{
    inline auto operator<=>(const SRasterizationParams& other) const = default;

    inline bool depthTestEnable() const
    {
        return depthCompareOp!=asset::ECO_ALWAYS || depthWriteEnable;
    }

    inline bool stencilTestEnable() const
    {
        const bool depthWillAlwaysPass = depthCompareOp==asset::ECO_ALWAYS;
        return frontStencilOps.needsStencilTestEnable(depthWillAlwaysPass) || backStencilOps.needsStencilTestEnable(depthWillAlwaysPass);
    }


    struct {
        uint8_t viewportCount : 5 = 1;
        uint8_t samplesLog2 : 3 = 0;
    };
    struct {
        uint8_t depthClampEnable : 1 = false;
        uint8_t rasterizerDiscard : 1 = false;
        E_POLYGON_MODE polygonMode : 2 = EPM_FILL;
        E_FACE_CULL_MODE faceCullingMode : 2 = EFCM_BACK_BIT;
        uint8_t frontFaceIsCCW : 1 = true;
        uint8_t depthBiasEnable : 1 = false;
    };
    struct {
        uint8_t alphaToCoverageEnable : 1 = false;
        uint8_t alphaToOneEnable : 1 = false;
        uint8_t depthWriteEnable : 1 = true;
        E_COMPARE_OP depthCompareOp : 3 = ECO_GREATER;
        uint8_t depthBoundsTestEnable : 1 = false;
    };
    SStencilOpParams frontStencilOps;
    SStencilOpParams backStencilOps;
    uint8_t minSampleShadingUnorm = 0;
    uint32_t sampleMask[2] = { ~0u,~0u };
};
static_assert(sizeof(SRasterizationParams)==16, "Unexpected size!");

enum E_BLEND_FACTOR : uint8_t
{
    EBF_ZERO = 0,
    EBF_ONE = 1,
    EBF_SRC_COLOR = 2,
    EBF_ONE_MINUS_SRC_COLOR = 3,
    EBF_DST_COLOR = 4,
    EBF_ONE_MINUS_DST_COLOR = 5,
    EBF_SRC_ALPHA = 6,
    EBF_ONE_MINUS_SRC_ALPHA = 7,
    EBF_DST_ALPHA = 8,
    EBF_ONE_MINUS_DST_ALPHA = 9,
    EBF_CONSTANT_COLOR = 10,
    EBF_ONE_MINUS_CONSTANT_COLOR = 11,
    EBF_CONSTANT_ALPHA = 12,
    EBF_ONE_MINUS_CONSTANT_ALPHA = 13,
    EBF_SRC_ALPHA_SATURATE = 14,
    EBF_SRC1_COLOR = 15,
    EBF_ONE_MINUS_SRC1_COLOR = 16,
    EBF_SRC1_ALPHA = 17,
    EBF_ONE_MINUS_SRC1_ALPHA = 18
};

enum E_BLEND_OP : uint8_t
{
    EBO_ADD = 0,
    EBO_SUBTRACT = 1,
    EBO_REVERSE_SUBTRACT = 2,
    EBO_MIN = 3,
    EBO_MAX = 4,
/* we don't support blend_operation_advanced
    EBO_ZERO_EXT,
    EBO_SRC_EXT,
    EBO_DST_EXT,
    EBO_SRC_OVER_EXT,
    EBO_DST_OVER_EXT,
    EBO_SRC_IN_EXT,
    EBO_DST_IN_EXT,
    EBO_SRC_OUT_EXT,
    EBO_DST_OUT_EXT,
    EBO_SRC_ATOP_EXT,
    EBO_DST_ATOP_EXT,
    EBO_XOR_EXT,
    EBO_MULTIPLY_EXT,
    EBO_SCREEN_EXT,
    EBO_OVERLAY_EXT,
    EBO_DARKEN_EXT,
    EBO_LIGHTEN_EXT,
    EBO_COLORDODGE_EXT,
    EBO_COLORBURN_EXT,
    EBO_HARDLIGHT_EXT,
    EBO_SOFTLIGHT_EXT,
    EBO_DIFFERENCE_EXT,
    EBO_EXCLUSION_EXT,
    EBO_INVERT_EXT,
    EBO_INVERT_RGB_EXT,
    EBO_LINEARDODGE_EXT,
    EBO_LINEARBURN_EXT,
    EBO_VIVIDLIGHT_EXT,
    EBO_LINEARLIGHT_EXT,
    EBO_PINLIGHT_EXT,
    EBO_HARDMIX_EXT,
    EBO_HSL_HUE_EXT,
    EBO_HSL_SATURATION_EXT,
    EBO_HSL_COLOR_EXT,
    EBO_HSL_LUMINOSITY_EXT,
    EBO_PLUS_EXT,
    EBO_PLUS_CLAMPED_EXT,
    EBO_PLUS_CLAMPED_ALPHA_EXT,
    EBO_PLUS_DARKER_EXT,
    EBO_MINUS_EXT,
    EBO_MINUS_CLAMPED_EXT,
    EBO_CONTRAST_EXT,
    EBO_INVERT_OVG_EXT,
    EBO_RED_EXT,
    EBO_GREEN_EXT,
    EBO_BLUE_EXT
*/
};

struct SColorAttachmentBlendParams
{
    inline auto operator<=>(const SColorAttachmentBlendParams& other) const = default;

    inline bool blendEnabled() const
    {
        SColorAttachmentBlendParams _default = {};
        _default.colorWriteMask = colorWriteMask;
        return _default!=*this;
    }

    // no blend enable flag, because defaults are identical to no blend
    uint32_t srcColorFactor : 5 = EBF_ONE;
    uint32_t dstColorFactor : 5 = EBF_ZERO;
    uint32_t colorBlendOp : 3 = EBO_ADD;
    
    uint32_t srcAlphaFactor : 5 = EBF_ONE;
    uint32_t dstAlphaFactor : 5 = EBF_ZERO;
    uint32_t alphaBlendOp : 3 = EBO_ADD;

    //RGBA, LSB is R, MSB is A
    uint32_t colorWriteMask : 4 = 0b1111;
};
static_assert(sizeof(SColorAttachmentBlendParams)==4u, "Unexpected size of SColorAttachmentBlendParams (should be 5)");

enum E_LOGIC_OP : uint8_t
{
    ELO_CLEAR = 0,
    ELO_AND = 1,
    ELO_AND_REVERSE = 2,
    ELO_COPY = 3,
    ELO_AND_INVERTED = 4,
    ELO_NO_OP = 5,
    ELO_XOR = 6,
    ELO_OR = 7,
    ELO_NOR = 8,
    ELO_EQUIVALENT = 9,
    ELO_INVERT = 10,
    ELO_OR_REVERSE = 11,
    ELO_COPY_INVERTED = 12,
    ELO_OR_INVERTED = 13,
    ELO_NAND = 14,
    ELO_SET = 15
};

struct SBlendParams
{
    inline auto operator<=>(const SBlendParams& other) const = default;

    // Implicitly satisfies:
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html#VUID-VkGraphicsPipelineCreateInfo-renderPass-07609
    SColorAttachmentBlendParams blendParams[IRenderpass::SCreationParams::SSubpassDescription::MaxColorAttachments] = {};
    // If logicOpEnable is not ELO_NO_OP, then a logical operation selected by logicOp is applied between each color attachment
    // and the fragment’s corresponding output value, and blending of all attachments is treated as if it were disabled.
    // Any attachments using color formats for which logical operations are not supported simply pass through the color values unmodified.
    // The logical operation is applied independently for each of the red, green, blue, and alpha components.
    E_LOGIC_OP logicOp : 4 = ELO_NO_OP;
};

}
#endif