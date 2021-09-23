#ifndef __NBL_VIDEO_C_VULKAN_COMMON_H_INCLUDED__

#include <volk.h>

namespace nbl::video
{

static inline asset::E_FORMAT getFormatFromVkFormat(VkFormat in)
{
    if (in <= VK_FORMAT_BC7_SRGB_BLOCK)
        return static_cast<asset::E_FORMAT>(in);

    // Note(achal): Some of this ugliness could be remedied if we put the range [EF_ETC2_R8G8B8_UNORM_BLOCK, EF_EAC_R11G11_SNORM_BLOCK] just
    // after EF_BC7_SRGB_BLOCK, not sure how rest of the code will react to it
    if (in >= VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK && in <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK) // [147, 156] --> [175, 184]
        return static_cast<asset::E_FORMAT>(in + 28u);

    if (in >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && in <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) // [157, 184]
        return static_cast<asset::E_FORMAT>(in - 10u);

    // Note(achal): This ugliness is not so easy to get rid of
    if (in >= VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG && in <= VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG) // [1000054000, 1000054007] --> [185, 192]
        return static_cast<asset::E_FORMAT>(in - 1000053815u);

    if (in >= VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM && in <= VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM) // [1000156002, 1000156006] --> [193, 197]
        return static_cast<asset::E_FORMAT>(in - 1000155809);

    return asset::EF_UNKNOWN;
}

static inline ISurface::SColorSpace getColorSpaceFromVkColorSpaceKHR(VkColorSpaceKHR in)
{
    ISurface::SColorSpace result = { asset::ECP_COUNT, asset::EOTF_UNKNOWN };

    switch (in)
    {
    case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
    {
        result.primary = asset::ECP_SRGB;
        result.eotf = asset::EOTF_sRGB;
    } break;

    case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
    {
        result.primary = asset::ECP_DISPLAY_P3;
        result.eotf = asset::EOTF_sRGB; // spec says "sRGB-like"
    } break;

    case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
    {
        result.primary = asset::ECP_SRGB;
        result.eotf = asset::EOTF_IDENTITY;
    } break;

    case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT:
    {
        result.primary = asset::ECP_DISPLAY_P3;
        result.eotf = asset::EOTF_IDENTITY;
    } break;

    case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
    {
        result.primary = asset::ECP_DCI_P3;
        result.eotf = asset::EOTF_DCI_P3_XYZ;
    } break;

    case VK_COLOR_SPACE_BT709_LINEAR_EXT:
    {
        result.primary = asset::ECP_SRGB;
        result.eotf = asset::EOTF_IDENTITY;
    } break;

    case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
    {
        result.primary = asset::ECP_SRGB;
        result.eotf = asset::EOTF_SMPTE_170M;
    } break;

    case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
    {
        result.primary = asset::ECP_BT2020;
        result.eotf = asset::EOTF_IDENTITY;
    } break;

    case VK_COLOR_SPACE_HDR10_ST2084_EXT:
    {
        result.primary = asset::ECP_BT2020;
        result.eotf = asset::EOTF_SMPTE_ST2084;
    } break;

    case VK_COLOR_SPACE_DOLBYVISION_EXT:
    {
        result.primary = asset::ECP_BT2020;
        result.eotf = asset::EOTF_SMPTE_ST2084;
    } break;

    case VK_COLOR_SPACE_HDR10_HLG_EXT:
    {
        result.primary = asset::ECP_BT2020;
        result.eotf = asset::EOTF_HDR10_HLG;
    } break;

    case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT:
    {
        result.primary = asset::ECP_ADOBERGB;
        result.eotf = asset::EOTF_IDENTITY;
    } break;

    case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT:
    {
        result.primary = asset::ECP_ADOBERGB;
        result.eotf = asset::EOTF_GAMMA_2_2;
    } break;

    case VK_COLOR_SPACE_PASS_THROUGH_EXT:
    {
        result.primary = asset::ECP_PASS_THROUGH;
        result.eotf = asset::EOTF_IDENTITY;
    } break;

    case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
    {
        result.primary = asset::ECP_SRGB;
        result.eotf = asset::EOTF_sRGB;
    } break;

    case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD: // this one is completely bogus, I don't understand it at all
    {
        result.primary = asset::ECP_SRGB;
        result.eotf = asset::EOTF_UNKNOWN;
    } break;
    }

    return result;
}

static inline ISurface::E_PRESENT_MODE getPresentModeFromVkPresentModeKHR(VkPresentModeKHR in)
{
    switch (in)
    {
    case VK_PRESENT_MODE_IMMEDIATE_KHR:
        return ISurface::EPM_IMMEDIATE;
    case VK_PRESENT_MODE_MAILBOX_KHR:
        return ISurface::EPM_MAILBOX;
    case VK_PRESENT_MODE_FIFO_KHR:
        return ISurface::EPM_FIFO;
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
        return ISurface::EPM_FIFO_RELAXED;
    default:
        return ISurface::EPM_UNKNOWN;
    }
}

static inline VkFormat getVkFormatFromFormat(asset::E_FORMAT in)
{
    if (in <= asset::EF_BC7_SRGB_BLOCK)
        return static_cast<VkFormat>(in);

    if (in >= asset::EF_ETC2_R8G8B8_UNORM_BLOCK && in <= asset::EF_EAC_R11G11_SNORM_BLOCK)
        return static_cast<VkFormat>(in - 28u);

    if (in >= asset::EF_ASTC_4x4_UNORM_BLOCK && in <= asset::EF_ASTC_12x12_SRGB_BLOCK)
        return static_cast<VkFormat>(in + 10u);

    if (in >= asset::EF_PVRTC1_2BPP_UNORM_BLOCK_IMG && in <= asset::EF_PVRTC2_4BPP_SRGB_BLOCK_IMG)
        return static_cast<VkFormat>(in + 1000053815u);

    if (in >= asset::EF_G8_B8_R8_3PLANE_420_UNORM && in <= asset::EF_G8_B8_R8_3PLANE_444_UNORM)
        return static_cast<VkFormat>(in + 1000155809);

    return VK_FORMAT_MAX_ENUM;
}

static inline VkColorSpaceKHR getVkColorSpaceKHRFromColorSpace(ISurface::SColorSpace in)
{
    if (in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_sRGB)
        return VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    if (in.primary == asset::ECP_DISPLAY_P3 && in.eotf == asset::EOTF_sRGB)
        return VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT;

    if (in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;

    if (in.primary == asset::ECP_DISPLAY_P3 && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT;

    if (in.primary == asset::ECP_DCI_P3 && in.eotf == asset::EOTF_DCI_P3_XYZ)
        return VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT;

    if (in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_BT709_LINEAR_EXT;

    if (in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_SMPTE_170M)
        return VK_COLOR_SPACE_BT709_NONLINEAR_EXT;

    if (in.primary == asset::ECP_BT2020 && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_BT2020_LINEAR_EXT;

    if (in.primary == asset::ECP_BT2020 && in.eotf == asset::EOTF_SMPTE_ST2084)
        return VK_COLOR_SPACE_HDR10_ST2084_EXT;

    if (in.primary == asset::ECP_BT2020 && in.eotf == asset::EOTF_SMPTE_ST2084)
        return VK_COLOR_SPACE_DOLBYVISION_EXT;

    if (in.primary == asset::ECP_BT2020 && in.eotf == asset::EOTF_HDR10_HLG)
        return VK_COLOR_SPACE_HDR10_HLG_EXT;

    if (in.primary == asset::ECP_ADOBERGB && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT;

    if (in.primary == asset::ECP_ADOBERGB && in.eotf == asset::EOTF_GAMMA_2_2)
        return VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT;

    if (in.primary == asset::ECP_PASS_THROUGH && in.eotf == asset::EOTF_IDENTITY)
        return VK_COLOR_SPACE_PASS_THROUGH_EXT;

    if (in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_sRGB)
        return VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT;

    if (in.primary == asset::ECP_SRGB && in.eotf == asset::EOTF_UNKNOWN)
        return VK_COLOR_SPACE_DISPLAY_NATIVE_AMD;

    return VK_COLOR_SPACE_MAX_ENUM_KHR;
}

static inline VkSamplerAddressMode getVkAddressModeFromTexClamp(const asset::ISampler::E_TEXTURE_CLAMP in)
{
    switch (in)
    {
    case asset::ISampler::ETC_REPEAT:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case asset::ISampler::ETC_CLAMP_TO_EDGE:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case asset::ISampler::ETC_CLAMP_TO_BORDER:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case asset::ISampler::ETC_MIRROR:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default:
        assert(!"ADDRESS MODE NOT SUPPORTED!");
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    }
}

static inline std::pair<VkDebugUtilsMessageSeverityFlagsEXT, VkDebugUtilsMessageTypeFlagsEXT> getDebugCallbackFlagsFromLogLevelMask(const core::bitflag<system::ILogger::E_LOG_LEVEL> logLevelMask)
{
    std::pair<VkDebugUtilsMessageSeverityFlagsEXT, VkDebugUtilsMessageTypeFlagsEXT> result = { 0, 0 };
    auto& sev = result.first;
    auto& type = result.second;
    
    if ((logLevelMask & system::ILogger::ELL_DEBUG).value)
    {
        type |= VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    }
    if ((logLevelMask & system::ILogger::ELL_INFO).value)
    {
        sev |= (VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT);
        type |= VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
    }
    if ((logLevelMask & system::ILogger::ELL_WARNING).value)
    {
        sev |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    }
    if ((logLevelMask & system::ILogger::ELL_PERFORMANCE).value)
    {
        type |= VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    }
    if ((logLevelMask & system::ILogger::ELL_ERROR).value)
    {
        sev |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    }

    return result;
}

}

#define __NBL_VIDEO_C_VULKAN_COMMON_H_INCLUDED__
#endif
