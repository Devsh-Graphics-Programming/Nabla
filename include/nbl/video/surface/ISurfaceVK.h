#ifndef __NBL_I_SURFACE_VK_H_INCLUDED__
#define __NBL_I_SURFACE_VK_H_INCLUDED__

#include <volk.h>
#include "nbl/video/surface/ISurface.h"

namespace nbl::video
{

class IPhysicalDevice;
class CVulkanConnection;

class ISurfaceVK : public ISurface
{
public:
    inline VkSurfaceKHR getInternalObject() const { return m_surface; }

    bool isSupported(const IPhysicalDevice* dev, uint32_t _queueIx) const override;

    static inline asset::E_FORMAT getFormat(VkFormat in)
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

        // Todo(achal): Log a warning that you got an unrecognized format
        return asset::EF_UNKNOWN;
    }

    // Todo(achal): Check it, a lot of stuff could be incorrect!
    static inline ISurface::SColorSpace getColorSpace(VkColorSpaceKHR in)
    {
        ISurface::SColorSpace result = {};

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

        default:
        {
            // Todo(achal): Log warning unknown color space
        } break;
        }

        return result;
    }

    static inline ISurface::E_PRESENT_MODE getPresentMode(VkPresentModeKHR in)
    {
        switch (in)
        {
        case VK_PRESENT_MODE_IMMEDIATE_KHR:
            return EPM_IMMEDIATE;
        case VK_PRESENT_MODE_MAILBOX_KHR:
            return EPM_MAILBOX;
        case VK_PRESENT_MODE_FIFO_KHR:
            return EPM_FIFO;
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
            return EPM_FIFO_RELAXED;
        default:
        {
            // Todo(achal): Log warning unknown present modes
            return static_cast<ISurface::E_PRESENT_MODE>(0);
        }
        }
    }

    static inline VkFormat getVkFormat(asset::E_FORMAT in)
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

        // Todo(achal): Log a warning that you got an unrecognized format
        return VK_FORMAT_MAX_ENUM;
    }

    // Todo(achal): Check it, a lot of stuff could be incorrect!
    static inline VkColorSpaceKHR getVkColorSpaceKHR(ISurface::SColorSpace in)
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

        // Todo(achal): Log warning unknown color space
        return VK_COLOR_SPACE_MAX_ENUM_KHR;
    }

// protected:
    ISurfaceVK(core::smart_refctd_ptr<const CVulkanConnection>&& apiConnection);

    virtual ~ISurfaceVK();

    VkSurfaceKHR m_surface;
    core::smart_refctd_ptr<const CVulkanConnection> m_apiConnection;
};

}

#endif