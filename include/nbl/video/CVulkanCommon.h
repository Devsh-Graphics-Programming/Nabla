#ifndef _NBL_VIDEO_C_VULKAN_COMMON_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_COMMON_H_INCLUDED_

#include <volk.h>

namespace nbl::video
{

static constexpr uint32_t MinimumVulkanApiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);

inline asset::E_FORMAT getFormatFromVkFormat(VkFormat in)
{
    switch (in)
    {
    case VK_FORMAT_D16_UNORM: return asset::E_FORMAT::EF_D16_UNORM;
    case VK_FORMAT_X8_D24_UNORM_PACK32: return asset::E_FORMAT::EF_X8_D24_UNORM_PACK32;
    case VK_FORMAT_D32_SFLOAT: return asset::E_FORMAT::EF_D32_SFLOAT;
    case VK_FORMAT_S8_UINT: return asset::E_FORMAT::EF_S8_UINT;
    case VK_FORMAT_D16_UNORM_S8_UINT: return asset::E_FORMAT::EF_D16_UNORM_S8_UINT;
    case VK_FORMAT_D24_UNORM_S8_UINT: return asset::E_FORMAT::EF_D24_UNORM_S8_UINT;
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return asset::E_FORMAT::EF_D32_SFLOAT_S8_UINT;
    case VK_FORMAT_R4G4_UNORM_PACK8: return asset::E_FORMAT::EF_R4G4_UNORM_PACK8;
    case VK_FORMAT_R4G4B4A4_UNORM_PACK16: return asset::E_FORMAT::EF_R4G4B4A4_UNORM_PACK16;
    case VK_FORMAT_B4G4R4A4_UNORM_PACK16: return asset::E_FORMAT::EF_B4G4R4A4_UNORM_PACK16;
    case VK_FORMAT_R5G6B5_UNORM_PACK16: return asset::E_FORMAT::EF_R5G6B5_UNORM_PACK16;
    case VK_FORMAT_B5G6R5_UNORM_PACK16: return asset::E_FORMAT::EF_B5G6R5_UNORM_PACK16;
    case VK_FORMAT_R5G5B5A1_UNORM_PACK16: return asset::E_FORMAT::EF_R5G5B5A1_UNORM_PACK16;
    case VK_FORMAT_B5G5R5A1_UNORM_PACK16: return asset::E_FORMAT::EF_B5G5R5A1_UNORM_PACK16;
    case VK_FORMAT_A1R5G5B5_UNORM_PACK16: return asset::E_FORMAT::EF_A1R5G5B5_UNORM_PACK16;
    case VK_FORMAT_R8_UNORM: return asset::E_FORMAT::EF_R8_UNORM;
    case VK_FORMAT_R8_SNORM: return asset::E_FORMAT::EF_R8_SNORM;
    case VK_FORMAT_R8_USCALED: return asset::E_FORMAT::EF_R8_USCALED;
    case VK_FORMAT_R8_SSCALED: return asset::E_FORMAT::EF_R8_SSCALED;
    case VK_FORMAT_R8_UINT: return asset::E_FORMAT::EF_R8_UINT;
    case VK_FORMAT_R8_SINT: return asset::E_FORMAT::EF_R8_SINT;
    case VK_FORMAT_R8_SRGB: return asset::E_FORMAT::EF_R8_SRGB;
    case VK_FORMAT_R8G8_UNORM: return asset::E_FORMAT::EF_R8G8_UNORM;
    case VK_FORMAT_R8G8_SNORM: return asset::E_FORMAT::EF_R8G8_SNORM;
    case VK_FORMAT_R8G8_USCALED: return asset::E_FORMAT::EF_R8G8_USCALED;
    case VK_FORMAT_R8G8_SSCALED: return asset::E_FORMAT::EF_R8G8_SSCALED;
    case VK_FORMAT_R8G8_UINT: return asset::E_FORMAT::EF_R8G8_UINT;
    case VK_FORMAT_R8G8_SINT: return asset::E_FORMAT::EF_R8G8_SINT;
    case VK_FORMAT_R8G8_SRGB: return asset::E_FORMAT::EF_R8G8_SRGB;
    case VK_FORMAT_R8G8B8_UNORM: return asset::E_FORMAT::EF_R8G8B8_UNORM;
    case VK_FORMAT_R8G8B8_SNORM: return asset::E_FORMAT::EF_R8G8B8_SNORM;
    case VK_FORMAT_R8G8B8_USCALED: return asset::E_FORMAT::EF_R8G8B8_USCALED;
    case VK_FORMAT_R8G8B8_SSCALED: return asset::E_FORMAT::EF_R8G8B8_SSCALED;
    case VK_FORMAT_R8G8B8_UINT: return asset::E_FORMAT::EF_R8G8B8_UINT;
    case VK_FORMAT_R8G8B8_SINT: return asset::E_FORMAT::EF_R8G8B8_SINT;
    case VK_FORMAT_R8G8B8_SRGB: return asset::E_FORMAT::EF_R8G8B8_SRGB;
    case VK_FORMAT_B8G8R8_UNORM: return asset::E_FORMAT::EF_B8G8R8_UNORM;
    case VK_FORMAT_B8G8R8_SNORM: return asset::E_FORMAT::EF_B8G8R8_SNORM;
    case VK_FORMAT_B8G8R8_USCALED: return asset::E_FORMAT::EF_B8G8R8_USCALED;
    case VK_FORMAT_B8G8R8_SSCALED: return asset::E_FORMAT::EF_B8G8R8_SSCALED;
    case VK_FORMAT_B8G8R8_UINT: return asset::E_FORMAT::EF_B8G8R8_UINT;
    case VK_FORMAT_B8G8R8_SINT: return asset::E_FORMAT::EF_B8G8R8_SINT;
    case VK_FORMAT_B8G8R8_SRGB: return asset::E_FORMAT::EF_B8G8R8_SRGB;
    case VK_FORMAT_R8G8B8A8_UNORM: return asset::E_FORMAT::EF_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SNORM: return asset::E_FORMAT::EF_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8A8_USCALED: return asset::E_FORMAT::EF_R8G8B8A8_USCALED;
    case VK_FORMAT_R8G8B8A8_SSCALED: return asset::E_FORMAT::EF_R8G8B8A8_SSCALED;
    case VK_FORMAT_R8G8B8A8_UINT: return asset::E_FORMAT::EF_R8G8B8A8_UINT;
    case VK_FORMAT_R8G8B8A8_SINT: return asset::E_FORMAT::EF_R8G8B8A8_SINT;
    case VK_FORMAT_R8G8B8A8_SRGB: return asset::E_FORMAT::EF_R8G8B8A8_SRGB;
    case VK_FORMAT_B8G8R8A8_UNORM: return asset::E_FORMAT::EF_B8G8R8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SNORM: return asset::E_FORMAT::EF_B8G8R8A8_SNORM;
    case VK_FORMAT_B8G8R8A8_USCALED: return asset::E_FORMAT::EF_B8G8R8A8_USCALED;
    case VK_FORMAT_B8G8R8A8_SSCALED: return asset::E_FORMAT::EF_B8G8R8A8_SSCALED;
    case VK_FORMAT_B8G8R8A8_UINT: return asset::E_FORMAT::EF_B8G8R8A8_UINT;
    case VK_FORMAT_B8G8R8A8_SINT: return asset::E_FORMAT::EF_B8G8R8A8_SINT;
    case VK_FORMAT_B8G8R8A8_SRGB: return asset::E_FORMAT::EF_B8G8R8A8_SRGB;
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_UNORM_PACK32;
    case VK_FORMAT_A8B8G8R8_SNORM_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_SNORM_PACK32;
    case VK_FORMAT_A8B8G8R8_USCALED_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_USCALED_PACK32;
    case VK_FORMAT_A8B8G8R8_SSCALED_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_SSCALED_PACK32;
    case VK_FORMAT_A8B8G8R8_UINT_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_UINT_PACK32;
    case VK_FORMAT_A8B8G8R8_SINT_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_SINT_PACK32;
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return asset::E_FORMAT::EF_A8B8G8R8_SRGB_PACK32;
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_UNORM_PACK32;
    case VK_FORMAT_A2R10G10B10_SNORM_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_SNORM_PACK32;
    case VK_FORMAT_A2R10G10B10_USCALED_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_USCALED_PACK32;
    case VK_FORMAT_A2R10G10B10_SSCALED_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_SSCALED_PACK32;
    case VK_FORMAT_A2R10G10B10_UINT_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_UINT_PACK32;
    case VK_FORMAT_A2R10G10B10_SINT_PACK32: return asset::E_FORMAT::EF_A2R10G10B10_SINT_PACK32;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_UNORM_PACK32;
    case VK_FORMAT_A2B10G10R10_SNORM_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_SNORM_PACK32;
    case VK_FORMAT_A2B10G10R10_USCALED_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_USCALED_PACK32;
    case VK_FORMAT_A2B10G10R10_SSCALED_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_SSCALED_PACK32;
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_UINT_PACK32;
    case VK_FORMAT_A2B10G10R10_SINT_PACK32: return asset::E_FORMAT::EF_A2B10G10R10_SINT_PACK32;
    case VK_FORMAT_R16_UNORM: return asset::E_FORMAT::EF_R16_UNORM;
    case VK_FORMAT_R16_SNORM: return asset::E_FORMAT::EF_R16_SNORM;
    case VK_FORMAT_R16_USCALED: return asset::E_FORMAT::EF_R16_USCALED;
    case VK_FORMAT_R16_SSCALED: return asset::E_FORMAT::EF_R16_SSCALED;
    case VK_FORMAT_R16_UINT: return asset::E_FORMAT::EF_R16_UINT;
    case VK_FORMAT_R16_SINT: return asset::E_FORMAT::EF_R16_SINT;
    case VK_FORMAT_R16_SFLOAT: return asset::E_FORMAT::EF_R16_SFLOAT;
    case VK_FORMAT_R16G16_UNORM: return asset::E_FORMAT::EF_R16G16_UNORM;
    case VK_FORMAT_R16G16_SNORM: return asset::E_FORMAT::EF_R16G16_SNORM;
    case VK_FORMAT_R16G16_USCALED: return asset::E_FORMAT::EF_R16G16_USCALED;
    case VK_FORMAT_R16G16_SSCALED: return asset::E_FORMAT::EF_R16G16_SSCALED;
    case VK_FORMAT_R16G16_UINT: return asset::E_FORMAT::EF_R16G16_UINT;
    case VK_FORMAT_R16G16_SINT: return asset::E_FORMAT::EF_R16G16_SINT;
    case VK_FORMAT_R16G16_SFLOAT: return asset::E_FORMAT::EF_R16G16_SFLOAT;
    case VK_FORMAT_R16G16B16_UNORM: return asset::E_FORMAT::EF_R16G16B16_UNORM;
    case VK_FORMAT_R16G16B16_SNORM: return asset::E_FORMAT::EF_R16G16B16_SNORM;
    case VK_FORMAT_R16G16B16_USCALED: return asset::E_FORMAT::EF_R16G16B16_USCALED;
    case VK_FORMAT_R16G16B16_SSCALED: return asset::E_FORMAT::EF_R16G16B16_SSCALED;
    case VK_FORMAT_R16G16B16_UINT: return asset::E_FORMAT::EF_R16G16B16_UINT;
    case VK_FORMAT_R16G16B16_SINT: return asset::E_FORMAT::EF_R16G16B16_SINT;
    case VK_FORMAT_R16G16B16_SFLOAT: return asset::E_FORMAT::EF_R16G16B16_SFLOAT;
    case VK_FORMAT_R16G16B16A16_UNORM: return asset::E_FORMAT::EF_R16G16B16A16_UNORM;
    case VK_FORMAT_R16G16B16A16_SNORM: return asset::E_FORMAT::EF_R16G16B16A16_SNORM;
    case VK_FORMAT_R16G16B16A16_USCALED: return asset::E_FORMAT::EF_R16G16B16A16_USCALED;
    case VK_FORMAT_R16G16B16A16_SSCALED: return asset::E_FORMAT::EF_R16G16B16A16_SSCALED;
    case VK_FORMAT_R16G16B16A16_UINT: return asset::E_FORMAT::EF_R16G16B16A16_UINT;
    case VK_FORMAT_R16G16B16A16_SINT: return asset::E_FORMAT::EF_R16G16B16A16_SINT;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return asset::E_FORMAT::EF_R16G16B16A16_SFLOAT;
    case VK_FORMAT_R32_UINT: return asset::E_FORMAT::EF_R32_UINT;
    case VK_FORMAT_R32_SINT: return asset::E_FORMAT::EF_R32_SINT;
    case VK_FORMAT_R32_SFLOAT: return asset::E_FORMAT::EF_R32_SFLOAT;
    case VK_FORMAT_R32G32_UINT: return asset::E_FORMAT::EF_R32G32_UINT;
    case VK_FORMAT_R32G32_SINT: return asset::E_FORMAT::EF_R32G32_SINT;
    case VK_FORMAT_R32G32_SFLOAT: return asset::E_FORMAT::EF_R32G32_SFLOAT;
    case VK_FORMAT_R32G32B32_UINT: return asset::E_FORMAT::EF_R32G32B32_UINT;
    case VK_FORMAT_R32G32B32_SINT: return asset::E_FORMAT::EF_R32G32B32_SINT;
    case VK_FORMAT_R32G32B32_SFLOAT: return asset::E_FORMAT::EF_R32G32B32_SFLOAT;
    case VK_FORMAT_R32G32B32A32_UINT: return asset::E_FORMAT::EF_R32G32B32A32_UINT;
    case VK_FORMAT_R32G32B32A32_SINT: return asset::E_FORMAT::EF_R32G32B32A32_SINT;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
    case VK_FORMAT_R64_UINT: return asset::E_FORMAT::EF_R64_UINT;
    case VK_FORMAT_R64_SINT: return asset::E_FORMAT::EF_R64_SINT;
    case VK_FORMAT_R64_SFLOAT: return asset::E_FORMAT::EF_R64_SFLOAT;
    case VK_FORMAT_R64G64_UINT: return asset::E_FORMAT::EF_R64G64_UINT;
    case VK_FORMAT_R64G64_SINT: return asset::E_FORMAT::EF_R64G64_SINT;
    case VK_FORMAT_R64G64_SFLOAT: return asset::E_FORMAT::EF_R64G64_SFLOAT;
    case VK_FORMAT_R64G64B64_UINT: return asset::E_FORMAT::EF_R64G64B64_UINT;
    case VK_FORMAT_R64G64B64_SINT: return asset::E_FORMAT::EF_R64G64B64_SINT;
    case VK_FORMAT_R64G64B64_SFLOAT: return asset::E_FORMAT::EF_R64G64B64_SFLOAT;
    case VK_FORMAT_R64G64B64A64_UINT: return asset::E_FORMAT::EF_R64G64B64A64_UINT;
    case VK_FORMAT_R64G64B64A64_SINT: return asset::E_FORMAT::EF_R64G64B64A64_SINT;
    case VK_FORMAT_R64G64B64A64_SFLOAT: return asset::E_FORMAT::EF_R64G64B64A64_SFLOAT;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return asset::E_FORMAT::EF_B10G11R11_UFLOAT_PACK32;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return asset::E_FORMAT::EF_E5B9G9R9_UFLOAT_PACK32;
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK: return asset::E_FORMAT::EF_BC1_RGB_UNORM_BLOCK;
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return asset::E_FORMAT::EF_BC1_RGB_SRGB_BLOCK;
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return asset::E_FORMAT::EF_BC1_RGBA_UNORM_BLOCK;
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return asset::E_FORMAT::EF_BC1_RGBA_SRGB_BLOCK;
    case VK_FORMAT_BC2_UNORM_BLOCK: return asset::E_FORMAT::EF_BC2_UNORM_BLOCK;
    case VK_FORMAT_BC2_SRGB_BLOCK: return asset::E_FORMAT::EF_BC2_SRGB_BLOCK;
    case VK_FORMAT_BC3_UNORM_BLOCK: return asset::E_FORMAT::EF_BC3_UNORM_BLOCK;
    case VK_FORMAT_BC3_SRGB_BLOCK: return asset::E_FORMAT::EF_BC3_SRGB_BLOCK;
    case VK_FORMAT_BC4_UNORM_BLOCK: return asset::E_FORMAT::EF_BC4_UNORM_BLOCK;
    case VK_FORMAT_BC4_SNORM_BLOCK: return asset::E_FORMAT::EF_BC4_SNORM_BLOCK;
    case VK_FORMAT_BC5_UNORM_BLOCK: return asset::E_FORMAT::EF_BC5_UNORM_BLOCK;
    case VK_FORMAT_BC5_SNORM_BLOCK: return asset::E_FORMAT::EF_BC5_SNORM_BLOCK;
    case VK_FORMAT_BC6H_UFLOAT_BLOCK: return asset::E_FORMAT::EF_BC6H_UFLOAT_BLOCK;
    case VK_FORMAT_BC6H_SFLOAT_BLOCK: return asset::E_FORMAT::EF_BC6H_SFLOAT_BLOCK;
    case VK_FORMAT_BC7_UNORM_BLOCK: return asset::E_FORMAT::EF_BC7_UNORM_BLOCK;
    case VK_FORMAT_BC7_SRGB_BLOCK: return asset::E_FORMAT::EF_BC7_SRGB_BLOCK;
    case VK_FORMAT_ASTC_4x4_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_4x4_UNORM_BLOCK;
    case VK_FORMAT_ASTC_4x4_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_4x4_SRGB_BLOCK;
    case VK_FORMAT_ASTC_5x4_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_5x4_UNORM_BLOCK;
    case VK_FORMAT_ASTC_5x4_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_5x4_SRGB_BLOCK;
    case VK_FORMAT_ASTC_5x5_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_5x5_UNORM_BLOCK;
    case VK_FORMAT_ASTC_5x5_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_5x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_6x5_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_6x5_UNORM_BLOCK;
    case VK_FORMAT_ASTC_6x5_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_6x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_6x6_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_6x6_UNORM_BLOCK;
    case VK_FORMAT_ASTC_6x6_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_6x6_SRGB_BLOCK;
    case VK_FORMAT_ASTC_8x5_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_8x5_UNORM_BLOCK;
    case VK_FORMAT_ASTC_8x5_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_8x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_8x6_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_8x6_UNORM_BLOCK;
    case VK_FORMAT_ASTC_8x6_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_8x6_SRGB_BLOCK;
    case VK_FORMAT_ASTC_8x8_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_8x8_UNORM_BLOCK;
    case VK_FORMAT_ASTC_8x8_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_8x8_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x5_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_10x5_UNORM_BLOCK;
    case VK_FORMAT_ASTC_10x5_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_10x5_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x6_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_10x6_UNORM_BLOCK;
    case VK_FORMAT_ASTC_10x6_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_10x6_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x8_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_10x8_UNORM_BLOCK;
    case VK_FORMAT_ASTC_10x8_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_10x8_SRGB_BLOCK;
    case VK_FORMAT_ASTC_10x10_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_10x10_UNORM_BLOCK;
    case VK_FORMAT_ASTC_10x10_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_10x10_SRGB_BLOCK;
    case VK_FORMAT_ASTC_12x10_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_12x10_UNORM_BLOCK;
    case VK_FORMAT_ASTC_12x10_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_12x10_SRGB_BLOCK;
    case VK_FORMAT_ASTC_12x12_UNORM_BLOCK: return asset::E_FORMAT::EF_ASTC_12x12_UNORM_BLOCK;
    case VK_FORMAT_ASTC_12x12_SRGB_BLOCK: return asset::E_FORMAT::EF_ASTC_12x12_SRGB_BLOCK;
    case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8_UNORM_BLOCK;
    case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8_SRGB_BLOCK;
    case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8A1_UNORM_BLOCK;
    case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8A1_SRGB_BLOCK;
    case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8A8_UNORM_BLOCK;
    case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK: return asset::E_FORMAT::EF_ETC2_R8G8B8A8_SRGB_BLOCK;
    case VK_FORMAT_EAC_R11_UNORM_BLOCK: return asset::E_FORMAT::EF_EAC_R11_UNORM_BLOCK;
    case VK_FORMAT_EAC_R11_SNORM_BLOCK: return asset::E_FORMAT::EF_EAC_R11_SNORM_BLOCK;
    case VK_FORMAT_EAC_R11G11_UNORM_BLOCK: return asset::E_FORMAT::EF_EAC_R11G11_UNORM_BLOCK;
    case VK_FORMAT_EAC_R11G11_SNORM_BLOCK: return asset::E_FORMAT::EF_EAC_R11G11_SNORM_BLOCK;
    case VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC1_2BPP_UNORM_BLOCK_IMG;
    case VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC1_4BPP_UNORM_BLOCK_IMG;
    case VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC2_2BPP_UNORM_BLOCK_IMG;
    case VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC2_4BPP_UNORM_BLOCK_IMG;
    case VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC1_2BPP_SRGB_BLOCK_IMG;
    case VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC1_4BPP_SRGB_BLOCK_IMG;
    case VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC2_2BPP_SRGB_BLOCK_IMG;
    case VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG: return asset::E_FORMAT::EF_PVRTC2_4BPP_SRGB_BLOCK_IMG;
    case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM: return asset::E_FORMAT::EF_G8_B8_R8_3PLANE_420_UNORM;
    case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM: return asset::E_FORMAT::EF_G8_B8R8_2PLANE_420_UNORM;
    case VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM: return asset::E_FORMAT::EF_G8_B8_R8_3PLANE_422_UNORM;
    case VK_FORMAT_G8_B8R8_2PLANE_422_UNORM: return asset::E_FORMAT::EF_G8_B8R8_2PLANE_422_UNORM;
    case VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM: return asset::E_FORMAT::EF_G8_B8_R8_3PLANE_444_UNORM;
    default:
        return asset::E_FORMAT::EF_UNKNOWN;
    }
}

inline ISurface::SColorSpace getColorSpaceFromVkColorSpaceKHR(VkColorSpaceKHR in)
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

inline ISurface::E_PRESENT_MODE getPresentModeFromVkPresentModeKHR(VkPresentModeKHR in)
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

//
inline VkPipelineStageFlagBits2 getVkPipelineStageFlagsFromPipelineStageFlags(core::bitflag<asset::PIPELINE_STAGE_FLAGS> stages)
{
    VkPipelineStageFlagBits2 retval = VK_PIPELINE_STAGE_2_NONE;
    using stage_flags_t = asset::PIPELINE_STAGE_FLAGS;
    // we want to "recoup" general flags first because they don't check capabilities for individual bits
    auto stripCompoundFlags = [&stages,&retval](const core::bitflag<stage_flags_t> compound, const VkPipelineStageFlagBits2 vkFlag)
    {
        if (stages.hasFlags(compound))
        {
            retval |= vkFlag;
            stages &= ~compound;
        }
    };
    stripCompoundFlags(stage_flags_t::ALL_COMMANDS_BITS,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    stripCompoundFlags(stage_flags_t::ALL_TRANSFER_BITS,VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
    stripCompoundFlags(stage_flags_t::ALL_GRAPHICS_BITS,VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
    stripCompoundFlags(stage_flags_t::PRE_RASTERIZATION_SHADERS_BITS,VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT);

    // now proceed normally
    if (stages.hasFlags(stage_flags_t::HOST_BIT)) retval |= VK_PIPELINE_STAGE_2_HOST_BIT;
    if (stages.hasFlags(stage_flags_t::COPY_BIT)) retval |= VK_PIPELINE_STAGE_2_COPY_BIT;
    if (stages.hasFlags(stage_flags_t::CLEAR_BIT)) retval |= VK_PIPELINE_STAGE_2_CLEAR_BIT;
    //    if (stages.hasFlags(stage_flags_t::MICROMAP_BUILD_BIT)) retval |= VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
    if (stages.hasFlags(stage_flags_t::ACCELERATION_STRUCTURE_COPY_BIT)) retval |= VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR;
    if (stages.hasFlags(stage_flags_t::ACCELERATION_STRUCTURE_BUILD_BIT)) retval |= VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    if (stages.hasFlags(stage_flags_t::COMMAND_PREPROCESS_BIT)) retval |= VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_NV;
    if (stages.hasFlags(stage_flags_t::CONDITIONAL_RENDERING_BIT)) retval |= VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT;
    if (stages.hasFlags(stage_flags_t::DISPATCH_INDIRECT_COMMAND_BIT)) retval |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    if (stages.hasFlags(stage_flags_t::COMPUTE_SHADER_BIT)) retval |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    if (stages.hasFlags(stage_flags_t::INDEX_INPUT_BIT)) retval |= VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
    if (stages.hasFlags(stage_flags_t::VERTEX_ATTRIBUTE_INPUT_BIT)) retval |= VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
    if (stages.hasFlags(stage_flags_t::VERTEX_SHADER_BIT)) retval |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    if (stages.hasFlags(stage_flags_t::TESSELLATION_CONTROL_SHADER_BIT)) retval |= VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT;
    if (stages.hasFlags(stage_flags_t::TESSELLATION_EVALUATION_SHADER_BIT)) retval |= VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT;
    if (stages.hasFlags(stage_flags_t::GEOMETRY_SHADER_BIT)) retval |= VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT;
//    if (stages.hasFlags(stage_flags_t::TASK_SHADER_BIT)) retval |= VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
//    if (stages.hasFlags(stage_flags_t::MESH_SHADER_BIT)) retval |= VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    if (stages.hasFlags(stage_flags_t::FRAGMENT_DENSITY_PROCESS_BIT)) retval |= VK_PIPELINE_STAGE_2_FRAGMENT_DENSITY_PROCESS_BIT_EXT;
    if (stages.hasFlags(stage_flags_t::SHADING_RATE_ATTACHMENT_BIT)) retval |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
    if (stages.hasFlags(stage_flags_t::EARLY_FRAGMENT_TESTS_BIT)) retval |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
    if (stages.hasFlags(stage_flags_t::FRAGMENT_SHADER_BIT)) retval |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    if (stages.hasFlags(stage_flags_t::LATE_FRAGMENT_TESTS_BIT)) retval |= VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    if (stages.hasFlags(stage_flags_t::COLOR_ATTACHMENT_OUTPUT_BIT)) retval |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (stages.hasFlags(stage_flags_t::RAY_TRACING_SHADER_BIT)) retval |= VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    if (stages.hasFlags(stage_flags_t::RESOLVE_BIT)) retval |= VK_PIPELINE_STAGE_2_RESOLVE_BIT;
    if (stages.hasFlags(stage_flags_t::BLIT_BIT)) retval |= VK_PIPELINE_STAGE_2_BLIT_BIT;
//    if (stages.hasFlags(stage_flags_t::VIDEO_DECODE)) retval |= VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
//    if (stages.hasFlags(stage_flags_t::VIDEO_ENCODE)) retval |= VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
//    if (stages.hasFlags(stage_flags_t::OPTICAL_FLOW)) retval |= VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV;

    return retval;
}

inline VkAccessFlagBits2 getVkAccessFlagsFromAccessFlags(core::bitflag<asset::ACCESS_FLAGS> accesses)
{
    VkAccessFlagBits2 retval = VK_ACCESS_2_NONE;
    using access_flags_t = asset::ACCESS_FLAGS;
    // we want to "recoup" general flags first because they don't check capabilities for individual bits
    auto stripCompoundFlags = [&accesses,&retval](const core::bitflag<access_flags_t> compound, const VkAccessFlagBits2 vkFlag)
    {
        if (accesses.hasFlags(compound))
        {
            retval |= vkFlag;
            accesses &= ~compound;
        }
    };
    stripCompoundFlags(access_flags_t::MEMORY_READ_BITS,VK_ACCESS_2_MEMORY_READ_BIT);
    stripCompoundFlags(access_flags_t::MEMORY_WRITE_BITS,VK_ACCESS_2_MEMORY_WRITE_BIT);
    stripCompoundFlags(access_flags_t::SHADER_READ_BITS,VK_ACCESS_2_SHADER_READ_BIT);
    stripCompoundFlags(access_flags_t::SHADER_WRITE_BITS,VK_ACCESS_2_SHADER_WRITE_BIT);

    // now proceed normally
    if (accesses.hasFlags(access_flags_t::HOST_READ_BIT)) retval |= VK_ACCESS_2_HOST_READ_BIT;
    if (accesses.hasFlags(access_flags_t::HOST_WRITE_BIT)) retval |= VK_ACCESS_2_HOST_WRITE_BIT;
    if (accesses.hasFlags(access_flags_t::TRANSFER_READ_BIT)) retval |= VK_ACCESS_2_TRANSFER_READ_BIT;
    if (accesses.hasFlags(access_flags_t::TRANSFER_WRITE_BIT)) retval |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
//    if (accesses.hasFlags(access_flags_t::MICROMAP_READ_BIT)) retval |= VK_ACCESS_2_MICROMAP_READ_BIT_EXT;
//    if (accesses.hasFlags(access_flags_t::MICROMAP_WRITE_BIT)) retval |= VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT;
    if (accesses.hasFlags(access_flags_t::ACCELERATION_STRUCTURE_READ_BIT)) retval |= VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    if (accesses.hasFlags(access_flags_t::ACCELERATION_STRUCTURE_WRITE_BIT)) retval |= VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    if (accesses.hasFlags(access_flags_t::COMMAND_PREPROCESS_READ_BIT)) retval |= VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV;
    if (accesses.hasFlags(access_flags_t::COMMAND_PREPROCESS_WRITE_BIT)) retval |= VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_NV;
    if (accesses.hasFlags(access_flags_t::CONDITIONAL_RENDERING_READ_BIT)) retval |= VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT;
    if (accesses.hasFlags(access_flags_t::INDIRECT_COMMAND_READ_BIT)) retval |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    if (accesses.hasFlags(access_flags_t::UNIFORM_READ_BIT)) retval |= VK_ACCESS_2_UNIFORM_READ_BIT;
    if (accesses.hasFlags(access_flags_t::SAMPLED_READ_BIT)) retval |= VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    if (accesses.hasFlags(access_flags_t::STORAGE_READ_BIT)) retval |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    if (accesses.hasFlags(access_flags_t::STORAGE_WRITE_BIT)) retval |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    if (accesses.hasFlags(access_flags_t::INDEX_READ_BIT)) retval |= VK_ACCESS_2_INDEX_READ_BIT;
    if (accesses.hasFlags(access_flags_t::VERTEX_ATTRIBUTE_READ_BIT)) retval |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    if (accesses.hasFlags(access_flags_t::FRAGMENT_DENSITY_MAP_READ_BIT)) retval |= VK_ACCESS_2_FRAGMENT_DENSITY_MAP_READ_BIT_EXT;
    if (accesses.hasFlags(access_flags_t::SHADING_RATE_ATTACHMENT_READ_BIT)) retval |= VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR;
    if (accesses.hasFlags(access_flags_t::INPUT_ATTACHMENT_READ_BIT)) retval |= VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT;
    if (accesses.hasFlags(access_flags_t::DEPTH_STENCIL_ATTACHMENT_READ_BIT)) retval |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    if (accesses.hasFlags(access_flags_t::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) retval |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    if (accesses.hasFlags(access_flags_t::COLOR_ATTACHMENT_READ_BIT)) retval |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
    if (accesses.hasFlags(access_flags_t::COLOR_ATTACHMENT_WRITE_BIT)) retval |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    if (accesses.hasFlags(access_flags_t::SHADER_BINDING_TABLE_READ_BIT)) retval |= VK_ACCESS_2_SHADER_BINDING_TABLE_READ_BIT_KHR;
//    if (accesses.hasFlags(access_flags_t::VIDEO_DECODE_READ_BIT)) retval |= VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR;
//    if (accesses.hasFlags(access_flags_t::VIDEO_DECODE_WRITE_BIT)) retval |= VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
//    if (accesses.hasFlags(access_flags_t::VIDEO_ENCODE_READ_BIT)) retval |= VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR;
//    if (accesses.hasFlags(access_flags_t::VIDEO_ENCODE_WRITE_BIT)) retval |= VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR;
//    if (accesses.hasFlags(access_flags_t::OPTICAL_FLOW_READ_BIT)) retval |= VK_ACCESS_2_OPTICAL_FLOW_READ_BIT_NV;
//    if (accesses.hasFlags(access_flags_t::OPTICAL_FLOW_WRITE_BIT)) retval |= VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV;

    return retval;
}

inline VkShaderStageFlags getVkShaderStageFlagsFromShaderStage(const core::bitflag<IGPUShader::E_SHADER_STAGE> in)
{
    VkShaderStageFlags ret = 0u;
    if(in.hasFlags(IGPUShader::ESS_VERTEX)) ret |= VK_SHADER_STAGE_VERTEX_BIT;
    if(in.hasFlags(IGPUShader::ESS_TESSELLATION_CONTROL)) ret |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    if(in.hasFlags(IGPUShader::ESS_TESSELLATION_EVALUATION)) ret |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    if(in.hasFlags(IGPUShader::ESS_GEOMETRY)) ret |= VK_SHADER_STAGE_GEOMETRY_BIT;
    if(in.hasFlags(IGPUShader::ESS_FRAGMENT)) ret |= VK_SHADER_STAGE_FRAGMENT_BIT;
    if(in.hasFlags(IGPUShader::ESS_COMPUTE)) ret |= VK_SHADER_STAGE_COMPUTE_BIT;
    if(in.hasFlags(IGPUShader::ESS_TASK)) ret |= VK_SHADER_STAGE_TASK_BIT_NV;
    if(in.hasFlags(IGPUShader::ESS_MESH)) ret |= VK_SHADER_STAGE_MESH_BIT_NV;
    if(in.hasFlags(IGPUShader::ESS_RAYGEN)) ret |= VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    if(in.hasFlags(IGPUShader::ESS_ANY_HIT)) ret |= VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    if(in.hasFlags(IGPUShader::ESS_CLOSEST_HIT)) ret |= VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    if(in.hasFlags(IGPUShader::ESS_MISS)) ret |= VK_SHADER_STAGE_MISS_BIT_KHR;
    if(in.hasFlags(IGPUShader::ESS_INTERSECTION)) ret |= VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    if(in.hasFlags(IGPUShader::ESS_CALLABLE)) ret |= VK_SHADER_STAGE_CALLABLE_BIT_KHR;
    if(in.hasFlags(IGPUShader::ESS_ALL_GRAPHICS)) ret |= VK_SHADER_STAGE_ALL_GRAPHICS;
    if(in.hasFlags(IGPUShader::ESS_ALL)) ret |= VK_SHADER_STAGE_ALL;
    return ret;
}

inline VkFormat getVkFormatFromFormat(asset::E_FORMAT in)
{
    switch (in)
    {
    case asset::E_FORMAT::EF_D16_UNORM: return VK_FORMAT_D16_UNORM;
    case asset::E_FORMAT::EF_X8_D24_UNORM_PACK32: return VK_FORMAT_X8_D24_UNORM_PACK32;
    case asset::E_FORMAT::EF_D32_SFLOAT: return VK_FORMAT_D32_SFLOAT;
    case asset::E_FORMAT::EF_S8_UINT: return VK_FORMAT_S8_UINT;
    case asset::E_FORMAT::EF_D16_UNORM_S8_UINT: return VK_FORMAT_D16_UNORM_S8_UINT;
    case asset::E_FORMAT::EF_D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
    case asset::E_FORMAT::EF_D32_SFLOAT_S8_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
    case asset::E_FORMAT::EF_R4G4_UNORM_PACK8: return VK_FORMAT_R4G4_UNORM_PACK8;
    case asset::E_FORMAT::EF_R4G4B4A4_UNORM_PACK16: return VK_FORMAT_R4G4B4A4_UNORM_PACK16;
    case asset::E_FORMAT::EF_B4G4R4A4_UNORM_PACK16: return VK_FORMAT_B4G4R4A4_UNORM_PACK16;
    case asset::E_FORMAT::EF_R5G6B5_UNORM_PACK16: return VK_FORMAT_R5G6B5_UNORM_PACK16;
    case asset::E_FORMAT::EF_B5G6R5_UNORM_PACK16: return VK_FORMAT_B5G6R5_UNORM_PACK16;
    case asset::E_FORMAT::EF_R5G5B5A1_UNORM_PACK16: return VK_FORMAT_R5G5B5A1_UNORM_PACK16;
    case asset::E_FORMAT::EF_B5G5R5A1_UNORM_PACK16: return VK_FORMAT_B5G5R5A1_UNORM_PACK16;
    case asset::E_FORMAT::EF_A1R5G5B5_UNORM_PACK16: return VK_FORMAT_A1R5G5B5_UNORM_PACK16;
    case asset::E_FORMAT::EF_R8_UNORM: return VK_FORMAT_R8_UNORM;
    case asset::E_FORMAT::EF_R8_SNORM: return VK_FORMAT_R8_SNORM;
    case asset::E_FORMAT::EF_R8_USCALED: return VK_FORMAT_R8_USCALED;
    case asset::E_FORMAT::EF_R8_SSCALED: return VK_FORMAT_R8_SSCALED;
    case asset::E_FORMAT::EF_R8_UINT: return VK_FORMAT_R8_UINT;
    case asset::E_FORMAT::EF_R8_SINT: return VK_FORMAT_R8_SINT;
    case asset::E_FORMAT::EF_R8_SRGB: return VK_FORMAT_R8_SRGB;
    case asset::E_FORMAT::EF_R8G8_UNORM: return VK_FORMAT_R8G8_UNORM;
    case asset::E_FORMAT::EF_R8G8_SNORM: return VK_FORMAT_R8G8_SNORM;
    case asset::E_FORMAT::EF_R8G8_USCALED: return VK_FORMAT_R8G8_USCALED;
    case asset::E_FORMAT::EF_R8G8_SSCALED: return VK_FORMAT_R8G8_SSCALED;
    case asset::E_FORMAT::EF_R8G8_UINT: return VK_FORMAT_R8G8_UINT;
    case asset::E_FORMAT::EF_R8G8_SINT: return VK_FORMAT_R8G8_SINT;
    case asset::E_FORMAT::EF_R8G8_SRGB: return VK_FORMAT_R8G8_SRGB;
    case asset::E_FORMAT::EF_R8G8B8_UNORM: return VK_FORMAT_R8G8B8_UNORM;
    case asset::E_FORMAT::EF_R8G8B8_SNORM: return VK_FORMAT_R8G8B8_SNORM;
    case asset::E_FORMAT::EF_R8G8B8_USCALED: return VK_FORMAT_R8G8B8_USCALED;
    case asset::E_FORMAT::EF_R8G8B8_SSCALED: return VK_FORMAT_R8G8B8_SSCALED;
    case asset::E_FORMAT::EF_R8G8B8_UINT: return VK_FORMAT_R8G8B8_UINT;
    case asset::E_FORMAT::EF_R8G8B8_SINT: return VK_FORMAT_R8G8B8_SINT;
    case asset::E_FORMAT::EF_R8G8B8_SRGB: return VK_FORMAT_R8G8B8_SRGB;
    case asset::E_FORMAT::EF_B8G8R8_UNORM: return VK_FORMAT_B8G8R8_UNORM;
    case asset::E_FORMAT::EF_B8G8R8_SNORM: return VK_FORMAT_B8G8R8_SNORM;
    case asset::E_FORMAT::EF_B8G8R8_USCALED: return VK_FORMAT_B8G8R8_USCALED;
    case asset::E_FORMAT::EF_B8G8R8_SSCALED: return VK_FORMAT_B8G8R8_SSCALED;
    case asset::E_FORMAT::EF_B8G8R8_UINT: return VK_FORMAT_B8G8R8_UINT;
    case asset::E_FORMAT::EF_B8G8R8_SINT: return VK_FORMAT_B8G8R8_SINT;
    case asset::E_FORMAT::EF_B8G8R8_SRGB: return VK_FORMAT_B8G8R8_SRGB;
    case asset::E_FORMAT::EF_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case asset::E_FORMAT::EF_R8G8B8A8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case asset::E_FORMAT::EF_R8G8B8A8_USCALED: return VK_FORMAT_R8G8B8A8_USCALED;
    case asset::E_FORMAT::EF_R8G8B8A8_SSCALED: return VK_FORMAT_R8G8B8A8_SSCALED;
    case asset::E_FORMAT::EF_R8G8B8A8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case asset::E_FORMAT::EF_R8G8B8A8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case asset::E_FORMAT::EF_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
    case asset::E_FORMAT::EF_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
    case asset::E_FORMAT::EF_B8G8R8A8_SNORM: return VK_FORMAT_B8G8R8A8_SNORM;
    case asset::E_FORMAT::EF_B8G8R8A8_USCALED: return VK_FORMAT_B8G8R8A8_USCALED;
    case asset::E_FORMAT::EF_B8G8R8A8_SSCALED: return VK_FORMAT_B8G8R8A8_SSCALED;
    case asset::E_FORMAT::EF_B8G8R8A8_UINT: return VK_FORMAT_B8G8R8A8_UINT;
    case asset::E_FORMAT::EF_B8G8R8A8_SINT: return VK_FORMAT_B8G8R8A8_SINT;
    case asset::E_FORMAT::EF_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
    case asset::E_FORMAT::EF_A8B8G8R8_UNORM_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    case asset::E_FORMAT::EF_A8B8G8R8_SNORM_PACK32: return VK_FORMAT_A8B8G8R8_SNORM_PACK32;
    case asset::E_FORMAT::EF_A8B8G8R8_USCALED_PACK32: return VK_FORMAT_A8B8G8R8_USCALED_PACK32;
    case asset::E_FORMAT::EF_A8B8G8R8_SSCALED_PACK32: return VK_FORMAT_A8B8G8R8_SSCALED_PACK32;
    case asset::E_FORMAT::EF_A8B8G8R8_UINT_PACK32: return VK_FORMAT_A8B8G8R8_UINT_PACK32;
    case asset::E_FORMAT::EF_A8B8G8R8_SINT_PACK32: return VK_FORMAT_A8B8G8R8_SINT_PACK32;
    case asset::E_FORMAT::EF_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_SRGB_PACK32;
    case asset::E_FORMAT::EF_A2R10G10B10_UNORM_PACK32: return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    case asset::E_FORMAT::EF_A2R10G10B10_SNORM_PACK32: return VK_FORMAT_A2R10G10B10_SNORM_PACK32;
    case asset::E_FORMAT::EF_A2R10G10B10_USCALED_PACK32: return VK_FORMAT_A2R10G10B10_USCALED_PACK32;
    case asset::E_FORMAT::EF_A2R10G10B10_SSCALED_PACK32: return VK_FORMAT_A2R10G10B10_SSCALED_PACK32;
    case asset::E_FORMAT::EF_A2R10G10B10_UINT_PACK32: return VK_FORMAT_A2R10G10B10_UINT_PACK32;
    case asset::E_FORMAT::EF_A2R10G10B10_SINT_PACK32: return VK_FORMAT_A2R10G10B10_SINT_PACK32;
    case asset::E_FORMAT::EF_A2B10G10R10_UNORM_PACK32: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case asset::E_FORMAT::EF_A2B10G10R10_SNORM_PACK32: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    case asset::E_FORMAT::EF_A2B10G10R10_USCALED_PACK32: return VK_FORMAT_A2B10G10R10_USCALED_PACK32;
    case asset::E_FORMAT::EF_A2B10G10R10_SSCALED_PACK32: return VK_FORMAT_A2B10G10R10_SSCALED_PACK32;
    case asset::E_FORMAT::EF_A2B10G10R10_UINT_PACK32: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case asset::E_FORMAT::EF_A2B10G10R10_SINT_PACK32: return VK_FORMAT_A2B10G10R10_SINT_PACK32;
    case asset::E_FORMAT::EF_R16_UNORM: return VK_FORMAT_R16_UNORM;
    case asset::E_FORMAT::EF_R16_SNORM: return VK_FORMAT_R16_SNORM;
    case asset::E_FORMAT::EF_R16_USCALED: return VK_FORMAT_R16_USCALED;
    case asset::E_FORMAT::EF_R16_SSCALED: return VK_FORMAT_R16_SSCALED;
    case asset::E_FORMAT::EF_R16_UINT: return VK_FORMAT_R16_UINT;
    case asset::E_FORMAT::EF_R16_SINT: return VK_FORMAT_R16_SINT;
    case asset::E_FORMAT::EF_R16_SFLOAT: return VK_FORMAT_R16_SFLOAT;
    case asset::E_FORMAT::EF_R16G16_UNORM: return VK_FORMAT_R16G16_UNORM;
    case asset::E_FORMAT::EF_R16G16_SNORM: return VK_FORMAT_R16G16_SNORM;
    case asset::E_FORMAT::EF_R16G16_USCALED: return VK_FORMAT_R16G16_USCALED;
    case asset::E_FORMAT::EF_R16G16_SSCALED: return VK_FORMAT_R16G16_SSCALED;
    case asset::E_FORMAT::EF_R16G16_UINT: return VK_FORMAT_R16G16_UINT;
    case asset::E_FORMAT::EF_R16G16_SINT: return VK_FORMAT_R16G16_SINT;
    case asset::E_FORMAT::EF_R16G16_SFLOAT: return VK_FORMAT_R16G16_SFLOAT;
    case asset::E_FORMAT::EF_R16G16B16_UNORM: return VK_FORMAT_R16G16B16_UNORM;
    case asset::E_FORMAT::EF_R16G16B16_SNORM: return VK_FORMAT_R16G16B16_SNORM;
    case asset::E_FORMAT::EF_R16G16B16_USCALED: return VK_FORMAT_R16G16B16_USCALED;
    case asset::E_FORMAT::EF_R16G16B16_SSCALED: return VK_FORMAT_R16G16B16_SSCALED;
    case asset::E_FORMAT::EF_R16G16B16_UINT: return VK_FORMAT_R16G16B16_UINT;
    case asset::E_FORMAT::EF_R16G16B16_SINT: return VK_FORMAT_R16G16B16_SINT;
    case asset::E_FORMAT::EF_R16G16B16_SFLOAT: return VK_FORMAT_R16G16B16_SFLOAT;
    case asset::E_FORMAT::EF_R16G16B16A16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case asset::E_FORMAT::EF_R16G16B16A16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case asset::E_FORMAT::EF_R16G16B16A16_USCALED: return VK_FORMAT_R16G16B16A16_USCALED;
    case asset::E_FORMAT::EF_R16G16B16A16_SSCALED: return VK_FORMAT_R16G16B16A16_SSCALED;
    case asset::E_FORMAT::EF_R16G16B16A16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case asset::E_FORMAT::EF_R16G16B16A16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    case asset::E_FORMAT::EF_R16G16B16A16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case asset::E_FORMAT::EF_R32_UINT: return VK_FORMAT_R32_UINT;
    case asset::E_FORMAT::EF_R32_SINT: return VK_FORMAT_R32_SINT;
    case asset::E_FORMAT::EF_R32_SFLOAT: return VK_FORMAT_R32_SFLOAT;
    case asset::E_FORMAT::EF_R32G32_UINT: return VK_FORMAT_R32G32_UINT;
    case asset::E_FORMAT::EF_R32G32_SINT: return VK_FORMAT_R32G32_SINT;
    case asset::E_FORMAT::EF_R32G32_SFLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case asset::E_FORMAT::EF_R32G32B32_UINT: return VK_FORMAT_R32G32B32_UINT;
    case asset::E_FORMAT::EF_R32G32B32_SINT: return VK_FORMAT_R32G32B32_SINT;
    case asset::E_FORMAT::EF_R32G32B32_SFLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    case asset::E_FORMAT::EF_R32G32B32A32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
    case asset::E_FORMAT::EF_R32G32B32A32_SINT: return VK_FORMAT_R32G32B32A32_SINT;
    case asset::E_FORMAT::EF_R32G32B32A32_SFLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case asset::E_FORMAT::EF_R64_UINT: return VK_FORMAT_R64_UINT;
    case asset::E_FORMAT::EF_R64_SINT: return VK_FORMAT_R64_SINT;
    case asset::E_FORMAT::EF_R64_SFLOAT: return VK_FORMAT_R64_SFLOAT;
    case asset::E_FORMAT::EF_R64G64_UINT: return VK_FORMAT_R64G64_UINT;
    case asset::E_FORMAT::EF_R64G64_SINT: return VK_FORMAT_R64G64_SINT;
    case asset::E_FORMAT::EF_R64G64_SFLOAT: return VK_FORMAT_R64G64_SFLOAT;
    case asset::E_FORMAT::EF_R64G64B64_UINT: return VK_FORMAT_R64G64B64_UINT;
    case asset::E_FORMAT::EF_R64G64B64_SINT: return VK_FORMAT_R64G64B64_SINT;
    case asset::E_FORMAT::EF_R64G64B64_SFLOAT: return VK_FORMAT_R64G64B64_SFLOAT;
    case asset::E_FORMAT::EF_R64G64B64A64_UINT: return VK_FORMAT_R64G64B64A64_UINT;
    case asset::E_FORMAT::EF_R64G64B64A64_SINT: return VK_FORMAT_R64G64B64A64_SINT;
    case asset::E_FORMAT::EF_R64G64B64A64_SFLOAT: return VK_FORMAT_R64G64B64A64_SFLOAT;
    case asset::E_FORMAT::EF_B10G11R11_UFLOAT_PACK32: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case asset::E_FORMAT::EF_E5B9G9R9_UFLOAT_PACK32: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case asset::E_FORMAT::EF_BC1_RGB_UNORM_BLOCK: return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    case asset::E_FORMAT::EF_BC1_RGB_SRGB_BLOCK: return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
    case asset::E_FORMAT::EF_BC1_RGBA_UNORM_BLOCK: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case asset::E_FORMAT::EF_BC1_RGBA_SRGB_BLOCK: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    case asset::E_FORMAT::EF_BC2_UNORM_BLOCK: return VK_FORMAT_BC2_UNORM_BLOCK;
    case asset::E_FORMAT::EF_BC2_SRGB_BLOCK: return VK_FORMAT_BC2_SRGB_BLOCK;
    case asset::E_FORMAT::EF_BC3_UNORM_BLOCK: return VK_FORMAT_BC3_UNORM_BLOCK;
    case asset::E_FORMAT::EF_BC3_SRGB_BLOCK: return VK_FORMAT_BC3_SRGB_BLOCK;
    case asset::E_FORMAT::EF_BC4_UNORM_BLOCK: return VK_FORMAT_BC4_UNORM_BLOCK;
    case asset::E_FORMAT::EF_BC4_SNORM_BLOCK: return VK_FORMAT_BC4_SNORM_BLOCK;
    case asset::E_FORMAT::EF_BC5_UNORM_BLOCK: return VK_FORMAT_BC5_UNORM_BLOCK;
    case asset::E_FORMAT::EF_BC5_SNORM_BLOCK: return VK_FORMAT_BC5_SNORM_BLOCK;
    case asset::E_FORMAT::EF_BC6H_UFLOAT_BLOCK: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
    case asset::E_FORMAT::EF_BC6H_SFLOAT_BLOCK: return VK_FORMAT_BC6H_SFLOAT_BLOCK;
    case asset::E_FORMAT::EF_BC7_UNORM_BLOCK: return VK_FORMAT_BC7_UNORM_BLOCK;
    case asset::E_FORMAT::EF_BC7_SRGB_BLOCK: return VK_FORMAT_BC7_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_4x4_UNORM_BLOCK: return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_4x4_SRGB_BLOCK: return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_5x4_UNORM_BLOCK: return VK_FORMAT_ASTC_5x4_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_5x4_SRGB_BLOCK: return VK_FORMAT_ASTC_5x4_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_5x5_UNORM_BLOCK: return VK_FORMAT_ASTC_5x5_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_5x5_SRGB_BLOCK: return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_6x5_UNORM_BLOCK: return VK_FORMAT_ASTC_6x5_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_6x5_SRGB_BLOCK: return VK_FORMAT_ASTC_6x5_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_6x6_UNORM_BLOCK: return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_6x6_SRGB_BLOCK: return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_8x5_UNORM_BLOCK: return VK_FORMAT_ASTC_8x5_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_8x5_SRGB_BLOCK: return VK_FORMAT_ASTC_8x5_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_8x6_UNORM_BLOCK: return VK_FORMAT_ASTC_8x6_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_8x6_SRGB_BLOCK: return VK_FORMAT_ASTC_8x6_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_8x8_UNORM_BLOCK: return VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_8x8_SRGB_BLOCK: return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_10x5_UNORM_BLOCK: return VK_FORMAT_ASTC_10x5_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_10x5_SRGB_BLOCK: return VK_FORMAT_ASTC_10x5_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_10x6_UNORM_BLOCK: return VK_FORMAT_ASTC_10x6_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_10x6_SRGB_BLOCK: return VK_FORMAT_ASTC_10x6_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_10x8_UNORM_BLOCK: return VK_FORMAT_ASTC_10x8_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_10x8_SRGB_BLOCK: return VK_FORMAT_ASTC_10x8_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_10x10_UNORM_BLOCK: return VK_FORMAT_ASTC_10x10_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_10x10_SRGB_BLOCK: return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_12x10_UNORM_BLOCK: return VK_FORMAT_ASTC_12x10_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_12x10_SRGB_BLOCK: return VK_FORMAT_ASTC_12x10_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ASTC_12x12_UNORM_BLOCK: return VK_FORMAT_ASTC_12x12_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ASTC_12x12_SRGB_BLOCK: return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ETC2_R8G8B8_UNORM_BLOCK: return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ETC2_R8G8B8_SRGB_BLOCK: return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ETC2_R8G8B8A1_UNORM_BLOCK: return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ETC2_R8G8B8A1_SRGB_BLOCK: return VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;
    case asset::E_FORMAT::EF_ETC2_R8G8B8A8_UNORM_BLOCK: return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
    case asset::E_FORMAT::EF_ETC2_R8G8B8A8_SRGB_BLOCK: return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
    case asset::E_FORMAT::EF_EAC_R11_UNORM_BLOCK: return VK_FORMAT_EAC_R11_UNORM_BLOCK;
    case asset::E_FORMAT::EF_EAC_R11_SNORM_BLOCK: return VK_FORMAT_EAC_R11_SNORM_BLOCK;
    case asset::E_FORMAT::EF_EAC_R11G11_UNORM_BLOCK: return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
    case asset::E_FORMAT::EF_EAC_R11G11_SNORM_BLOCK: return VK_FORMAT_EAC_R11G11_SNORM_BLOCK;
    case asset::E_FORMAT::EF_PVRTC1_2BPP_UNORM_BLOCK_IMG: return VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
    case asset::E_FORMAT::EF_PVRTC1_4BPP_UNORM_BLOCK_IMG: return VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;
    case asset::E_FORMAT::EF_PVRTC2_2BPP_UNORM_BLOCK_IMG: return VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG;
    case asset::E_FORMAT::EF_PVRTC2_4BPP_UNORM_BLOCK_IMG: return VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG;
    case asset::E_FORMAT::EF_PVRTC1_2BPP_SRGB_BLOCK_IMG: return VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;
    case asset::E_FORMAT::EF_PVRTC1_4BPP_SRGB_BLOCK_IMG: return VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;
    case asset::E_FORMAT::EF_PVRTC2_2BPP_SRGB_BLOCK_IMG: return VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG;
    case asset::E_FORMAT::EF_PVRTC2_4BPP_SRGB_BLOCK_IMG: return VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG;
    case asset::E_FORMAT::EF_G8_B8_R8_3PLANE_420_UNORM: return VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
    case asset::E_FORMAT::EF_G8_B8R8_2PLANE_420_UNORM: return VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    case asset::E_FORMAT::EF_G8_B8_R8_3PLANE_422_UNORM: return VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM;
    case asset::E_FORMAT::EF_G8_B8R8_2PLANE_422_UNORM: return VK_FORMAT_G8_B8R8_2PLANE_422_UNORM;
    case asset::E_FORMAT::EF_G8_B8_R8_3PLANE_444_UNORM: return VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM;
    default:
    case asset::E_FORMAT::EF_UNKNOWN:
        return VK_FORMAT_MAX_ENUM;
    }
}

inline VkImageLayout getVkImageLayoutFromImageLayout(asset::IImage::LAYOUT in)
{
    using layout_t = asset::IImage::LAYOUT;
    switch (in)
    {
        case layout_t::UNDEFINED:
            return VK_IMAGE_LAYOUT_UNDEFINED;
        case layout_t::GENERAL:
            return VK_IMAGE_LAYOUT_GENERAL;
        case layout_t::READ_ONLY_OPTIMAL:
            return VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
        case layout_t::ATTACHMENT_OPTIMAL:
            return VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
        case layout_t::TRANSFER_SRC_OPTIMAL:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case layout_t::TRANSFER_DST_OPTIMAL:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        case layout_t::PREINITIALIZED:
            return VK_IMAGE_LAYOUT_PREINITIALIZED;
        case layout_t::PRESENT_SRC:
            return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        case layout_t::SHARED_PRESENT:
            return VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
        default:
            assert(!"IMAGE LAYOUT NOT SUPPORTED!");
            return VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

inline VkColorSpaceKHR getVkColorSpaceKHRFromColorSpace(ISurface::SColorSpace in)
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

inline VkBufferUsageFlags getVkBufferUsageFlagsFromBufferUsageFlags(const core::bitflag<IGPUBuffer::E_USAGE_FLAGS> in)
{
    VkBufferUsageFlags ret = 0u;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT)) ret |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT)) ret |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_UNIFORM_TEXEL_BUFFER_BIT)) ret |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_TEXEL_BUFFER_BIT)) ret |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_UNIFORM_BUFFER_BIT)) ret |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT)) ret |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_INDEX_BUFFER_BIT)) ret |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_VERTEX_BUFFER_BIT)) ret |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_INDIRECT_BUFFER_BIT)) ret |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT)) ret |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    //if(in.hasFlags(asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT)) ret |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    //if(in.hasFlags(asset::IBuffer::E_USAGE_FLAGS::EUF_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT)) ret |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_CONDITIONAL_RENDERING_BIT_EXT)) ret |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT)) ret |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT)) ret |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    if(in.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_BINDING_TABLE_BIT)) ret |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
    return ret;
}

inline VkImageUsageFlags getVkImageUsageFlagsFromImageUsageFlags(const core::bitflag<IGPUImage::E_USAGE_FLAGS> in, const bool depthStencilFormat)
{
    VkImageUsageFlags ret = 0u;
    if (in.hasFlags(IGPUImage::EUF_TRANSFER_SRC_BIT)) ret |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (in.hasFlags(IGPUImage::EUF_TRANSFER_DST_BIT)) ret |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (in.hasFlags(IGPUImage::EUF_SAMPLED_BIT)) ret |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (in.hasFlags(IGPUImage::EUF_STORAGE_BIT)) ret |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (in.hasFlags(IGPUImage::EUF_RENDER_ATTACHMENT_BIT)) ret |= depthStencilFormat ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (in.hasFlags(IGPUImage::EUF_TRANSIENT_ATTACHMENT_BIT)) ret |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
    if (in.hasFlags(IGPUImage::EUF_INPUT_ATTACHMENT_BIT)) ret |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
    if (in.hasFlags(IGPUImage::EUF_SHADING_RATE_ATTACHMENT_BIT)) ret |= VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
    if (in.hasFlags(IGPUImage::EUF_FRAGMENT_DENSITY_MAP_BIT)) ret |= VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT;
    return ret;
}

inline core::bitflag<IGPUImage::E_USAGE_FLAGS> getImageUsageFlagsFromVkImageUsageFlags(const VkImageUsageFlags in)
{
    core::bitflag<IGPUImage::E_USAGE_FLAGS> ret = IGPUImage::EUF_NONE;
    if (in&VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ret |= IGPUImage::EUF_TRANSFER_SRC_BIT;
    if (in&VK_IMAGE_USAGE_TRANSFER_DST_BIT) ret |= IGPUImage::EUF_TRANSFER_DST_BIT;
    if (in&VK_IMAGE_USAGE_SAMPLED_BIT) ret |= IGPUImage::EUF_SAMPLED_BIT;
    if (in&VK_IMAGE_USAGE_STORAGE_BIT) ret |= IGPUImage::EUF_STORAGE_BIT;
    if (in&(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) ret |= IGPUImage::EUF_RENDER_ATTACHMENT_BIT;
    if (in&VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) ret |= IGPUImage::EUF_TRANSIENT_ATTACHMENT_BIT;
    if (in&VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) ret |= IGPUImage::EUF_INPUT_ATTACHMENT_BIT;
    if (in&VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR) ret |= IGPUImage::EUF_SHADING_RATE_ATTACHMENT_BIT;
    if (in&VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT) ret |= IGPUImage::EUF_FRAGMENT_DENSITY_MAP_BIT;
    return ret;
}

inline VkSamplerAddressMode getVkAddressModeFromTexClamp(const IGPUSampler::E_TEXTURE_CLAMP in)
{
    switch (in)
    {
    case IGPUSampler::ETC_REPEAT:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case IGPUSampler::ETC_CLAMP_TO_EDGE:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case IGPUSampler::ETC_CLAMP_TO_BORDER:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case IGPUSampler::ETC_MIRROR:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default:
        assert(!"ADDRESS MODE NOT SUPPORTED!");
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    }
}

inline std::pair<VkDebugUtilsMessageSeverityFlagsEXT, VkDebugUtilsMessageTypeFlagsEXT> getDebugCallbackFlagsFromLogLevelMask(const core::bitflag<system::ILogger::E_LOG_LEVEL> logLevelMask)
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
        type |= VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    }

    return result;
}

inline VkBlendFactor getVkBlendFactorFromBlendFactor(const asset::E_BLEND_FACTOR in)
{
    switch (in)
    {
        case asset::EBF_ZERO:
            return VK_BLEND_FACTOR_ZERO;
            break;
        case asset::EBF_ONE:
            return VK_BLEND_FACTOR_ONE;
            break;
        case asset::EBF_SRC_COLOR:
            return VK_BLEND_FACTOR_SRC_COLOR;
            break;
        case asset::EBF_ONE_MINUS_SRC_COLOR:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
            break;
        case asset::EBF_DST_COLOR:
            return VK_BLEND_FACTOR_DST_COLOR;
            break;
        case asset::EBF_ONE_MINUS_DST_COLOR:
            return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
            break;
        case asset::EBF_SRC_ALPHA:
            return VK_BLEND_FACTOR_SRC_ALPHA;
            break;
        case asset::EBF_ONE_MINUS_SRC_ALPHA:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            break;
        case asset::EBF_DST_ALPHA:
            return VK_BLEND_FACTOR_DST_ALPHA;
            break;
        case asset::EBF_ONE_MINUS_DST_ALPHA:
            return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
            break;
        case asset::EBF_CONSTANT_COLOR:
            return VK_BLEND_FACTOR_CONSTANT_COLOR;
            break;
        case asset::EBF_ONE_MINUS_CONSTANT_COLOR:
            return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
            break;
        case asset::EBF_CONSTANT_ALPHA:
            return VK_BLEND_FACTOR_CONSTANT_ALPHA;
            break;
        case asset::EBF_ONE_MINUS_CONSTANT_ALPHA:
            return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
            break;
        case asset::EBF_SRC_ALPHA_SATURATE:
            return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
            break;
        case asset::EBF_SRC1_COLOR:
            return VK_BLEND_FACTOR_SRC1_COLOR;
            break;
        case asset::EBF_ONE_MINUS_SRC1_COLOR:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
            break;
        case asset::EBF_SRC1_ALPHA:
            return VK_BLEND_FACTOR_SRC1_ALPHA;
            break;
        case asset::EBF_ONE_MINUS_SRC1_ALPHA:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
            break;
        default:
            assert(false);
            break;
    }
    return VK_BLEND_FACTOR_MAX_ENUM;
}

inline VkBlendOp getVkBlendOpFromBlendOp(const asset::E_BLEND_OP in)
{
    return static_cast<VkBlendOp>(in);
}

inline VkLogicOp getVkLogicOpFromLogicOp(const asset::E_LOGIC_OP in)
{
    return static_cast<VkLogicOp>(in);
}

inline VkColorComponentFlags getVkColorComponentFlagsFromColorWriteMask(const uint64_t in)
{
    return static_cast<VkColorComponentFlags>(in);
}

inline VkMemoryPropertyFlags getVkMemoryPropertyFlagsFromMemoryPropertyFlags(const core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> in)
{
    VkMemoryPropertyFlags ret = 0u;
    if(in.hasFlags(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_DEVICE_LOCAL_BIT))
        ret |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if(in.hasFlags(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_READABLE_BIT) || in.hasFlags(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_WRITABLE_BIT))
        ret |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    if(in.hasFlags(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_COHERENT_BIT))
        ret |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if(in.hasFlags(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_CACHED_BIT))
        ret |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    //if(in.hasFlags(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_LAZILY_ALLOCATED_BIT))
    //    ret |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
    //if(in.hasFlags(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_PROTECTED_BIT))
    //    ret |= VK_MEMORY_PROPERTY_PROTECTED_BIT;
    //if(in.hasFlags(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_DEVICE_COHERENT_BIT_AMD))
    //    ret |= VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD;
    //if(in.hasFlags(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_DEVICE_UNCACHED_BIT_AMD))
    //    ret |= VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
    //if(in.hasFlags(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_RDMA_CAPABLE_BIT_NV))
    //    ret |= VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV;
    return ret;
}

inline core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> getMemoryPropertyFlagsFromVkMemoryPropertyFlags(const VkMemoryPropertyFlags in)
{
    core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> ret(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_NONE);

    if((in & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
        ret |= IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_DEVICE_LOCAL_BIT;
    if((in & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
        ret |= core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_READABLE_BIT) | IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_WRITABLE_BIT;
    if((in & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0)
        ret |= IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_COHERENT_BIT;
    if((in & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0)
        ret |= IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_HOST_CACHED_BIT;
    
    //if((in & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0)
    //    ret |= IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_LAZILY_ALLOCATED_BIT;
    //if((in & VK_MEMORY_PROPERTY_PROTECTED_BIT) != 0)
    //    ret |= IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_PROTECTED_BIT;
    //if((in & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD) != 0)
    //    ret |= IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_DEVICE_COHERENT_BIT_AMD;
    //if((in & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD) != 0)
    //    ret |= IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_DEVICE_UNCACHED_BIT_AMD;
    //if((in & VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV) != 0)
    //    ret |= IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS::EMPF_RDMA_CAPABLE_BIT_NV;

    return ret;
}

inline constexpr VkDescriptorType getVkDescriptorTypeFromDescriptorType(const asset::IDescriptor::E_TYPE descriptorType)
{
    switch (descriptorType)
    {
        case asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER:
            return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        case asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE:
            return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER:
            return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        case asset::IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER:
            return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        case asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER:
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        case asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC:
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        case asset::IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT:
            return VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        case asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE:
            return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        default:
            assert(!"Invalid code path.");
            return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
}

inline IPhysicalDevice::E_DRIVER_ID getDriverIdFromVkDriverId(const VkDriverId in)
{
    if(in == VK_DRIVER_ID_AMD_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_AMD_PROPRIETARY;
    if (in == VK_DRIVER_ID_AMD_OPEN_SOURCE) return IPhysicalDevice::E_DRIVER_ID::EDI_AMD_OPEN_SOURCE;
    if (in == VK_DRIVER_ID_MESA_RADV) return IPhysicalDevice::E_DRIVER_ID::EDI_MESA_RADV;
    if (in == VK_DRIVER_ID_NVIDIA_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY;
    if (in == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS) return IPhysicalDevice::E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS;
    if (in == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA) return IPhysicalDevice::E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA;
    if (in == VK_DRIVER_ID_IMAGINATION_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_IMAGINATION_PROPRIETARY;
    if (in == VK_DRIVER_ID_QUALCOMM_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_QUALCOMM_PROPRIETARY;
    if (in == VK_DRIVER_ID_ARM_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_ARM_PROPRIETARY;
    if (in == VK_DRIVER_ID_GOOGLE_SWIFTSHADER) return IPhysicalDevice::E_DRIVER_ID::EDI_GOOGLE_SWIFTSHADER;
    if (in == VK_DRIVER_ID_GGP_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_GGP_PROPRIETARY;
    if (in == VK_DRIVER_ID_BROADCOM_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_BROADCOM_PROPRIETARY;
    if (in == VK_DRIVER_ID_MESA_LLVMPIPE) return IPhysicalDevice::E_DRIVER_ID::EDI_MESA_LLVMPIPE;
    if (in == VK_DRIVER_ID_MOLTENVK) return IPhysicalDevice::E_DRIVER_ID::EDI_MOLTENVK;                  
    if (in == VK_DRIVER_ID_COREAVI_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_COREAVI_PROPRIETARY;
    if (in == VK_DRIVER_ID_JUICE_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_JUICE_PROPRIETARY;         
    if (in == VK_DRIVER_ID_VERISILICON_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_VERISILICON_PROPRIETARY;
    if (in == VK_DRIVER_ID_MESA_TURNIP) return IPhysicalDevice::E_DRIVER_ID::EDI_MESA_TURNIP;
    if (in == VK_DRIVER_ID_MESA_V3DV) return IPhysicalDevice::E_DRIVER_ID::EDI_MESA_V3DV;
    if (in == VK_DRIVER_ID_MESA_PANVK) return IPhysicalDevice::E_DRIVER_ID::EDI_MESA_PANVK;                
    if (in == VK_DRIVER_ID_SAMSUNG_PROPRIETARY) return IPhysicalDevice::E_DRIVER_ID::EDI_SAMSUNG_PROPRIETARY;
    if (in == VK_DRIVER_ID_MESA_VENUS) return IPhysicalDevice::E_DRIVER_ID::EDI_MESA_VENUS; 
    return IPhysicalDevice::E_DRIVER_ID::EDI_UNKNOWN;
}

inline VkAttachmentLoadOp getVkAttachmentLoadOpFrom(const asset::IRenderpass::LOAD_OP op)
{
    return static_cast<VkAttachmentLoadOp>(op);
}
inline asset::IRenderpass::LOAD_OP getAttachmentLoadOpFrom(const VkAttachmentLoadOp op)
{
    return static_cast<asset::IRenderpass::LOAD_OP>(op);
}

inline VkAttachmentStoreOp getVkAttachmentStoreOpFrom(const asset::IRenderpass::STORE_OP op)
{
    return static_cast<VkAttachmentStoreOp>(op);
}
inline asset::IRenderpass::STORE_OP getAttachmentStoreOpFrom(const VkAttachmentStoreOp op)
{
    return static_cast<asset::IRenderpass::STORE_OP>(op);
}


}

#endif
