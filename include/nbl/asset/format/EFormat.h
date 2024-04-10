// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_E_FORMAT_H_INCLUDED__
#define __NBL_ASSET_E_FORMAT_H_INCLUDED__

#include <cstdint>
#include <type_traits>
#include "BuildConfigOptions.h"
#include "vectorSIMD.h"
#include "nbl/core/math/rational.h"
#include "nbl/core/math/colorutil.h"

namespace nbl::asset
{

//! An enum for the color format of textures used by the Nabla.
// @Crisspl it would be dandy if the values (or at least ordering) of our enums matched vulkan's
/** A color format specifies how color information is stored. */
enum E_FORMAT : uint8_t
{
    //! Custom shizz we wont ever use
    EF_D16_UNORM,
    EF_X8_D24_UNORM_PACK32,
    EF_D32_SFLOAT,
    EF_S8_UINT,
    EF_D16_UNORM_S8_UINT,
    EF_D24_UNORM_S8_UINT,
    EF_D32_SFLOAT_S8_UINT,

    //! Vulkan
    EF_R4G4_UNORM_PACK8,
    EF_R4G4B4A4_UNORM_PACK16,
    EF_B4G4R4A4_UNORM_PACK16,
    EF_R5G6B5_UNORM_PACK16,
    EF_B5G6R5_UNORM_PACK16,
    EF_R5G5B5A1_UNORM_PACK16,
    EF_B5G5R5A1_UNORM_PACK16,
    EF_A1R5G5B5_UNORM_PACK16,
    EF_R8_UNORM,
    EF_R8_SNORM,
    EF_R8_USCALED,
    EF_R8_SSCALED,
    EF_R8_UINT,
    EF_R8_SINT,
    EF_R8_SRGB,
    EF_R8G8_UNORM,
    EF_R8G8_SNORM,
    EF_R8G8_USCALED,
    EF_R8G8_SSCALED,
    EF_R8G8_UINT,
    EF_R8G8_SINT,
    EF_R8G8_SRGB,
    EF_R8G8B8_UNORM,
    EF_R8G8B8_SNORM,
    EF_R8G8B8_USCALED,
    EF_R8G8B8_SSCALED,
    EF_R8G8B8_UINT,
    EF_R8G8B8_SINT,
    EF_R8G8B8_SRGB,
    EF_B8G8R8_UNORM,
    EF_B8G8R8_SNORM,
    EF_B8G8R8_USCALED,
    EF_B8G8R8_SSCALED,
    EF_B8G8R8_UINT,
    EF_B8G8R8_SINT,
    EF_B8G8R8_SRGB,
    EF_R8G8B8A8_UNORM,
    EF_R8G8B8A8_SNORM,
    EF_R8G8B8A8_USCALED,
    EF_R8G8B8A8_SSCALED,
    EF_R8G8B8A8_UINT,
    EF_R8G8B8A8_SINT,
    EF_R8G8B8A8_SRGB,
    EF_B8G8R8A8_UNORM,
    EF_B8G8R8A8_SNORM,
    EF_B8G8R8A8_USCALED,
    EF_B8G8R8A8_SSCALED,
    EF_B8G8R8A8_UINT,
    EF_B8G8R8A8_SINT,
    EF_B8G8R8A8_SRGB,
    EF_A8B8G8R8_UNORM_PACK32,
    EF_A8B8G8R8_SNORM_PACK32,
    EF_A8B8G8R8_USCALED_PACK32,
    EF_A8B8G8R8_SSCALED_PACK32,
    EF_A8B8G8R8_UINT_PACK32,
    EF_A8B8G8R8_SINT_PACK32,
    EF_A8B8G8R8_SRGB_PACK32,
    EF_A2R10G10B10_UNORM_PACK32,
    EF_A2R10G10B10_SNORM_PACK32,
    EF_A2R10G10B10_USCALED_PACK32,
    EF_A2R10G10B10_SSCALED_PACK32,
    EF_A2R10G10B10_UINT_PACK32,
    EF_A2R10G10B10_SINT_PACK32,
    EF_A2B10G10R10_UNORM_PACK32,
    EF_A2B10G10R10_SNORM_PACK32,
    EF_A2B10G10R10_USCALED_PACK32,
    EF_A2B10G10R10_SSCALED_PACK32,
    EF_A2B10G10R10_UINT_PACK32,
    EF_A2B10G10R10_SINT_PACK32,
    EF_R16_UNORM,
    EF_R16_SNORM,
    EF_R16_USCALED,
    EF_R16_SSCALED,
    EF_R16_UINT,
    EF_R16_SINT,
    EF_R16_SFLOAT,
    EF_R16G16_UNORM,
    EF_R16G16_SNORM,
    EF_R16G16_USCALED,
    EF_R16G16_SSCALED,
    EF_R16G16_UINT,
    EF_R16G16_SINT,
    EF_R16G16_SFLOAT,
    EF_R16G16B16_UNORM,
    EF_R16G16B16_SNORM,
    EF_R16G16B16_USCALED,
    EF_R16G16B16_SSCALED,
    EF_R16G16B16_UINT,
    EF_R16G16B16_SINT,
    EF_R16G16B16_SFLOAT,
    EF_R16G16B16A16_UNORM,
    EF_R16G16B16A16_SNORM,
    EF_R16G16B16A16_USCALED,
    EF_R16G16B16A16_SSCALED,
    EF_R16G16B16A16_UINT,
    EF_R16G16B16A16_SINT,
    EF_R16G16B16A16_SFLOAT,
    EF_R32_UINT,
    EF_R32_SINT,
    EF_R32_SFLOAT,
    EF_R32G32_UINT,
    EF_R32G32_SINT,
    EF_R32G32_SFLOAT,
    EF_R32G32B32_UINT,
    EF_R32G32B32_SINT,
    EF_R32G32B32_SFLOAT,
    EF_R32G32B32A32_UINT,
    EF_R32G32B32A32_SINT,
    EF_R32G32B32A32_SFLOAT,
    EF_R64_UINT,
    EF_R64_SINT,
    EF_R64_SFLOAT,
    EF_R64G64_UINT,
    EF_R64G64_SINT,
    EF_R64G64_SFLOAT,
    EF_R64G64B64_UINT,
    EF_R64G64B64_SINT,
    EF_R64G64B64_SFLOAT,
    EF_R64G64B64A64_UINT,
    EF_R64G64B64A64_SINT,
    EF_R64G64B64A64_SFLOAT,
    EF_B10G11R11_UFLOAT_PACK32,
    EF_E5B9G9R9_UFLOAT_PACK32,

    //! Block Compression Formats!
    EF_BC1_RGB_UNORM_BLOCK,
    EF_BC1_RGB_SRGB_BLOCK,
    EF_BC1_RGBA_UNORM_BLOCK,
    EF_BC1_RGBA_SRGB_BLOCK,
    EF_BC2_UNORM_BLOCK,
    EF_BC2_SRGB_BLOCK,
    EF_BC3_UNORM_BLOCK,
    EF_BC3_SRGB_BLOCK,
    EF_BC4_UNORM_BLOCK,
    EF_BC4_SNORM_BLOCK,
    EF_BC5_UNORM_BLOCK,
    EF_BC5_SNORM_BLOCK,
    EF_BC6H_UFLOAT_BLOCK,
    EF_BC6H_SFLOAT_BLOCK,
    EF_BC7_UNORM_BLOCK,
    EF_BC7_SRGB_BLOCK,
    EF_ASTC_4x4_UNORM_BLOCK,
    EF_ASTC_4x4_SRGB_BLOCK,
    EF_ASTC_5x4_UNORM_BLOCK,
    EF_ASTC_5x4_SRGB_BLOCK,
    EF_ASTC_5x5_UNORM_BLOCK,
    EF_ASTC_5x5_SRGB_BLOCK,
    EF_ASTC_6x5_UNORM_BLOCK,
    EF_ASTC_6x5_SRGB_BLOCK,
    EF_ASTC_6x6_UNORM_BLOCK,
    EF_ASTC_6x6_SRGB_BLOCK,
    EF_ASTC_8x5_UNORM_BLOCK,
    EF_ASTC_8x5_SRGB_BLOCK,
    EF_ASTC_8x6_UNORM_BLOCK,
    EF_ASTC_8x6_SRGB_BLOCK,
    EF_ASTC_8x8_UNORM_BLOCK,
    EF_ASTC_8x8_SRGB_BLOCK,
    EF_ASTC_10x5_UNORM_BLOCK,
    EF_ASTC_10x5_SRGB_BLOCK,
    EF_ASTC_10x6_UNORM_BLOCK,
    EF_ASTC_10x6_SRGB_BLOCK,
    EF_ASTC_10x8_UNORM_BLOCK,
    EF_ASTC_10x8_SRGB_BLOCK,
    EF_ASTC_10x10_UNORM_BLOCK,
    EF_ASTC_10x10_SRGB_BLOCK,
    EF_ASTC_12x10_UNORM_BLOCK,
    EF_ASTC_12x10_SRGB_BLOCK,
    EF_ASTC_12x12_UNORM_BLOCK,
    EF_ASTC_12x12_SRGB_BLOCK,
    EF_ETC2_R8G8B8_UNORM_BLOCK,
    EF_ETC2_R8G8B8_SRGB_BLOCK,
    EF_ETC2_R8G8B8A1_UNORM_BLOCK,
    EF_ETC2_R8G8B8A1_SRGB_BLOCK,
    EF_ETC2_R8G8B8A8_UNORM_BLOCK,
    EF_ETC2_R8G8B8A8_SRGB_BLOCK,
    EF_EAC_R11_UNORM_BLOCK,
    EF_EAC_R11_SNORM_BLOCK,
    EF_EAC_R11G11_UNORM_BLOCK,
    EF_EAC_R11G11_SNORM_BLOCK,
    EF_PVRTC1_2BPP_UNORM_BLOCK_IMG,
    EF_PVRTC1_4BPP_UNORM_BLOCK_IMG,
    EF_PVRTC2_2BPP_UNORM_BLOCK_IMG,
    EF_PVRTC2_4BPP_UNORM_BLOCK_IMG,
    EF_PVRTC1_2BPP_SRGB_BLOCK_IMG,
    EF_PVRTC1_4BPP_SRGB_BLOCK_IMG,
    EF_PVRTC2_2BPP_SRGB_BLOCK_IMG,
    EF_PVRTC2_4BPP_SRGB_BLOCK_IMG,

    //! Planar formats
    EF_G8_B8_R8_3PLANE_420_UNORM,
    EF_G8_B8R8_2PLANE_420_UNORM,
    EF_G8_B8_R8_3PLANE_422_UNORM,
    EF_G8_B8R8_2PLANE_422_UNORM,
    EF_G8_B8_R8_3PLANE_444_UNORM,

    //! Unknown color format:
    EF_UNKNOWN,
    EF_COUNT = EF_UNKNOWN
};

enum E_FORMAT_CLASS : uint8_t
{
    EFC_8_BIT,
    EFC_16_BIT,
    EFC_24_BIT,
    EFC_32_BIT,
    EFC_48_BIT,
    EFC_64_BIT,
    EFC_96_BIT,
    EFC_128_BIT,
    EFC_192_BIT,
    EFC_256_BIT,

    EFC_BC1_RGB,
    EFC_BC1_RGBA,
    EFC_BC2,
    EFC_BC3,
    EFC_BC4,
    EFC_BC5,
    EFC_BC6,
    EFC_BC7,

    EFC_ETC2_RGB,
    EFC_ETC2_RGBA,
    EFC_ETC2_EAC_RGBA,
    EFC_ETC2_EAC_R,
    EFC_ETC2_EAC_RG,

    EFC_ASTC_4X4,
    EFC_ASTC_5X4,
    EFC_ASTC_5X5,
    EFC_ASTC_6X5,
    EFC_ASTC_6X6,
    EFC_ASTC_8X5,
    EFC_ASTC_8X6,
    EFC_ASTC_8X8,
    EFC_ASTC_10X5,
    EFC_ASTC_10X6,
    EFC_ASTC_10X8,
    EFC_ASTC_10X10,
    EFC_ASTC_12X10,
    EFC_ASTC_12X12,

    // [TODO] there are still more format classes; https://registry.khronos.org/vulkan/specs/1.2-extensions/html/chap43.html#formats-compatibility-classes
};

enum E_FORMAT_FEATURE : uint32_t
{
    EFF_SAMPLED_IMAGE_BIT = 0x00000001,
    EFF_STORAGE_IMAGE_BIT = 0x00000002,
    EFF_STORAGE_IMAGE_ATOMIC_BIT = 0x00000004,
    EFF_UNIFORM_TEXEL_BUFFER_BIT = 0x00000008,
    EFF_STORAGE_TEXEL_BUFFER_BIT = 0x00000010,
    EFF_STORAGE_TEXEL_BUFFER_ATOMIC_BIT = 0x00000020,
    EFF_VERTEX_BUFFER_BIT = 0x00000040,
    EFF_COLOR_ATTACHMENT_BIT = 0x00000080,
    EFF_COLOR_ATTACHMENT_BLEND_BIT = 0x00000100,
    EFF_DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000200,
    EFF_BLIT_SRC_BIT = 0x00000400,
    EFF_BLIT_DST_BIT = 0x00000800,
    EFF_SAMPLED_IMAGE_FILTER_LINEAR_BIT = 0x00001000,
    EFF_TRANSFER_SRC_BIT = 0x00004000,
    EFF_TRANSFER_DST_BIT = 0x00008000,
    EFF_MIDPOINT_CHROMA_SAMPLES_BIT = 0x00020000,
    EFF_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT = 0x00040000,
    EFF_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT = 0x00080000,
    EFF_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT = 0x00100000,
    EFF_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT = 0x00200000,
    EFF_DISJOINT_BIT = 0x00400000,
    EFF_COSITED_CHROMA_SAMPLES_BIT = 0x00800000,
    EFF_SAMPLED_IMAGE_FILTER_MINMAX_BIT = 0x00010000,
    EFF_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG = 0x00002000,
    EFF_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT = 0x20000000,
    EFF_FRAGMENT_DENSITY_MAP_BIT = 0x01000000,
    EFF_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT = 0x40000000,
    EFF_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
};

//
inline E_FORMAT_CLASS getFormatClass(E_FORMAT _fmt)
{
#include "nbl/asset/format/impl/EFormat_getFormatClass.h"
}
template<E_FORMAT _fmt>
constexpr E_FORMAT_CLASS getFormatClass()
{
#include "nbl/asset/format/impl/EFormat_getFormatClass.h"
}

//
inline uint32_t getFormatClassBlockBytesize(E_FORMAT_CLASS _fclass)
{
#include "nbl/asset/format/impl/EFormat_getFormatClassBlockBytesize.h"
}
template<E_FORMAT_CLASS _fclass>
constexpr uint32_t getFormatClassBlockBytesize()
{
#include "nbl/asset/format/impl/EFormat_getFormatClassBlockBytesize.h"
}

//
static inline constexpr uint32_t MaxTexelBlockDimensions[] = { 12u, 12u, 1u, 1u };
inline core::vector3du32_SIMD getBlockDimensions(E_FORMAT_CLASS _fclass)
{
#include "nbl/asset/format/impl/EFormat_getBlockDimensions.h"
}
template<E_FORMAT_CLASS _fclass>
const core::vector3du32_SIMD getBlockDimensions()
{
#include "nbl/asset/format/impl/EFormat_getBlockDimensions.h"
}
// TODO: do via `getBlockDimensions(getFormatClass(_fmt))`
inline core::vector3du32_SIMD getBlockDimensions(asset::E_FORMAT _fmt)
{
    switch (_fmt)
    {
    case EF_BC1_RGB_UNORM_BLOCK:
    case EF_BC1_RGB_SRGB_BLOCK:
    case EF_BC1_RGBA_UNORM_BLOCK:
    case EF_BC1_RGBA_SRGB_BLOCK:
    case EF_BC2_UNORM_BLOCK:
    case EF_BC2_SRGB_BLOCK:
    case EF_BC3_UNORM_BLOCK:
    case EF_BC3_SRGB_BLOCK:
    case EF_BC4_SNORM_BLOCK:
    case EF_BC4_UNORM_BLOCK:
    case EF_BC5_SNORM_BLOCK:
    case EF_BC5_UNORM_BLOCK:
    case EF_BC6H_SFLOAT_BLOCK:
    case EF_BC6H_UFLOAT_BLOCK:
    case EF_BC7_SRGB_BLOCK:
    case EF_BC7_UNORM_BLOCK:
    case EF_ASTC_4x4_UNORM_BLOCK:
    case EF_ASTC_4x4_SRGB_BLOCK:
    case EF_ETC2_R8G8B8_UNORM_BLOCK:
    case EF_ETC2_R8G8B8_SRGB_BLOCK:
    case EF_ETC2_R8G8B8A1_UNORM_BLOCK:
    case EF_ETC2_R8G8B8A1_SRGB_BLOCK:
    case EF_ETC2_R8G8B8A8_UNORM_BLOCK:
    case EF_ETC2_R8G8B8A8_SRGB_BLOCK:
    case EF_EAC_R11_UNORM_BLOCK:
    case EF_EAC_R11_SNORM_BLOCK:
    case EF_EAC_R11G11_UNORM_BLOCK:
    case EF_EAC_R11G11_SNORM_BLOCK:
        return core::vector4du32_SIMD(4u, 4u, 1u, 1u);
    case EF_ASTC_5x4_UNORM_BLOCK:
    case EF_ASTC_5x4_SRGB_BLOCK:
        return core::vector4du32_SIMD(5u, 4u, 1u, 1u);
    case EF_ASTC_5x5_UNORM_BLOCK:
    case EF_ASTC_5x5_SRGB_BLOCK:
        return core::vector4du32_SIMD(5u, 5u, 1u, 1u);
    case EF_ASTC_6x5_UNORM_BLOCK:
    case EF_ASTC_6x5_SRGB_BLOCK:
        return core::vector4du32_SIMD(6u, 5u, 1u, 1u);
    case EF_ASTC_6x6_UNORM_BLOCK:
    case EF_ASTC_6x6_SRGB_BLOCK:
        return core::vector4du32_SIMD(6u, 6u, 1u, 1u);
    case EF_ASTC_8x5_UNORM_BLOCK:
    case EF_ASTC_8x5_SRGB_BLOCK:
        return core::vector4du32_SIMD(8u, 5u, 1u, 1u);
    case EF_ASTC_8x6_UNORM_BLOCK:
    case EF_ASTC_8x6_SRGB_BLOCK:
        return core::vector4du32_SIMD(8u, 6u, 1u, 1u);
    case EF_ASTC_8x8_UNORM_BLOCK:
    case EF_ASTC_8x8_SRGB_BLOCK:
        return core::vector4du32_SIMD(8u, 8u, 1u, 1u);
    case EF_ASTC_10x5_UNORM_BLOCK:
    case EF_ASTC_10x5_SRGB_BLOCK:
        return core::vector4du32_SIMD(10u, 5u, 1u, 1u);
    case EF_ASTC_10x6_UNORM_BLOCK:
    case EF_ASTC_10x6_SRGB_BLOCK:
        return core::vector4du32_SIMD(10u, 6u, 1u, 1u);
    case EF_ASTC_10x8_UNORM_BLOCK:
    case EF_ASTC_10x8_SRGB_BLOCK:
        return core::vector4du32_SIMD(10u, 8u, 1u, 1u);
    case EF_ASTC_10x10_UNORM_BLOCK:
    case EF_ASTC_10x10_SRGB_BLOCK:
        return core::vector4du32_SIMD(10u, 10u, 1u, 1u);
    case EF_ASTC_12x10_UNORM_BLOCK:
    case EF_ASTC_12x10_SRGB_BLOCK:
        return core::vector4du32_SIMD(12u, 10u, 1u, 1u);
    case EF_ASTC_12x12_UNORM_BLOCK:
    case EF_ASTC_12x12_SRGB_BLOCK:
        return core::vector4du32_SIMD(12u, 12u, 1u, 1u);
    case EF_PVRTC1_2BPP_UNORM_BLOCK_IMG:
    case EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
    case EF_PVRTC2_2BPP_UNORM_BLOCK_IMG:
    case EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
        return core::vector4du32_SIMD(8u, 4u, 1u, 1u);
    case EF_PVRTC1_4BPP_UNORM_BLOCK_IMG:
    case EF_PVRTC2_4BPP_UNORM_BLOCK_IMG:
    case EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
    case EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
        return core::vector4du32_SIMD(4u, 4u, 1u, 1u);
    default:
        return core::vector4du32_SIMD(1u);
    }
}
template<asset::E_FORMAT _fmt>
const core::vector3du32_SIMD getBlockDimensions()
{
    switch (_fmt)
    {
    case EF_BC1_RGB_UNORM_BLOCK:
    case EF_BC1_RGB_SRGB_BLOCK:
    case EF_BC1_RGBA_UNORM_BLOCK:
    case EF_BC1_RGBA_SRGB_BLOCK:
    case EF_BC2_UNORM_BLOCK:
    case EF_BC2_SRGB_BLOCK:
    case EF_BC3_UNORM_BLOCK:
    case EF_BC3_SRGB_BLOCK:
    case EF_BC4_SNORM_BLOCK:
    case EF_BC4_UNORM_BLOCK:
    case EF_BC5_SNORM_BLOCK:
    case EF_BC5_UNORM_BLOCK:
    case EF_BC6H_SFLOAT_BLOCK:
    case EF_BC6H_UFLOAT_BLOCK:
    case EF_BC7_SRGB_BLOCK:
    case EF_BC7_UNORM_BLOCK:
    case EF_ASTC_4x4_UNORM_BLOCK:
    case EF_ASTC_4x4_SRGB_BLOCK:
    case EF_ETC2_R8G8B8_UNORM_BLOCK:
    case EF_ETC2_R8G8B8_SRGB_BLOCK:
    case EF_ETC2_R8G8B8A1_UNORM_BLOCK:
    case EF_ETC2_R8G8B8A1_SRGB_BLOCK:
    case EF_ETC2_R8G8B8A8_UNORM_BLOCK:
    case EF_ETC2_R8G8B8A8_SRGB_BLOCK:
    case EF_EAC_R11_UNORM_BLOCK:
    case EF_EAC_R11_SNORM_BLOCK:
    case EF_EAC_R11G11_UNORM_BLOCK:
    case EF_EAC_R11G11_SNORM_BLOCK:
        return core::vector4du32_SIMD(4u, 4u, 1u, 1u);
    case EF_ASTC_5x4_UNORM_BLOCK:
    case EF_ASTC_5x4_SRGB_BLOCK:
        return core::vector4du32_SIMD(5u, 4u, 1u, 1u);
    case EF_ASTC_5x5_UNORM_BLOCK:
    case EF_ASTC_5x5_SRGB_BLOCK:
        return core::vector4du32_SIMD(5u, 5u, 1u, 1u);
    case EF_ASTC_6x5_UNORM_BLOCK:
    case EF_ASTC_6x5_SRGB_BLOCK:
        return core::vector4du32_SIMD(6u, 5u, 1u, 1u);
    case EF_ASTC_6x6_UNORM_BLOCK:
    case EF_ASTC_6x6_SRGB_BLOCK:
        return core::vector4du32_SIMD(6u, 6u, 1u, 1u);
    case EF_ASTC_8x5_UNORM_BLOCK:
    case EF_ASTC_8x5_SRGB_BLOCK:
        return core::vector4du32_SIMD(8u, 5u, 1u, 1u);
    case EF_ASTC_8x6_UNORM_BLOCK:
    case EF_ASTC_8x6_SRGB_BLOCK:
        return core::vector4du32_SIMD(8u, 6u, 1u, 1u);
    case EF_ASTC_8x8_UNORM_BLOCK:
    case EF_ASTC_8x8_SRGB_BLOCK:
        return core::vector4du32_SIMD(8u, 8u, 1u, 1u);
    case EF_ASTC_10x5_UNORM_BLOCK:
    case EF_ASTC_10x5_SRGB_BLOCK:
        return core::vector4du32_SIMD(10u, 5u, 1u, 1u);
    case EF_ASTC_10x6_UNORM_BLOCK:
    case EF_ASTC_10x6_SRGB_BLOCK:
        return core::vector4du32_SIMD(10u, 6u, 1u, 1u);
    case EF_ASTC_10x8_UNORM_BLOCK:
    case EF_ASTC_10x8_SRGB_BLOCK:
        return core::vector4du32_SIMD(10u, 8u, 1u, 1u);
    case EF_ASTC_10x10_UNORM_BLOCK:
    case EF_ASTC_10x10_SRGB_BLOCK:
        return core::vector4du32_SIMD(10u, 10u, 1u, 1u);
    case EF_ASTC_12x10_UNORM_BLOCK:
    case EF_ASTC_12x10_SRGB_BLOCK:
        return core::vector4du32_SIMD(12u, 10u, 1u, 1u);
    case EF_ASTC_12x12_UNORM_BLOCK:
    case EF_ASTC_12x12_SRGB_BLOCK:
        return core::vector4du32_SIMD(12u, 12u, 1u, 1u);
    case EF_PVRTC1_2BPP_UNORM_BLOCK_IMG:
    case EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
    case EF_PVRTC2_2BPP_UNORM_BLOCK_IMG:
    case EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
        return core::vector4du32_SIMD(8u, 4u, 1u, 1u);
    case EF_PVRTC1_4BPP_UNORM_BLOCK_IMG:
    case EF_PVRTC2_4BPP_UNORM_BLOCK_IMG:
    case EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
    case EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
        return core::vector4du32_SIMD(4u, 4u, 1u, 1u);
    default:
        return core::vector4du32_SIMD(1u);
    }
}

// TODO: do it via `getFormatClassBlockBytesize(getFormatClass(_fmt))`
inline uint32_t getTexelOrBlockBytesize(asset::E_FORMAT _fmt)
{
#include "nbl/asset/format/impl/EFormat_getTexelOrBlockBytesize.h"
}
template<asset::E_FORMAT _fmt>
constexpr uint32_t getTexelOrBlockBytesize()
{
#include "nbl/asset/format/impl/EFormat_getTexelOrBlockBytesize.h"
}

//
inline uint32_t getFormatChannelCount(asset::E_FORMAT _fmt)
{
#include "nbl/asset/format/impl/EFormat_getFormatChannelCount.h"
}
template<E_FORMAT _fmt>
constexpr uint32_t getFormatChannelCount()
{
#include "nbl/asset/format/impl/EFormat_getFormatChannelCount.h"
}

/*
inline uint32_t getBitsPerChannel(asset::E_FORMAT _fmt, uint8_t _channel)
{
    if constexpr (_channel>=getFormatChannelCount(_fmt))
        return 0u;

#include "nbl/asset/format/impl/EFormat_getBitsPerChannel.h"
}
template<asset::E_FORMAT _fmt, uint8_t _channel>
constexpr const uint8_t getBitsPerChannel()
{
    if constexpr (_channel>=getFormatChannelCount<_fmt>())
        return 0u;

#include "nbl/asset/format/impl/EFormat_getBitsPerChannel.h"
}
*/

/*
    It provides some useful functions for dealing
    with texel-block conversions, rounding up
    for alignment rules and specific format
    information such as dimension or block byte size.
*/

struct TexelBlockInfo
{
    public:
        TexelBlockInfo(E_FORMAT format) :
            dimension(getBlockDimensions(format)),
            maxCoord(dimension-core::vector3du32_SIMD(1u, 1u, 1u, 1u)),
            blockByteSize(getTexelOrBlockBytesize(format))
        {}
            
        inline bool operator==(const TexelBlockInfo& rhs) const
        {
            return
                (dimension == rhs.dimension).xyzz().all() &&
                (maxCoord == rhs.maxCoord).xyzz().all() &&
                blockByteSize == rhs.blockByteSize;
        }

        inline bool operator!=(const TexelBlockInfo& rhs) const
        {
            return !this->operator==(rhs);
        }

        //! It converts input texels strides to blocks while rounding up at the same time
        /*
            The true extent is a dimension of stride in texels or in blocks, depending
            of what are you dealing with.

            @param coord it's a dimension in texels.
            @see convertTexelsToBlocks
        */

        inline auto convertTexelsToBlocks(const core::vector3du32_SIMD& coord) const
        {
            return (coord+maxCoord)/dimension;
        }

        //! It converts input texels strides to compute multiples of block sizes
        /*
            Since your format may be block compressed, you can gain your 
            true extent either in texels or blocks, but there are certain
            rules for adjusting alignments, that's why you sometimes need to
            round up values.

            For instance - given a BC texetur 4x4 having 64 texels in a row, the function
            will return 64 for the row, but for 1,2 or 3 texels it will return 4, so 
            round-up appears.

            As mentioned and generally - it's quite useful to determine actual mipmap sizes 
            of BC compressed mip map levels or textures not aligned to block size, because 
            it gives you the size of the overlap.
        */

        inline auto roundToBlockSize(const core::vector3du32_SIMD& coord) const
        {
            return convertTexelsToBlocks(coord)*dimension;
        }


        inline core::vector4du32_SIMD convert3DBlockStridesTo1DByteStrides(core::vector3du32_SIMD blockStrides) const
        {
            // shuffle and put a 1 in the first element
            core::vector4du32_SIMD retval = blockStrides;
            retval = retval.wxyz();
            // byte stride for x+ step
            retval[0] = blockByteSize;
            // row by bytesize
            retval[1] *= retval[0];
            // slice by row
            retval[2] *= retval[1];
            // layer by slice
            retval[3] *= retval[2];
            return retval;
        }

        inline core::vector4du32_SIMD convert3DTexelStridesTo1DByteStrides(core::vector3du32_SIMD texelStrides) const
        {
            return convert3DBlockStridesTo1DByteStrides(convertTexelsToBlocks(texelStrides));
        }

        inline const auto& getDimension() const { return dimension; }

        inline const auto& getBlockByteSize() const { return blockByteSize; }

    private:
        core::vector3du32_SIMD dimension;
        core::vector3du32_SIMD maxCoord;
        uint32_t blockByteSize;
};

inline core::rational<uint32_t> getBytesPerPixel(asset::E_FORMAT _fmt)
{
    auto dims = getBlockDimensions(_fmt);
    return { getTexelOrBlockBytesize(_fmt), dims[0]*dims[1]*dims[2] };
}


//! Boolean Queries

inline bool isDepthOnlyFormat(asset::E_FORMAT _fmt)
{
    switch (_fmt)
    {
    case EF_D16_UNORM:
    case EF_X8_D24_UNORM_PACK32:
    case EF_D32_SFLOAT:
        return true;
    default:
        return false;
    }
}

inline bool isStencilOnlyFormat(asset::E_FORMAT _fmt)
{
    return (_fmt == EF_S8_UINT);
}

inline bool isDepthOrStencilFormat(asset::E_FORMAT _fmt)
{
    switch (_fmt)
    {
    case EF_D16_UNORM:
    case EF_X8_D24_UNORM_PACK32:
    case EF_D32_SFLOAT:
    case EF_S8_UINT:
    case EF_D16_UNORM_S8_UINT:
    case EF_D24_UNORM_S8_UINT:
    case EF_D32_SFLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

template<asset::E_FORMAT _fmt>
constexpr bool isDepthOrStencilFormat()
{
    switch (_fmt)
    {
    case EF_D16_UNORM:
    case EF_X8_D24_UNORM_PACK32:
    case EF_D32_SFLOAT:
    case EF_S8_UINT:
    case EF_D16_UNORM_S8_UINT:
    case EF_D24_UNORM_S8_UINT:
    case EF_D32_SFLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

inline bool isBGRALayoutFormat(asset::E_FORMAT _fmt)
{
    switch (_fmt)
    {
    case EF_B4G4R4A4_UNORM_PACK16:
    case EF_B5G6R5_UNORM_PACK16:
    case EF_B5G5R5A1_UNORM_PACK16:
    case EF_B8G8R8_UNORM:
    case EF_B8G8R8_SNORM:
    case EF_B8G8R8_USCALED:
    case EF_B8G8R8_SSCALED:
    case EF_B8G8R8_UINT:
    case EF_B8G8R8_SINT:
    case EF_B8G8R8_SRGB:
    case EF_B8G8R8A8_UNORM:
    case EF_B8G8R8A8_SNORM:
    case EF_B8G8R8A8_USCALED:
    case EF_B8G8R8A8_SSCALED:
    case EF_B8G8R8A8_UINT:
    case EF_B8G8R8A8_SINT:
    case EF_B8G8R8A8_SRGB:
    case EF_A8B8G8R8_UNORM_PACK32:
    case EF_A8B8G8R8_SNORM_PACK32:
    case EF_A8B8G8R8_USCALED_PACK32:
    case EF_A8B8G8R8_SSCALED_PACK32:
    case EF_A8B8G8R8_UINT_PACK32:
    case EF_A8B8G8R8_SINT_PACK32:
    case EF_A8B8G8R8_SRGB_PACK32:
    case EF_A2B10G10R10_UNORM_PACK32:
    case EF_A2B10G10R10_SNORM_PACK32:
    case EF_A2B10G10R10_USCALED_PACK32:
    case EF_A2B10G10R10_SSCALED_PACK32:
    case EF_A2B10G10R10_UINT_PACK32:
    case EF_A2B10G10R10_SINT_PACK32:
    case EF_B10G11R11_UFLOAT_PACK32:
    case EF_E5B9G9R9_UFLOAT_PACK32:
        return true;
    default:
        return false;
    }
}

template<asset::E_FORMAT _fmt>
constexpr bool isBGRALayoutFormat()
{
    switch (_fmt)
    {
    case EF_B4G4R4A4_UNORM_PACK16:
    case EF_B5G6R5_UNORM_PACK16:
    case EF_B5G5R5A1_UNORM_PACK16:
    case EF_B8G8R8_UNORM:
    case EF_B8G8R8_SNORM:
    case EF_B8G8R8_USCALED:
    case EF_B8G8R8_SSCALED:
    case EF_B8G8R8_UINT:
    case EF_B8G8R8_SINT:
    case EF_B8G8R8_SRGB:
    case EF_B8G8R8A8_UNORM:
    case EF_B8G8R8A8_SNORM:
    case EF_B8G8R8A8_USCALED:
    case EF_B8G8R8A8_SSCALED:
    case EF_B8G8R8A8_UINT:
    case EF_B8G8R8A8_SINT:
    case EF_B8G8R8A8_SRGB:
    case EF_A8B8G8R8_UNORM_PACK32:
    case EF_A8B8G8R8_SNORM_PACK32:
    case EF_A8B8G8R8_USCALED_PACK32:
    case EF_A8B8G8R8_SSCALED_PACK32:
    case EF_A8B8G8R8_UINT_PACK32:
    case EF_A8B8G8R8_SINT_PACK32:
    case EF_A8B8G8R8_SRGB_PACK32:
    case EF_A2B10G10R10_UNORM_PACK32:
    case EF_A2B10G10R10_SNORM_PACK32:
    case EF_A2B10G10R10_USCALED_PACK32:
    case EF_A2B10G10R10_SSCALED_PACK32:
    case EF_A2B10G10R10_UINT_PACK32:
    case EF_A2B10G10R10_SINT_PACK32:
    case EF_B10G11R11_UFLOAT_PACK32:
    case EF_E5B9G9R9_UFLOAT_PACK32:
        return true;
    default:
        return false;
    }
}

    template<asset::E_FORMAT cf>
    constexpr bool isSignedFormat()
    {
        return is_any_of_values<
            cf,
            EF_R8_SNORM,
            EF_R8_SSCALED,
            EF_R8_SINT,
            EF_R8G8_SNORM,
            EF_R8G8_SSCALED,
            EF_R8G8_SINT,
            EF_R8G8B8_SNORM,
            EF_R8G8B8_SSCALED,
            EF_R8G8B8_SINT,
            EF_B8G8R8_SNORM,
            EF_B8G8R8_SSCALED,
            EF_B8G8R8_SINT,
            EF_R8G8B8A8_SNORM,
            EF_R8G8B8A8_SSCALED,
            EF_R8G8B8A8_SINT,
            EF_B8G8R8A8_SNORM,
            EF_B8G8R8A8_SSCALED,
            EF_B8G8R8A8_SINT,
            EF_A8B8G8R8_SNORM_PACK32,
            EF_A8B8G8R8_SSCALED_PACK32,
            EF_A8B8G8R8_SINT_PACK32,
            EF_A2R10G10B10_SNORM_PACK32,
            EF_A2R10G10B10_SSCALED_PACK32,
            EF_A2R10G10B10_SINT_PACK32,
            EF_A2B10G10R10_SNORM_PACK32,
            EF_A2B10G10R10_SSCALED_PACK32,
            EF_A2B10G10R10_SINT_PACK32,
            EF_R16_SNORM,
            EF_R16_SSCALED,
            EF_R16_SINT,
            EF_R16G16_SNORM,
            EF_R16G16_SSCALED,
            EF_R16G16_SINT,
            EF_R16G16B16_SNORM,
            EF_R16G16B16_SSCALED,
            EF_R16G16B16_SINT,
            EF_R16G16B16A16_SNORM,
            EF_R16G16B16A16_SSCALED,
            EF_R16G16B16A16_SINT,
            EF_R32_SINT,
            EF_R32G32_SINT,
            EF_R32G32B32_SINT,
            EF_R32G32B32A32_SINT,
            EF_R64_SINT,
            EF_R64G64_SINT,
            EF_R64G64B64_SINT,
            EF_R64G64B64A64_SINT,
            EF_R16G16B16_SFLOAT,
            EF_R32G32B32_SFLOAT,
            EF_R64_SFLOAT,
            EF_R64G64_SFLOAT,
            EF_R64G64B64_SFLOAT,
            EF_R64G64B64A64_SFLOAT,
            EF_EAC_R11_SNORM_BLOCK,
            EF_EAC_R11G11_SNORM_BLOCK,
            EF_EAC_R11G11_SNORM_BLOCK,
            EF_EAC_R11G11_SNORM_BLOCK,
            EF_BC4_SNORM_BLOCK,
            EF_BC5_SNORM_BLOCK,
            EF_BC6H_SFLOAT_BLOCK,
            EF_BC6H_SFLOAT_BLOCK
        > ::value;
    }
    template<asset::E_FORMAT cf>
    constexpr bool isIntegerFormat()
    {
        return is_any_of_values<
            cf,
            EF_R8_SINT,
            EF_R8_UINT,
            EF_R8G8_SINT,
            EF_R8G8_UINT,
            EF_R8G8B8_SINT,
            EF_B8G8R8_UINT,
            EF_B8G8R8_SINT,
            EF_R8G8B8A8_SINT,
            EF_B8G8R8A8_UINT,
            EF_B8G8R8A8_SINT,
            EF_R8G8B8_UINT,
            EF_R8G8B8A8_UINT,
            EF_A8B8G8R8_UINT_PACK32,
            EF_A8B8G8R8_SINT_PACK32,
            EF_A2R10G10B10_UINT_PACK32,
            EF_A2R10G10B10_SINT_PACK32,
            EF_A2B10G10R10_UINT_PACK32,
            EF_A2B10G10R10_SINT_PACK32,
            EF_R16_UINT,
            EF_R16_SINT,
            EF_R16G16_UINT,
            EF_R16G16_SINT,
            EF_R16G16B16_UINT,
            EF_R16G16B16_SINT,
            EF_R16G16B16A16_UINT,
            EF_R16G16B16A16_SINT,
            EF_R32_UINT,
            EF_R32_SINT,
            EF_R32G32_UINT,
            EF_R32G32_SINT,
            EF_R32G32B32_UINT,
            EF_R32G32B32_SINT,
            EF_R32G32B32A32_UINT,
            EF_R32G32B32A32_SINT,
            EF_R64_UINT,
            EF_R64_SINT,
            EF_R64G64_UINT,
            EF_R64G64_SINT,
            EF_R64G64B64_UINT,
            EF_R64G64B64_SINT,
            EF_R64G64B64A64_UINT,
            EF_R64G64B64A64_SINT,
            EF_R64G64B64A64_SINT
        > ::value;
    }
    template<asset::E_FORMAT cf>
    constexpr bool isFloatingPointFormat()
    {
        return is_any_of_values<
            cf,
            EF_R16_SFLOAT,
            EF_R16G16_SFLOAT,
            EF_R16G16B16_SFLOAT,
            EF_R16G16B16A16_SFLOAT,
            EF_R32_SFLOAT,
            EF_R32G32_SFLOAT,
            EF_R32G32B32_SFLOAT,
            EF_R32G32B32A32_SFLOAT,
            EF_R64_SFLOAT,
            EF_R64G64_SFLOAT,
            EF_R64G64B64_SFLOAT,
            EF_R64G64B64A64_SFLOAT,
            EF_B10G11R11_UFLOAT_PACK32,
            EF_E5B9G9R9_UFLOAT_PACK32,
            EF_BC6H_SFLOAT_BLOCK,
            EF_BC6H_UFLOAT_BLOCK
        >::value;
    }
    template<asset::E_FORMAT cf>
    constexpr bool isNormalizedFormat()
    {
        return is_any_of_values<
            cf,
            EF_R4G4_UNORM_PACK8,
            EF_R4G4B4A4_UNORM_PACK16,
            EF_B4G4R4A4_UNORM_PACK16,
            EF_R5G6B5_UNORM_PACK16,
            EF_B5G6R5_UNORM_PACK16,
            EF_R5G5B5A1_UNORM_PACK16,
            EF_B5G5R5A1_UNORM_PACK16,
            EF_A1R5G5B5_UNORM_PACK16,
            EF_R8_UNORM,
            EF_R8_SNORM,
            EF_R8G8_UNORM,
            EF_R8G8_SNORM,
            EF_R8G8B8_UNORM,
            EF_R8G8B8_SNORM,
            EF_B8G8R8_UNORM,
            EF_B8G8R8_SNORM,
            EF_R8G8B8A8_UNORM,
            EF_R8G8B8A8_SNORM,
            EF_B8G8R8A8_UNORM,
            EF_B8G8R8A8_SNORM,
            EF_A8B8G8R8_UNORM_PACK32,
            EF_A8B8G8R8_SNORM_PACK32,
            EF_A2R10G10B10_UNORM_PACK32,
            EF_A2R10G10B10_SNORM_PACK32,
            EF_A2B10G10R10_UNORM_PACK32,
            EF_A2B10G10R10_SNORM_PACK32,
            EF_R16_UNORM,
            EF_R16_SNORM,
            EF_R16G16_UNORM,
            EF_R16G16_SNORM,
            EF_R16G16B16_UNORM,
            EF_R16G16B16_SNORM,
            EF_R16G16B16A16_UNORM,
            EF_R16G16B16A16_SNORM,
            EF_G8_B8_R8_3PLANE_420_UNORM,
            EF_G8_B8R8_2PLANE_420_UNORM,
            EF_G8_B8_R8_3PLANE_422_UNORM,
            EF_G8_B8R8_2PLANE_422_UNORM,
            EF_G8_B8_R8_3PLANE_444_UNORM,
            EF_G8_B8_R8_3PLANE_444_UNORM,
            EF_R8_SRGB,
            EF_R8G8_SRGB,
            EF_R8G8B8_SRGB,
            EF_B8G8R8_SRGB,
            EF_R8G8B8A8_SRGB,
            EF_B8G8R8A8_SRGB,
            EF_A8B8G8R8_SRGB_PACK32,
            EF_BC1_RGB_UNORM_BLOCK,
            EF_BC1_RGB_SRGB_BLOCK,
            EF_BC1_RGBA_UNORM_BLOCK,
            EF_BC1_RGBA_SRGB_BLOCK,
            EF_BC2_UNORM_BLOCK,
            EF_BC2_SRGB_BLOCK,
            EF_BC3_UNORM_BLOCK,
            EF_BC3_SRGB_BLOCK,
            EF_BC3_SRGB_BLOCK,
            EF_BC4_SNORM_BLOCK,
            EF_BC4_UNORM_BLOCK,
            EF_BC5_SNORM_BLOCK,
            EF_BC5_UNORM_BLOCK,
            EF_BC7_SRGB_BLOCK,
            EF_BC7_UNORM_BLOCK,
            EF_ASTC_4x4_UNORM_BLOCK,
            EF_ASTC_4x4_SRGB_BLOCK,
            EF_ASTC_5x4_UNORM_BLOCK,
            EF_ASTC_5x4_SRGB_BLOCK,
            EF_ASTC_5x5_UNORM_BLOCK,
            EF_ASTC_5x5_SRGB_BLOCK,
            EF_ASTC_6x5_UNORM_BLOCK,
            EF_ASTC_6x5_SRGB_BLOCK,
            EF_ASTC_6x6_UNORM_BLOCK,
            EF_ASTC_6x6_SRGB_BLOCK,
            EF_ASTC_8x5_UNORM_BLOCK,
            EF_ASTC_8x5_SRGB_BLOCK,
            EF_ASTC_8x6_UNORM_BLOCK,
            EF_ASTC_8x6_SRGB_BLOCK,
            EF_ASTC_8x8_UNORM_BLOCK,
            EF_ASTC_8x8_SRGB_BLOCK,
            EF_ASTC_10x5_UNORM_BLOCK,
            EF_ASTC_10x5_SRGB_BLOCK,
            EF_ASTC_10x6_UNORM_BLOCK,
            EF_ASTC_10x6_SRGB_BLOCK,
            EF_ASTC_10x8_UNORM_BLOCK,
            EF_ASTC_10x8_SRGB_BLOCK,
            EF_ASTC_10x10_UNORM_BLOCK,
            EF_ASTC_10x10_SRGB_BLOCK,
            EF_ASTC_12x10_UNORM_BLOCK,
            EF_ASTC_12x10_SRGB_BLOCK,
            EF_ASTC_12x12_UNORM_BLOCK,
            EF_ASTC_12x12_SRGB_BLOCK,
            EF_ETC2_R8G8B8_UNORM_BLOCK,
            EF_ETC2_R8G8B8_SRGB_BLOCK,
            EF_ETC2_R8G8B8A1_UNORM_BLOCK,
            EF_ETC2_R8G8B8A1_SRGB_BLOCK,
            EF_ETC2_R8G8B8A8_UNORM_BLOCK,
            EF_ETC2_R8G8B8A8_SRGB_BLOCK,
            EF_EAC_R11_UNORM_BLOCK,
            EF_EAC_R11_SNORM_BLOCK,
            EF_EAC_R11G11_UNORM_BLOCK,
            EF_EAC_R11G11_SNORM_BLOCK,
            EF_PVRTC1_2BPP_UNORM_BLOCK_IMG,
            EF_PVRTC1_4BPP_UNORM_BLOCK_IMG,
            EF_PVRTC2_2BPP_UNORM_BLOCK_IMG,
            EF_PVRTC2_4BPP_UNORM_BLOCK_IMG,
            EF_PVRTC1_2BPP_SRGB_BLOCK_IMG,
            EF_PVRTC1_4BPP_SRGB_BLOCK_IMG,
            EF_PVRTC2_2BPP_SRGB_BLOCK_IMG,
            EF_PVRTC2_4BPP_SRGB_BLOCK_IMG,
            EF_PVRTC2_4BPP_SRGB_BLOCK_IMG
        > ::value;
    }
    template<asset::E_FORMAT cf>
    constexpr bool isScaledFormat()
    {
        return is_any_of_values<
            cf,
            EF_R8_USCALED,
            EF_R8_SSCALED,
            EF_R8G8_USCALED,
            EF_R8G8_SSCALED,
            EF_R8G8B8_USCALED,
            EF_R8G8B8_SSCALED,
            EF_B8G8R8_USCALED,
            EF_B8G8R8_SSCALED,
            EF_R8G8B8A8_USCALED,
            EF_R8G8B8A8_SSCALED,
            EF_B8G8R8A8_USCALED,
            EF_B8G8R8A8_SSCALED,
            EF_A8B8G8R8_USCALED_PACK32,
            EF_A8B8G8R8_SSCALED_PACK32,
            EF_A2R10G10B10_USCALED_PACK32,
            EF_A2R10G10B10_SSCALED_PACK32,
            EF_A2B10G10R10_USCALED_PACK32,
            EF_A2B10G10R10_SSCALED_PACK32,
            EF_R16_USCALED,
            EF_R16_SSCALED,
            EF_R16G16_USCALED,
            EF_R16G16_SSCALED,
            EF_R16G16B16_USCALED,
            EF_R16G16B16_SSCALED,
            EF_R16G16B16A16_USCALED,
            EF_R16G16B16A16_SSCALED,
            EF_R16G16B16A16_SSCALED
        > ::value;
    }
    template<asset::E_FORMAT cf>
    constexpr bool isSRGBFormat()
    {
        return is_any_of_values<
            cf,
            EF_R8_SRGB,
            EF_R8G8_SRGB,
            EF_R8G8B8_SRGB,
            EF_B8G8R8_SRGB,
            EF_R8G8B8A8_SRGB,
            EF_B8G8R8A8_SRGB,
            EF_A8B8G8R8_SRGB_PACK32,
            EF_BC7_SRGB_BLOCK,
            EF_ASTC_4x4_SRGB_BLOCK,
            EF_ASTC_5x4_SRGB_BLOCK,
            EF_ASTC_5x5_SRGB_BLOCK,
            EF_ASTC_6x5_SRGB_BLOCK,
            EF_ASTC_6x6_SRGB_BLOCK,
            EF_ASTC_8x5_SRGB_BLOCK,
            EF_ASTC_8x6_SRGB_BLOCK,
            EF_ASTC_8x8_SRGB_BLOCK,
            EF_ASTC_10x5_SRGB_BLOCK,
            EF_ASTC_10x6_SRGB_BLOCK,
            EF_ASTC_10x8_SRGB_BLOCK,
            EF_ASTC_10x10_SRGB_BLOCK,
            EF_ASTC_12x10_SRGB_BLOCK,
            EF_ASTC_12x12_SRGB_BLOCK,
            EF_ETC2_R8G8B8_SRGB_BLOCK,
            EF_ETC2_R8G8B8A1_SRGB_BLOCK,
            EF_ETC2_R8G8B8A8_SRGB_BLOCK,
            EF_PVRTC1_2BPP_SRGB_BLOCK_IMG,
            EF_PVRTC1_4BPP_SRGB_BLOCK_IMG,
            EF_PVRTC2_2BPP_SRGB_BLOCK_IMG,
            EF_PVRTC2_4BPP_SRGB_BLOCK_IMG,
            EF_PVRTC2_4BPP_SRGB_BLOCK_IMG
        >::value;
    }
    template<asset::E_FORMAT cf>
    constexpr bool isBlockCompressionFormat()
    {
        return is_any_of_values<
            cf,
            EF_BC1_RGB_UNORM_BLOCK,
            EF_BC1_RGB_SRGB_BLOCK,
            EF_BC1_RGBA_UNORM_BLOCK,
            EF_BC1_RGBA_SRGB_BLOCK,
            EF_BC2_UNORM_BLOCK,
            EF_BC2_SRGB_BLOCK,
            EF_BC3_UNORM_BLOCK,
            EF_BC3_SRGB_BLOCK,
            EF_BC4_SNORM_BLOCK,
            EF_BC4_UNORM_BLOCK,
            EF_BC5_SNORM_BLOCK,
            EF_BC5_UNORM_BLOCK,
            EF_BC6H_SFLOAT_BLOCK,
            EF_BC6H_UFLOAT_BLOCK,
            EF_BC7_SRGB_BLOCK,
            EF_BC7_UNORM_BLOCK,
            EF_ASTC_4x4_UNORM_BLOCK,
            EF_ASTC_4x4_SRGB_BLOCK,
            EF_ASTC_5x4_UNORM_BLOCK,
            EF_ASTC_5x4_SRGB_BLOCK,
            EF_ASTC_5x5_UNORM_BLOCK,
            EF_ASTC_5x5_SRGB_BLOCK,
            EF_ASTC_6x5_UNORM_BLOCK,
            EF_ASTC_6x5_SRGB_BLOCK,
            EF_ASTC_6x6_UNORM_BLOCK,
            EF_ASTC_6x6_SRGB_BLOCK,
            EF_ASTC_8x5_UNORM_BLOCK,
            EF_ASTC_8x5_SRGB_BLOCK,
            EF_ASTC_8x6_UNORM_BLOCK,
            EF_ASTC_8x6_SRGB_BLOCK,
            EF_ASTC_8x8_UNORM_BLOCK,
            EF_ASTC_8x8_SRGB_BLOCK,
            EF_ASTC_10x5_UNORM_BLOCK,
            EF_ASTC_10x5_SRGB_BLOCK,
            EF_ASTC_10x6_UNORM_BLOCK,
            EF_ASTC_10x6_SRGB_BLOCK,
            EF_ASTC_10x8_UNORM_BLOCK,
            EF_ASTC_10x8_SRGB_BLOCK,
            EF_ASTC_10x10_UNORM_BLOCK,
            EF_ASTC_10x10_SRGB_BLOCK,
            EF_ASTC_12x10_UNORM_BLOCK,
            EF_ASTC_12x10_SRGB_BLOCK,
            EF_ASTC_12x12_UNORM_BLOCK,
            EF_ASTC_12x12_SRGB_BLOCK,
            EF_ETC2_R8G8B8_UNORM_BLOCK,
            EF_ETC2_R8G8B8_SRGB_BLOCK,
            EF_ETC2_R8G8B8A1_UNORM_BLOCK,
            EF_ETC2_R8G8B8A1_SRGB_BLOCK,
            EF_ETC2_R8G8B8A8_UNORM_BLOCK,
            EF_ETC2_R8G8B8A8_SRGB_BLOCK,
            EF_EAC_R11_UNORM_BLOCK,
            EF_EAC_R11_SNORM_BLOCK,
            EF_EAC_R11G11_UNORM_BLOCK,
            EF_EAC_R11G11_SNORM_BLOCK,
            EF_PVRTC1_2BPP_UNORM_BLOCK_IMG,
            EF_PVRTC1_4BPP_UNORM_BLOCK_IMG,
            EF_PVRTC2_2BPP_UNORM_BLOCK_IMG,
            EF_PVRTC2_4BPP_UNORM_BLOCK_IMG,
            EF_PVRTC1_2BPP_SRGB_BLOCK_IMG,
            EF_PVRTC1_4BPP_SRGB_BLOCK_IMG,
            EF_PVRTC2_2BPP_SRGB_BLOCK_IMG,
            EF_PVRTC2_4BPP_SRGB_BLOCK_IMG,
            EF_PVRTC2_4BPP_SRGB_BLOCK_IMG
        >::value;
    }

    inline bool isSignedFormat(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case EF_R8_SNORM:
        case EF_R8_SSCALED:
        case EF_R8_SINT:
        case EF_R8G8_SNORM:
        case EF_R8G8_SSCALED:
        case EF_R8G8_SINT:
        case EF_R8G8B8_SNORM:
        case EF_R8G8B8_SSCALED:
        case EF_R8G8B8_SINT:
        case EF_B8G8R8_SNORM:
        case EF_B8G8R8_SSCALED:
        case EF_B8G8R8_SINT:
        case EF_R8G8B8A8_SNORM:
        case EF_R8G8B8A8_SSCALED:
        case EF_R8G8B8A8_SINT:
        case EF_B8G8R8A8_SNORM:
        case EF_B8G8R8A8_SSCALED:
        case EF_B8G8R8A8_SINT:
        case EF_A8B8G8R8_SNORM_PACK32:
        case EF_A8B8G8R8_SSCALED_PACK32:
        case EF_A8B8G8R8_SINT_PACK32:
        case EF_A2R10G10B10_SNORM_PACK32:
        case EF_A2R10G10B10_SSCALED_PACK32:
        case EF_A2R10G10B10_SINT_PACK32:
        case EF_A2B10G10R10_SNORM_PACK32:
        case EF_A2B10G10R10_SSCALED_PACK32:
        case EF_A2B10G10R10_SINT_PACK32:
        case EF_R16_SNORM:
        case EF_R16_SSCALED:
        case EF_R16_SINT:
        case EF_R16_SFLOAT:
        case EF_R16G16_SNORM:
        case EF_R16G16_SSCALED:
        case EF_R16G16_SINT:
        case EF_R16G16_SFLOAT:
        case EF_R16G16B16_SNORM:
        case EF_R16G16B16_SSCALED:
        case EF_R16G16B16_SINT:
        case EF_R16G16B16_SFLOAT:
        case EF_R16G16B16A16_SNORM:
        case EF_R16G16B16A16_SSCALED:
        case EF_R16G16B16A16_SINT:
        case EF_R16G16B16A16_SFLOAT:
        case EF_R32_SINT:
        case EF_R32_SFLOAT:
        case EF_R32G32_SINT:
        case EF_R32G32_SFLOAT:
        case EF_R32G32B32_SINT:
        case EF_R32G32B32_SFLOAT:
        case EF_R32G32B32A32_SINT:
        case EF_R32G32B32A32_SFLOAT:
        case EF_R64_SINT: 
        case EF_R64_SFLOAT:
        case EF_R64G64_SINT:
        case EF_R64G64_SFLOAT:
        case EF_R64G64B64_SINT:
        case EF_R64G64B64_SFLOAT:
        case EF_R64G64B64A64_SINT:
        case EF_R64G64B64A64_SFLOAT:
        case EF_EAC_R11_SNORM_BLOCK:
        case EF_EAC_R11G11_SNORM_BLOCK:
        case EF_BC4_SNORM_BLOCK:
        case EF_BC5_SNORM_BLOCK:
        case EF_BC6H_SFLOAT_BLOCK:
            return true;
        default: return false;
        }
    }
    inline bool isIntegerFormat(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
            case EF_S8_UINT:
            case EF_R8_SINT:
            case EF_R8_UINT:
            case EF_R8G8_SINT:
            case EF_R8G8_UINT:
            case EF_R8G8B8_SINT:
            case EF_R8G8B8_UINT:
            case EF_B8G8R8_SINT:
            case EF_B8G8R8_UINT:
            case EF_R8G8B8A8_SINT:
            case EF_R8G8B8A8_UINT:
            case EF_B8G8R8A8_SINT:
            case EF_B8G8R8A8_UINT:
            case EF_A8B8G8R8_UINT_PACK32:
            case EF_A8B8G8R8_SINT_PACK32:
            case EF_A2R10G10B10_UINT_PACK32:
            case EF_A2R10G10B10_SINT_PACK32:
            case EF_A2B10G10R10_UINT_PACK32:
            case EF_A2B10G10R10_SINT_PACK32:
            case EF_R16_UINT:
            case EF_R16_SINT:
            case EF_R16G16_UINT:
            case EF_R16G16_SINT:
            case EF_R16G16B16_UINT:
            case EF_R16G16B16_SINT:
            case EF_R16G16B16A16_UINT:
            case EF_R16G16B16A16_SINT:
            case EF_R32_UINT:
            case EF_R32_SINT:
            case EF_R32G32_UINT:
            case EF_R32G32_SINT:
            case EF_R32G32B32_UINT:
            case EF_R32G32B32_SINT:
            case EF_R32G32B32A32_UINT:
            case EF_R32G32B32A32_SINT:
            case EF_R64_UINT:
            case EF_R64_SINT:
            case EF_R64G64_UINT:
            case EF_R64G64_SINT:
            case EF_R64G64B64_UINT:
            case EF_R64G64B64_SINT:
            case EF_R64G64B64A64_UINT:
            case EF_R64G64B64A64_SINT:
                return true;
            default:
                return false;
        }
    }
    inline bool isFloatingPointFormat(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case EF_D32_SFLOAT:
        case EF_R16_SFLOAT:
        case EF_R16G16_SFLOAT:
        case EF_R16G16B16_SFLOAT:
        case EF_R16G16B16A16_SFLOAT:
        case EF_R32_SFLOAT:
        case EF_R32G32_SFLOAT:
        case EF_R32G32B32_SFLOAT:
        case EF_R32G32B32A32_SFLOAT:
        case EF_R64_SFLOAT:
        case EF_R64G64_SFLOAT:
        case EF_R64G64B64_SFLOAT:
        case EF_R64G64B64A64_SFLOAT:
        case EF_B10G11R11_UFLOAT_PACK32:
        case EF_E5B9G9R9_UFLOAT_PACK32:
        case EF_BC6H_SFLOAT_BLOCK:
        case EF_BC6H_UFLOAT_BLOCK:
            return true;
        default: return false;
        }
    }
    //! Note that scaled formats are subset of normalized formats
    inline bool isNormalizedFormat(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case EF_D16_UNORM:
        case EF_X8_D24_UNORM_PACK32:
        case EF_D16_UNORM_S8_UINT:
        case EF_D24_UNORM_S8_UINT:
        case EF_R4G4_UNORM_PACK8:
        case EF_R4G4B4A4_UNORM_PACK16:
        case EF_B4G4R4A4_UNORM_PACK16:
        case EF_R5G6B5_UNORM_PACK16:
        case EF_B5G6R5_UNORM_PACK16:
        case EF_R5G5B5A1_UNORM_PACK16:
        case EF_B5G5R5A1_UNORM_PACK16:
        case EF_A1R5G5B5_UNORM_PACK16:
        case EF_R8_UNORM:
        case EF_R8_SNORM:
        case EF_R8G8_UNORM:
        case EF_R8G8_SNORM:
        case EF_R8G8B8_UNORM:
        case EF_R8G8B8_SNORM:
        case EF_B8G8R8_UNORM:
        case EF_B8G8R8_SNORM:
        case EF_R8G8B8A8_UNORM:
        case EF_R8G8B8A8_SNORM:
        case EF_B8G8R8A8_UNORM:
        case EF_B8G8R8A8_SNORM:
        case EF_A8B8G8R8_UNORM_PACK32:
        case EF_A8B8G8R8_SNORM_PACK32:
        case EF_A2R10G10B10_UNORM_PACK32:
        case EF_A2R10G10B10_SNORM_PACK32:
        case EF_A2B10G10R10_UNORM_PACK32:
        case EF_A2B10G10R10_SNORM_PACK32:
        case EF_R16_UNORM:
        case EF_R16_SNORM:
        case EF_R16G16_UNORM:
        case EF_R16G16_SNORM:
        case EF_R16G16B16_UNORM:
        case EF_R16G16B16_SNORM:
        case EF_R16G16B16A16_UNORM:
        case EF_R16G16B16A16_SNORM:
        case EF_R8_SRGB:
        case EF_R8G8_SRGB:
        case EF_R8G8B8_SRGB:
        case EF_B8G8R8_SRGB:
        case EF_R8G8B8A8_SRGB:
        case EF_B8G8R8A8_SRGB:
        case EF_A8B8G8R8_SRGB_PACK32:
        case EF_BC1_RGB_UNORM_BLOCK:
        case EF_BC1_RGB_SRGB_BLOCK:
        case EF_BC1_RGBA_UNORM_BLOCK:
        case EF_BC1_RGBA_SRGB_BLOCK:
        case EF_BC2_UNORM_BLOCK:
        case EF_BC2_SRGB_BLOCK:
        case EF_BC3_UNORM_BLOCK:
        case EF_BC3_SRGB_BLOCK:
        case EF_ASTC_4x4_UNORM_BLOCK:
        case EF_ASTC_4x4_SRGB_BLOCK:
        case EF_ASTC_5x4_UNORM_BLOCK:
        case EF_ASTC_5x4_SRGB_BLOCK:
        case EF_ASTC_5x5_UNORM_BLOCK:
        case EF_ASTC_5x5_SRGB_BLOCK:
        case EF_ASTC_6x5_UNORM_BLOCK:
        case EF_ASTC_6x5_SRGB_BLOCK:
        case EF_ASTC_6x6_UNORM_BLOCK:
        case EF_ASTC_6x6_SRGB_BLOCK:
        case EF_ASTC_8x5_UNORM_BLOCK:
        case EF_ASTC_8x5_SRGB_BLOCK:
        case EF_ASTC_8x6_UNORM_BLOCK:
        case EF_ASTC_8x6_SRGB_BLOCK:
        case EF_ASTC_8x8_UNORM_BLOCK:
        case EF_ASTC_8x8_SRGB_BLOCK:
        case EF_ASTC_10x5_UNORM_BLOCK:
        case EF_ASTC_10x5_SRGB_BLOCK:
        case EF_ASTC_10x6_UNORM_BLOCK:
        case EF_ASTC_10x6_SRGB_BLOCK:
        case EF_ASTC_10x8_UNORM_BLOCK:
        case EF_ASTC_10x8_SRGB_BLOCK:
        case EF_ASTC_10x10_UNORM_BLOCK:
        case EF_ASTC_10x10_SRGB_BLOCK:
        case EF_ASTC_12x10_UNORM_BLOCK:
        case EF_ASTC_12x10_SRGB_BLOCK:
        case EF_ASTC_12x12_UNORM_BLOCK:
        case EF_ASTC_12x12_SRGB_BLOCK:
        case EF_ETC2_R8G8B8_UNORM_BLOCK:
        case EF_ETC2_R8G8B8_SRGB_BLOCK:
        case EF_ETC2_R8G8B8A1_UNORM_BLOCK:
        case EF_ETC2_R8G8B8A1_SRGB_BLOCK:
        case EF_ETC2_R8G8B8A8_UNORM_BLOCK:
        case EF_ETC2_R8G8B8A8_SRGB_BLOCK:
        case EF_EAC_R11_UNORM_BLOCK:
        case EF_EAC_R11_SNORM_BLOCK:
        case EF_EAC_R11G11_UNORM_BLOCK:
        case EF_EAC_R11G11_SNORM_BLOCK:
        case EF_BC4_SNORM_BLOCK:
        case EF_BC4_UNORM_BLOCK:
        case EF_BC5_SNORM_BLOCK:
        case EF_BC5_UNORM_BLOCK:
        case EF_BC7_SRGB_BLOCK:
        case EF_BC7_UNORM_BLOCK:
        case EF_PVRTC1_2BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC1_4BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC2_2BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC2_4BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
            return true;
        default: return false;
        }
    }
    //SCALED implies NORMALIZED!
    inline bool isScaledFormat(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case EF_R8_USCALED:
        case EF_R8_SSCALED:
        case EF_R8G8_USCALED:
        case EF_R8G8_SSCALED:
        case EF_R8G8B8_USCALED:
        case EF_R8G8B8_SSCALED:
        case EF_B8G8R8_USCALED:
        case EF_B8G8R8_SSCALED:
        case EF_R8G8B8A8_USCALED:
        case EF_R8G8B8A8_SSCALED:
        case EF_B8G8R8A8_USCALED:
        case EF_B8G8R8A8_SSCALED:
        case EF_A8B8G8R8_USCALED_PACK32:
        case EF_A8B8G8R8_SSCALED_PACK32:
        case EF_A2R10G10B10_USCALED_PACK32:
        case EF_A2R10G10B10_SSCALED_PACK32:
        case EF_A2B10G10R10_USCALED_PACK32:
        case EF_A2B10G10R10_SSCALED_PACK32:
        case EF_R16_USCALED:
        case EF_R16_SSCALED:
        case EF_R16G16_USCALED:
        case EF_R16G16_SSCALED:
        case EF_R16G16B16_USCALED:
        case EF_R16G16B16_SSCALED:
        case EF_R16G16B16A16_USCALED:
        case EF_R16G16B16A16_SSCALED:
            return true;
        default: return false;
        }
    }
    inline bool isSRGBFormat(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case EF_R8_SRGB:
        case EF_R8G8_SRGB:
        case EF_R8G8B8_SRGB:
        case EF_B8G8R8_SRGB:
        case EF_R8G8B8A8_SRGB:
        case EF_B8G8R8A8_SRGB:
        case EF_A8B8G8R8_SRGB_PACK32:
        case EF_BC1_RGB_SRGB_BLOCK:
        case EF_BC1_RGBA_SRGB_BLOCK:
        case EF_BC2_SRGB_BLOCK:
        case EF_BC3_SRGB_BLOCK:
        case EF_BC7_SRGB_BLOCK:
        case EF_ASTC_4x4_SRGB_BLOCK:
        case EF_ASTC_5x4_SRGB_BLOCK:
        case EF_ASTC_5x5_SRGB_BLOCK:
        case EF_ASTC_6x5_SRGB_BLOCK:
        case EF_ASTC_6x6_SRGB_BLOCK:
        case EF_ASTC_8x5_SRGB_BLOCK:
        case EF_ASTC_8x6_SRGB_BLOCK:
        case EF_ASTC_8x8_SRGB_BLOCK:
        case EF_ASTC_10x5_SRGB_BLOCK:
        case EF_ASTC_10x6_SRGB_BLOCK:
        case EF_ASTC_10x8_SRGB_BLOCK:
        case EF_ASTC_10x10_SRGB_BLOCK:
        case EF_ASTC_12x10_SRGB_BLOCK:
        case EF_ASTC_12x12_SRGB_BLOCK:
        case EF_ETC2_R8G8B8_SRGB_BLOCK:
        case EF_ETC2_R8G8B8A1_SRGB_BLOCK:
        case EF_ETC2_R8G8B8A8_SRGB_BLOCK:
        case EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
            return true;
        default: return false;
        }
    }
    inline bool isBlockCompressionFormat(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case EF_BC1_RGB_UNORM_BLOCK:
        case EF_BC1_RGB_SRGB_BLOCK:
        case EF_BC1_RGBA_UNORM_BLOCK:
        case EF_BC1_RGBA_SRGB_BLOCK:
        case EF_BC2_UNORM_BLOCK:
        case EF_BC2_SRGB_BLOCK:
        case EF_BC3_UNORM_BLOCK:
        case EF_BC3_SRGB_BLOCK:
        case EF_BC4_SNORM_BLOCK:
        case EF_BC4_UNORM_BLOCK:
        case EF_BC5_SNORM_BLOCK:
        case EF_BC5_UNORM_BLOCK:
        case EF_BC6H_SFLOAT_BLOCK:
        case EF_BC6H_UFLOAT_BLOCK:
        case EF_BC7_SRGB_BLOCK:
        case EF_BC7_UNORM_BLOCK:
        case EF_ASTC_4x4_UNORM_BLOCK:
        case EF_ASTC_4x4_SRGB_BLOCK:
        case EF_ASTC_5x4_UNORM_BLOCK:
        case EF_ASTC_5x4_SRGB_BLOCK:
        case EF_ASTC_5x5_UNORM_BLOCK:
        case EF_ASTC_5x5_SRGB_BLOCK:
        case EF_ASTC_6x5_UNORM_BLOCK:
        case EF_ASTC_6x5_SRGB_BLOCK:
        case EF_ASTC_6x6_UNORM_BLOCK:
        case EF_ASTC_6x6_SRGB_BLOCK:
        case EF_ASTC_8x5_UNORM_BLOCK:
        case EF_ASTC_8x5_SRGB_BLOCK:
        case EF_ASTC_8x6_UNORM_BLOCK:
        case EF_ASTC_8x6_SRGB_BLOCK:
        case EF_ASTC_8x8_UNORM_BLOCK:
        case EF_ASTC_8x8_SRGB_BLOCK:
        case EF_ASTC_10x5_UNORM_BLOCK:
        case EF_ASTC_10x5_SRGB_BLOCK:
        case EF_ASTC_10x6_UNORM_BLOCK:
        case EF_ASTC_10x6_SRGB_BLOCK:
        case EF_ASTC_10x8_UNORM_BLOCK:
        case EF_ASTC_10x8_SRGB_BLOCK:
        case EF_ASTC_10x10_UNORM_BLOCK:
        case EF_ASTC_10x10_SRGB_BLOCK:
        case EF_ASTC_12x10_UNORM_BLOCK:
        case EF_ASTC_12x10_SRGB_BLOCK:
        case EF_ASTC_12x12_UNORM_BLOCK:
        case EF_ASTC_12x12_SRGB_BLOCK:
        case EF_ETC2_R8G8B8_UNORM_BLOCK:
        case EF_ETC2_R8G8B8_SRGB_BLOCK:
        case EF_ETC2_R8G8B8A1_UNORM_BLOCK:
        case EF_ETC2_R8G8B8A1_SRGB_BLOCK:
        case EF_ETC2_R8G8B8A8_UNORM_BLOCK:
        case EF_ETC2_R8G8B8A8_SRGB_BLOCK:
        case EF_EAC_R11_UNORM_BLOCK:
        case EF_EAC_R11_SNORM_BLOCK:
        case EF_EAC_R11G11_UNORM_BLOCK:
        case EF_EAC_R11G11_SNORM_BLOCK:
        case EF_PVRTC1_2BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC1_4BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC2_2BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC2_4BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
            return true;
        default: return false;
        }
    }
    inline bool isPlanarFormat(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case EF_G8_B8_R8_3PLANE_420_UNORM:
        case EF_G8_B8R8_2PLANE_420_UNORM:
        case EF_G8_B8_R8_3PLANE_422_UNORM:
        case EF_G8_B8R8_2PLANE_422_UNORM:
        case EF_G8_B8_R8_3PLANE_444_UNORM:
            return true;
        default: return false;
        }
    }
    template<asset::E_FORMAT cf>
    constexpr bool isPlanarFormat()
    {
        return is_any_of_values<
            cf,
            EF_G8_B8_R8_3PLANE_420_UNORM,
            EF_G8_B8R8_2PLANE_420_UNORM,
            EF_G8_B8_R8_3PLANE_422_UNORM,
            EF_G8_B8R8_2PLANE_422_UNORM,
            EF_G8_B8_R8_3PLANE_444_UNORM,
            EF_G8_B8_R8_3PLANE_444_UNORM
        >::value;
    }

    inline void getHorizontalReductionFactorPerPlane(asset::E_FORMAT _planarFmt, uint32_t _reductionFactor[4])
    {
        switch (_planarFmt)
        {
        case EF_G8_B8_R8_3PLANE_420_UNORM:
        case EF_G8_B8_R8_3PLANE_422_UNORM:
            _reductionFactor[0] = 1u;
            _reductionFactor[1] = _reductionFactor[2] = 2u;
            return;
        case EF_G8_B8R8_2PLANE_420_UNORM:
        case EF_G8_B8R8_2PLANE_422_UNORM:
            _reductionFactor[0] = 1u;
            _reductionFactor[1] = 2u;
            return;
        case EF_G8_B8_R8_3PLANE_444_UNORM:
            _reductionFactor[0] = _reductionFactor[1] = _reductionFactor[2] = 1u;
            return;
        default:
            return;
        }
    }
    inline void getVerticalReductionFactorPerPlane(asset::E_FORMAT _planarFmt, uint32_t _reductionFactor[4])
    {
        switch (_planarFmt)
        {
        case EF_G8_B8_R8_3PLANE_420_UNORM:
            _reductionFactor[0] = 1u;
            _reductionFactor[1] = _reductionFactor[2] = 2u;
            return;
        case EF_G8_B8R8_2PLANE_420_UNORM:
            _reductionFactor[0] = 1u;
            _reductionFactor[1] = 2u;
            return;
        case EF_G8_B8_R8_3PLANE_422_UNORM:
        case EF_G8_B8R8_2PLANE_422_UNORM:
        case EF_G8_B8_R8_3PLANE_444_UNORM:
            _reductionFactor[0] = _reductionFactor[1] = _reductionFactor[2] = 1u;
            return;
        default:
            return;
        }
    }
    inline void getChannelsPerPlane(asset::E_FORMAT _planarFmt, uint32_t _chCnt[4])
    {
        switch (_planarFmt)
        {
        case EF_G8_B8_R8_3PLANE_420_UNORM:
        case EF_G8_B8_R8_3PLANE_422_UNORM:
        case EF_G8_B8_R8_3PLANE_444_UNORM:
            _chCnt[0] = _chCnt[1] = _chCnt[2] = 1u;
            return;
        case EF_G8_B8R8_2PLANE_420_UNORM:
        case EF_G8_B8R8_2PLANE_422_UNORM:
            _chCnt[0] = 1u;
            _chCnt[1] = 2u;
            return;
        default:
            return;
        }
    }


// ! Precision

template<E_FORMAT format>
struct format_interm_storage_type
{
    using type = typename std::conditional<isIntegerFormat<format>(),typename std::conditional<isSignedFormat<format>(),int64_t,uint64_t>::type,double>::type;
};

// TODO: add precision functions 
/*
constexpr getFormatMinValue<E_FORMAT>(channel)
inline value_type getFormatMinValue<value_type>(format,channel)

constexpr getFormatMaxValue<E_FORMAT>(channel)
inline value_type getFormatMaxValue<value_type>(format,channel)

*/
template <typename value_type>
inline value_type getFormatMaxValue(E_FORMAT format, uint32_t channel)
{
    const bool _signed = isSignedFormat(format);
    if (isIntegerFormat(format) || isScaledFormat(format))
    {
        switch (format)
        {
        case EF_A2R10G10B10_UINT_PACK32:
        case EF_A2B10G10R10_UINT_PACK32:
            return (channel == 3u) ? 3 : 1023;
        case EF_A2R10G10B10_SINT_PACK32:
        case EF_A2B10G10R10_SINT_PACK32:
            return (channel == 3u) ? 3 : 1023;
        default: break;
        }

        auto bytesPerChannel = (getBytesPerPixel(format)*core::rational(1,getFormatChannelCount(format))).getIntegerApprox();
        if (_signed)
        {
            switch (bytesPerChannel)
            {
            case 1u: return SCHAR_MAX;
            case 2u: return SHRT_MAX;
            case 4u: return INT_MAX;
            case 8u: return LLONG_MAX;
            default: break;
            }
        }
        else
        {
            switch (bytesPerChannel)
            {
            case 1u: return UCHAR_MAX;
            case 2u: return USHRT_MAX;
            case 4u: return UINT_MAX;
            case 8u: return ULLONG_MAX;
            default: break;
            }
        }
    }
    else if (isNormalizedFormat(format))
    {
        return 1;
    }
    else if (isFloatingPointFormat(format))
    {
        switch (format)
        {
            case EF_B10G11R11_UFLOAT_PACK32:
                if (channel<=1)
                    return 65520;
                else if (channel==2)
                    return 65504;
                break;
            case EF_E5B9G9R9_UFLOAT_PACK32:
                if (channel<3)
                    return 32704;
                break;
            case EF_BC6H_SFLOAT_BLOCK: return 32767;
            case EF_BC6H_UFLOAT_BLOCK: return 65504;
            default: break;
        }

        auto bytesPerChannel = (getBytesPerPixel(format)*core::rational(1,getFormatChannelCount(format))).getIntegerApprox();
        switch (bytesPerChannel)
        {
            case 2u: return 65504;
            case 4u: return FLT_MAX;
            case 8u: return DBL_MAX;
            default: break;
        }
    }
    return 0;
}

template <typename value_type>
inline value_type getFormatMinValue(E_FORMAT format, uint32_t channel)
{
    const bool _signed = isSignedFormat(format);
    if (!_signed)
        return 0;
    if (isIntegerFormat(format) || isScaledFormat(format))
    {
        switch (format)
        {
        case EF_A2R10G10B10_SINT_PACK32:
        case EF_A2B10G10R10_SINT_PACK32:
            return (channel == 3u) ? 0 : -1023;
        default: break;
        }

        auto bytesPerChannel = (getBytesPerPixel(format)*core::rational(1,getFormatChannelCount(format))).getIntegerApprox();
        switch (bytesPerChannel)
        {
        case 1u: return SCHAR_MIN;
        case 2u: return SHRT_MIN;
        case 4u: return INT_MIN;
        case 8u: return LLONG_MIN;
        default: break;
        }
    }
    else if (isNormalizedFormat(format))
    {
        return _signed ? -1 : 0;
    }
    else if (isFloatingPointFormat(format))
    {
        switch (format)
        {
        case EF_BC6H_SFLOAT_BLOCK: return -32767;
        default: break;
        }

        auto bytesPerChannel = (getBytesPerPixel(format)*core::rational(1,getFormatChannelCount(format))).getIntegerApprox();
        switch (bytesPerChannel)
        {
        case 2u: return -65504;
        case 4u: return -FLT_MAX;
        case 8u: return -DBL_MAX;
        default: break;
        }
    }
    return 0;
}

// in SFLOAT and SRGB formats, the precision is dependant on the current value of the channel
template <typename value_type>
inline value_type getFormatPrecision(E_FORMAT format, uint32_t channel, value_type value)
{
    _NBL_DEBUG_BREAK_IF(isBlockCompressionFormat(format)); //????

    if (isIntegerFormat(format) || isScaledFormat(format))
        return 1;

    if (isSRGBFormat(format))
    {
        if (channel==3u)
            return 1.0/255.0;
        return core::srgb2lin(value+1.0/255.0)-core::srgb2lin(value);
    }
    else if (isNormalizedFormat(format))
    {
        switch (format)
        {
        case EF_A2R10G10B10_UNORM_PACK32: [[fallthrough]];
        case EF_A2R10G10B10_SNORM_PACK32: [[fallthrough]];
        case EF_A2B10G10R10_UNORM_PACK32: [[fallthrough]];
        case EF_A2B10G10R10_SNORM_PACK32:
            return (channel==3u) ? 1.0/3.0 : 1.0/1023.0;
        case EF_R4G4_UNORM_PACK8:
            return 1.0/15.0;
        case EF_R4G4B4A4_UNORM_PACK16:
            return 1.0/15.0;
        case EF_B4G4R4A4_UNORM_PACK16:
            return 1.0/15.0;
        case EF_R5G6B5_UNORM_PACK16: [[fallthrough]];
        case EF_B5G6R5_UNORM_PACK16:
            return (channel==1u) ? (1.0/63.0) : (1.0/31.0);
        case EF_R5G5B5A1_UNORM_PACK16: [[fallthrough]];
        case EF_B5G5R5A1_UNORM_PACK16: [[fallthrough]];
        case EF_A1R5G5B5_UNORM_PACK16:
            return (channel==3u) ? 1.0 : (1.0/31.0);
        default: break;
        }

        auto bytesPerChannel = (getBytesPerPixel(format)*core::rational(1,getFormatChannelCount(format))).getIntegerApprox();
        switch (bytesPerChannel)
        {
        case 1u:
            return isSignedFormat(format) ? 1.0/127.0 : 1.0/255.0;
        case 2u:
            return isSignedFormat(format) ? 1.0/32765.0 : 1.0/65535.0;
        default: break;
        }
    }
    else if (isFloatingPointFormat(format))
    {
        switch (format)
        {
            // unsigned values are always ordered as + 1
            case EF_B10G11R11_UFLOAT_PACK32: [[fallthrough]];
            case EF_E5B9G9R9_UFLOAT_PACK32: // TODO: probably need to change signature and take all values?
            {
                float f = std::abs(static_cast<float>(value));
                int bitshift;
                if (format==EF_B10G11R11_UFLOAT_PACK32)
                    bitshift = channel==2u ? 6:5;
                else
                    bitshift = 4;

                uint16_t f16 = core::Float16Compressor::compress(f);
                uint16_t enc = f16 >> bitshift;
                uint16_t next_f16 = (enc + 1) << bitshift;

                return core::Float16Compressor::decompress(next_f16) - f;
            }
            default: break;
        }
        auto bytesPerChannel = (getBytesPerPixel(format)*core::rational(1,getFormatChannelCount(format))).getIntegerApprox();
        switch (bytesPerChannel)
        {
            case 2u:
            {
                float f = std::abs(static_cast<float>(value));
                uint16_t f16 = core::Float16Compressor::compress(f);
                uint16_t dir = core::Float16Compressor::compress(2.f*(f+1.f));
                return core::Float16Compressor::decompress( core::nextafter16(f16, dir) ) - f;
            }
            case 4u:
            {
                float f32 = std::abs(static_cast<float>(value));
                return core::nextafter32(f32,2.f*(f32+1.f))-f32;
            }
            case 8u:
            {
                double f64 = std::abs(static_cast<double>(value));
                return core::nextafter64(f64,2.0*(f64+1.0))-f64;
            }
            default: break;
        }
    }

    return 0;
}

}

namespace std
{
    template <>
    struct hash<nbl::asset::E_FORMAT>
    {
        std::size_t operator()(const nbl::asset::E_FORMAT& k) const noexcept { return k; }
    };
}

#endif