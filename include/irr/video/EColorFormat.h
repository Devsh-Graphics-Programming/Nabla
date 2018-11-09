#ifndef __IRR_E_COLOR_H_INCLUDED__
#define __IRR_E_COLOR_H_INCLUDED__

#include <cstdint>
#include <type_traits>
#include "IrrCompileConfig.h"
#include "vector3d.h"

namespace irr
{
namespace video
{
	//! An enum for the color format of textures used by the Irrlicht Engine.
	/** A color format specifies how color information is stored. */
	enum ECOLOR_FORMAT
	{
		//! 16 bit color format used by the software driver.
		/** It is thus preferred by all other irrlicht engine video drivers.
		There are 5 bits for every color component, and a single bit is left
		for alpha information. */
		ECF_A1R5G5B5 = 0, // this doesn't have equivalent in Vulkan (???)
        ECF_R5G6B5, // this also doesn't have vulkan equivalent..

        //! In all freaking honesty, use texture view objects to cast between same bitsize pixel formats
        // remove this?? todo
        ECF_8BIT_PIX,
        ECF_16BIT_PIX,
        ECF_24BIT_PIX,
        ECF_32BIT_PIX,
        ECF_48BIT_PIX,
        ECF_64BIT_PIX,
        ECF_96BIT_PIX,
        ECF_128BIT_PIX,

        //! Custom shizz we wont ever use
        ECF_DEPTH16,
        ECF_DEPTH24,
        ECF_DEPTH32F,
        ECF_DEPTH24_STENCIL8,
        ECF_DEPTH32F_STENCIL8,
        ECF_STENCIL8,

        //! Vulkan
        ECF_R4G4_UNORM_PACK8,
        ECF_R4G4B4A4_UNORM_PACK16,
        ECF_B4G4R4A4_UNORM_PACK16,
        ECF_R5G6B5_UNORM_PACK16,
        ECF_B5G6R5_UNORM_PACK16,
        ECF_R5G5B5A1_UNORM_PACK16,
        ECF_B5G5R5A1_UNORM_PACK16,
        ECF_A1R5G5B5_UNORM_PACK16,
        ECF_R8_UNORM,
        ECF_R8_SNORM,
        ECF_R8_USCALED,
        ECF_R8_SSCALED,
        ECF_R8_UINT,
        ECF_R8_SINT,
        ECF_R8_SRGB,
        ECF_R8G8_UNORM,
        ECF_R8G8_SNORM,
        ECF_R8G8_USCALED,
        ECF_R8G8_SSCALED,
        ECF_R8G8_UINT,
        ECF_R8G8_SINT,
        ECF_R8G8_SRGB,
        ECF_R8G8B8_UNORM,
        ECF_R8G8B8_SNORM,
        ECF_R8G8B8_USCALED,
        ECF_R8G8B8_SSCALED,
        ECF_R8G8B8_UINT,
        ECF_R8G8B8_SINT,
        ECF_R8G8B8_SRGB,
        ECF_B8G8R8_UNORM,
        ECF_B8G8R8_SNORM,
        ECF_B8G8R8_USCALED,
        ECF_B8G8R8_SSCALED,
        ECF_B8G8R8_UINT,
        ECF_B8G8R8_SINT,
        ECF_B8G8R8_SRGB,
        ECF_R8G8B8A8_UNORM,
        ECF_R8G8B8A8_SNORM,
        ECF_R8G8B8A8_USCALED,
        ECF_R8G8B8A8_SSCALED,
        ECF_R8G8B8A8_UINT,
        ECF_R8G8B8A8_SINT,
        ECF_R8G8B8A8_SRGB,
        ECF_B8G8R8A8_UNORM,
        ECF_B8G8R8A8_SNORM,
        ECF_B8G8R8A8_USCALED,
        ECF_B8G8R8A8_SSCALED,
        ECF_B8G8R8A8_UINT,
        ECF_B8G8R8A8_SINT,
        ECF_B8G8R8A8_SRGB,
        ECF_A8B8G8R8_UNORM_PACK32,
        ECF_A8B8G8R8_SNORM_PACK32,
        ECF_A8B8G8R8_USCALED_PACK32,
        ECF_A8B8G8R8_SSCALED_PACK32,
        ECF_A8B8G8R8_UINT_PACK32,
        ECF_A8B8G8R8_SINT_PACK32,
        ECF_A8B8G8R8_SRGB_PACK32,
        ECF_A2R10G10B10_UNORM_PACK32,
        ECF_A2R10G10B10_SNORM_PACK32,
        ECF_A2R10G10B10_USCALED_PACK32,
        ECF_A2R10G10B10_SSCALED_PACK32,
        ECF_A2R10G10B10_UINT_PACK32,
        ECF_A2R10G10B10_SINT_PACK32,
        ECF_A2B10G10R10_UNORM_PACK32,
        ECF_A2B10G10R10_SNORM_PACK32,
        ECF_A2B10G10R10_USCALED_PACK32,
        ECF_A2B10G10R10_SSCALED_PACK32,
        ECF_A2B10G10R10_UINT_PACK32,
        ECF_A2B10G10R10_SINT_PACK32,
        ECF_R16_UNORM,
        ECF_R16_SNORM,
        ECF_R16_USCALED,
        ECF_R16_SSCALED,
        ECF_R16_UINT,
        ECF_R16_SINT,
        ECF_R16_SFLOAT,
        ECF_R16G16_UNORM,
        ECF_R16G16_SNORM,
        ECF_R16G16_USCALED,
        ECF_R16G16_SSCALED,
        ECF_R16G16_UINT,
        ECF_R16G16_SINT,
        ECF_R16G16_SFLOAT,
        ECF_R16G16B16_UNORM,
        ECF_R16G16B16_SNORM,
        ECF_R16G16B16_USCALED,
        ECF_R16G16B16_SSCALED,
        ECF_R16G16B16_UINT,
        ECF_R16G16B16_SINT,
        ECF_R16G16B16_SFLOAT,
        ECF_R16G16B16A16_UNORM,
        ECF_R16G16B16A16_SNORM,
        ECF_R16G16B16A16_USCALED,
        ECF_R16G16B16A16_SSCALED,
        ECF_R16G16B16A16_UINT,
        ECF_R16G16B16A16_SINT,
        ECF_R16G16B16A16_SFLOAT,
        ECF_R32_UINT,
        ECF_R32_SINT,
        ECF_R32_SFLOAT,
        ECF_R32G32_UINT,
        ECF_R32G32_SINT,
        ECF_R32G32_SFLOAT,
        ECF_R32G32B32_UINT,
        ECF_R32G32B32_SINT,
        ECF_R32G32B32_SFLOAT,
        ECF_R32G32B32A32_UINT,
        ECF_R32G32B32A32_SINT,
        ECF_R32G32B32A32_SFLOAT,
        ECF_R64_UINT,
        ECF_R64_SINT,
        ECF_R64_SFLOAT,
        ECF_R64G64_UINT,
        ECF_R64G64_SINT,
        ECF_R64G64_SFLOAT,
        ECF_R64G64B64_UINT,
        ECF_R64G64B64_SINT,
        ECF_R64G64B64_SFLOAT,
        ECF_R64G64B64A64_UINT,
        ECF_R64G64B64A64_SINT,
        ECF_R64G64B64A64_SFLOAT,
        ECF_B10G11R11_UFLOAT_PACK32,
        ECF_E5B9G9R9_UFLOAT_PACK32,
        ECF_D16_UNORM,
        ECF_X8_D24_UNORM_PACK32,
        ECF_D16_UNORM_S8_UINT,
        ECF_D24_UNORM_S8_UINT,
        ECF_D32_SFLOAT_S8_UINT,

        //! Block Compression Formats!
        ECF_BC1_RGB_UNORM_BLOCK,
        ECF_BC1_RGB_SRGB_BLOCK,
        ECF_BC1_RGBA_UNORM_BLOCK,
        ECF_BC1_RGBA_SRGB_BLOCK,
        ECF_BC2_UNORM_BLOCK,
        ECF_BC2_SRGB_BLOCK,
        ECF_BC3_UNORM_BLOCK,
        ECF_BC3_SRGB_BLOCK,
        ECF_ASTC_4x4_UNORM_BLOCK,
        ECF_ASTC_4x4_SRGB_BLOCK,
        ECF_ASTC_5x4_UNORM_BLOCK,
        ECF_ASTC_5x4_SRGB_BLOCK,
        ECF_ASTC_5x5_UNORM_BLOCK,
        ECF_ASTC_5x5_SRGB_BLOCK,
        ECF_ASTC_6x5_UNORM_BLOCK,
        ECF_ASTC_6x5_SRGB_BLOCK,
        ECF_ASTC_6x6_UNORM_BLOCK,
        ECF_ASTC_6x6_SRGB_BLOCK,
        ECF_ASTC_8x5_UNORM_BLOCK,
        ECF_ASTC_8x5_SRGB_BLOCK,
        ECF_ASTC_8x6_UNORM_BLOCK,
        ECF_ASTC_8x6_SRGB_BLOCK,
        ECF_ASTC_8x8_UNORM_BLOCK,
        ECF_ASTC_8x8_SRGB_BLOCK,
        ECF_ASTC_10x5_UNORM_BLOCK,
        ECF_ASTC_10x5_SRGB_BLOCK,
        ECF_ASTC_10x6_UNORM_BLOCK,
        ECF_ASTC_10x6_SRGB_BLOCK,
        ECF_ASTC_10x8_UNORM_BLOCK,
        ECF_ASTC_10x8_SRGB_BLOCK,
        ECF_ASTC_10x10_UNORM_BLOCK,
        ECF_ASTC_10x10_SRGB_BLOCK,
        ECF_ASTC_12x10_UNORM_BLOCK,
        ECF_ASTC_12x10_SRGB_BLOCK,
        ECF_ASTC_12x12_UNORM_BLOCK,
        ECF_ASTC_12x12_SRGB_BLOCK,

        //! Planar formats
        ECF_G8_B8_R8_3PLANE_420_UNORM,
        ECF_G8_B8R8_2PLANE_420_UNORM,
        ECF_G8_B8_R8_3PLANE_422_UNORM,
        ECF_G8_B8R8_2PLANE_422_UNORM,
        ECF_G8_B8_R8_3PLANE_444_UNORM,

		//! Unknown color format:
		ECF_UNKNOWN
	};

    namespace impl
    {
        template<ECOLOR_FORMAT cf, ECOLOR_FORMAT cmp, ECOLOR_FORMAT... searchtab>
        struct is_any_of : is_any_of<cf, searchtab...> {};

        template<ECOLOR_FORMAT cf, ECOLOR_FORMAT cmp>
        struct is_any_of<cf, cmp> : std::false_type {}; //if last comparison is also false, than return false

        template<ECOLOR_FORMAT cf, ECOLOR_FORMAT... searchtab>
        struct is_any_of<cf, cf, searchtab...> : std::true_type {};
    }//namespace impl

    //! Utility functions
    inline uint32_t getTexelOrBlockSize(ECOLOR_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case ECF_A1R5G5B5: return 2;
        case ECF_R8G8B8_UINT: return 3;
        case ECF_R8G8B8A8_UINT: return 4;
        case ECF_B10G11R11_UFLOAT_PACK32: return 4;
        case ECF_R16_SFLOAT: return 2;
        case ECF_R16G16_SFLOAT: return 4;
        case ECF_R16G16B16A16_SFLOAT: return 8;
        case ECF_R32_SFLOAT: return 4;
        case ECF_R32G32_SFLOAT: return 8;
        case ECF_R32G32B32A32_SFLOAT: return 16;
        case ECF_R8_UINT: return 1;
        case ECF_R8G8_UINT: return 2;
        case ECF_BC1_RGB_UNORM_BLOCK:
        case ECF_BC1_RGBA_UNORM_BLOCK:
            return 8;
        case ECF_BC2_UNORM_BLOCK:
        case ECF_BC3_UNORM_BLOCK:
            return 16;
        case ECF_8BIT_PIX: return 1;
        case ECF_16BIT_PIX: return 2;
        case ECF_24BIT_PIX: return 3;
        case ECF_32BIT_PIX: return 4;
        case ECF_48BIT_PIX: return 6;
        case ECF_64BIT_PIX: return 8;
        case ECF_96BIT_PIX: return 12;
        case ECF_128BIT_PIX: return 16;
        case ECF_DEPTH16: return 2;
        case ECF_DEPTH24: return 3;
        case ECF_DEPTH32F:
        case ECF_DEPTH24_STENCIL8: return 4;
        case ECF_DEPTH32F_STENCIL8: return 5;
        case ECF_STENCIL8: return 2;
        case ECF_E5B9G9R9_UFLOAT_PACK32: return 4;
        case ECF_R4G4_UNORM_PACK8: return 1;
        case ECF_R4G4B4A4_UNORM_PACK16: return 2;
        case ECF_B4G4R4A4_UNORM_PACK16: return 2;
        case ECF_R5G6B5_UNORM_PACK16: return 2;
        case ECF_B5G6R5_UNORM_PACK16: return 2;
        case ECF_R5G5B5A1_UNORM_PACK16: return 2;
        case ECF_B5G5R5A1_UNORM_PACK16: return 2;
        case ECF_A1R5G5B5_UNORM_PACK16: return 2;
        case ECF_R8_UNORM: return 1;
        case ECF_R8_SNORM: return 1;
        case ECF_R8_USCALED: return 1;
        case ECF_R8_SSCALED: return 1;
        case ECF_R8_SINT: return 1;
        case ECF_R8_SRGB: return 1;
        case ECF_R8G8_UNORM: return 2;
        case ECF_R8G8_SNORM: return 2;
        case ECF_R8G8_USCALED: return 2;
        case ECF_R8G8_SSCALED: return 2;
        case ECF_R8G8_SINT: return 2;
        case ECF_R8G8_SRGB: return 2;
        case ECF_R8G8B8_UNORM: return 3;
        case ECF_R8G8B8_SNORM: return 3;
        case ECF_R8G8B8_USCALED: return 3;
        case ECF_R8G8B8_SSCALED: return 3;
        case ECF_R8G8B8_SINT: return 3;
        case ECF_R8G8B8_SRGB: return 3;
        case ECF_B8G8R8_UNORM: return 3;
        case ECF_B8G8R8_SNORM: return 3;
        case ECF_B8G8R8_USCALED: return 3;
        case ECF_B8G8R8_SSCALED: return 3;
        case ECF_B8G8R8_UINT: return 3;
        case ECF_B8G8R8_SINT: return 3;
        case ECF_B8G8R8_SRGB: return 3;
        case ECF_R8G8B8A8_UNORM: return 4;
        case ECF_R8G8B8A8_SNORM: return 4;
        case ECF_R8G8B8A8_USCALED: return 4;
        case ECF_R8G8B8A8_SSCALED: return 4;
        case ECF_R8G8B8A8_SINT: return 4;
        case ECF_R8G8B8A8_SRGB: return 4;
        case ECF_B8G8R8A8_UNORM: return 4;
        case ECF_B8G8R8A8_SNORM: return 4;
        case ECF_B8G8R8A8_USCALED: return 4;
        case ECF_B8G8R8A8_SSCALED: return 4;
        case ECF_B8G8R8A8_UINT: return 4;
        case ECF_B8G8R8A8_SINT: return 4;
        case ECF_B8G8R8A8_SRGB: return 4;
        case ECF_A8B8G8R8_UNORM_PACK32: return 4;
        case ECF_A8B8G8R8_SNORM_PACK32: return 4;
        case ECF_A8B8G8R8_USCALED_PACK32: return 4;
        case ECF_A8B8G8R8_SSCALED_PACK32: return 4;
        case ECF_A8B8G8R8_UINT_PACK32: return 4;
        case ECF_A8B8G8R8_SINT_PACK32: return 4;
        case ECF_A8B8G8R8_SRGB_PACK32: return 4;
        case ECF_A2R10G10B10_UNORM_PACK32: return 4;
        case ECF_A2R10G10B10_SNORM_PACK32: return 4;
        case ECF_A2R10G10B10_USCALED_PACK32: return 4;
        case ECF_A2R10G10B10_SSCALED_PACK32: return 4;
        case ECF_A2R10G10B10_UINT_PACK32: return 4;
        case ECF_A2R10G10B10_SINT_PACK32: return 4;
        case ECF_A2B10G10R10_UNORM_PACK32: return 4;
        case ECF_A2B10G10R10_SNORM_PACK32: return 4;
        case ECF_A2B10G10R10_USCALED_PACK32: return 4;
        case ECF_A2B10G10R10_SSCALED_PACK32: return 4;
        case ECF_A2B10G10R10_UINT_PACK32: return 4;
        case ECF_A2B10G10R10_SINT_PACK32: return 4;
        case ECF_R16_UNORM: return 2;
        case ECF_R16_SNORM: return 2;
        case ECF_R16_USCALED: return 2;
        case ECF_R16_SSCALED: return 2;
        case ECF_R16_UINT: return 2;
        case ECF_R16_SINT: return 2;
        case ECF_R16G16_UNORM: return 4;
        case ECF_R16G16_SNORM: return 4;
        case ECF_R16G16_USCALED: return 4;
        case ECF_R16G16_SSCALED: return 4;
        case ECF_R16G16_UINT: return 4;
        case ECF_R16G16_SINT: return 4;
        case ECF_R16G16B16_UNORM: return 6;
        case ECF_R16G16B16_SNORM: return 6;
        case ECF_R16G16B16_USCALED: return 6;
        case ECF_R16G16B16_SSCALED: return 6;
        case ECF_R16G16B16_UINT: return 6;
        case ECF_R16G16B16_SINT: return 6;
        case ECF_R16G16B16A16_UNORM: return 8;
        case ECF_R16G16B16A16_SNORM: return 8;
        case ECF_R16G16B16A16_USCALED: return 8;
        case ECF_R16G16B16A16_SSCALED: return 8;
        case ECF_R16G16B16A16_UINT: return 8;
        case ECF_R16G16B16A16_SINT: return 8;
        case ECF_R32_UINT: return 4;
        case ECF_R32_SINT: return 4;
        case ECF_R32G32_UINT: return 8;
        case ECF_R32G32_SINT: return 8;
        case ECF_R32G32B32_UINT: return 12;
        case ECF_R32G32B32_SINT: return 12;
        case ECF_R32G32B32A32_UINT: return 16;
        case ECF_R32G32B32A32_SINT: return 16;
        case ECF_R64_UINT: return 8;
        case ECF_R64_SINT: return 8;
        case ECF_R64G64_UINT: return 16;
        case ECF_R64G64_SINT: return 16;
        case ECF_R64G64B64_UINT: return 24;
        case ECF_R64G64B64_SINT: return 24;
        case ECF_R64G64B64A64_UINT: return 32;
        case ECF_R64G64B64A64_SINT: return 32;
        case ECF_R16G16B16_SFLOAT: return 6;
        case ECF_R32G32B32_SFLOAT: return 12;
        case ECF_R64_SFLOAT: return 8;
        case ECF_R64G64_SFLOAT: return 16;
        case ECF_R64G64B64_SFLOAT: return 24;
        case ECF_R64G64B64A64_SFLOAT: return 32;

        case ECF_ASTC_4x4_UNORM_BLOCK:
        case ECF_ASTC_4x4_SRGB_BLOCK:
        case ECF_ASTC_5x4_UNORM_BLOCK:
        case ECF_ASTC_5x4_SRGB_BLOCK:
        case ECF_ASTC_5x5_UNORM_BLOCK:
        case ECF_ASTC_5x5_SRGB_BLOCK:
        case ECF_ASTC_6x5_UNORM_BLOCK:
        case ECF_ASTC_6x5_SRGB_BLOCK:
        case ECF_ASTC_6x6_UNORM_BLOCK:
        case ECF_ASTC_6x6_SRGB_BLOCK:
        case ECF_ASTC_8x5_UNORM_BLOCK:
        case ECF_ASTC_8x5_SRGB_BLOCK:
        case ECF_ASTC_8x6_UNORM_BLOCK:
        case ECF_ASTC_8x6_SRGB_BLOCK:
        case ECF_ASTC_8x8_UNORM_BLOCK:
        case ECF_ASTC_8x8_SRGB_BLOCK:
        case ECF_ASTC_10x5_UNORM_BLOCK:
        case ECF_ASTC_10x5_SRGB_BLOCK:
        case ECF_ASTC_10x6_UNORM_BLOCK:
        case ECF_ASTC_10x6_SRGB_BLOCK:
        case ECF_ASTC_10x8_UNORM_BLOCK:
        case ECF_ASTC_10x8_SRGB_BLOCK:
        case ECF_ASTC_10x10_UNORM_BLOCK:
        case ECF_ASTC_10x10_SRGB_BLOCK:
        case ECF_ASTC_12x10_UNORM_BLOCK:
        case ECF_ASTC_12x10_SRGB_BLOCK:
        case ECF_ASTC_12x12_UNORM_BLOCK:
        case ECF_ASTC_12x12_SRGB_BLOCK:
            return 16;
        default: return 0;
        }
    }

    inline core::vector3d<uint32_t> getBlockDimensions(ECOLOR_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case ECF_BC1_RGB_UNORM_BLOCK:
        case ECF_BC1_RGB_SRGB_BLOCK:
        case ECF_BC1_RGBA_UNORM_BLOCK:
        case ECF_BC1_RGBA_SRGB_BLOCK:
        case ECF_BC2_UNORM_BLOCK:
        case ECF_BC2_SRGB_BLOCK:
        case ECF_BC3_UNORM_BLOCK:
        case ECF_BC3_SRGB_BLOCK:
        case ECF_ASTC_4x4_UNORM_BLOCK:
        case ECF_ASTC_4x4_SRGB_BLOCK:
            return core::vector3d<uint32_t>(4u, 4u, 1u);
        case ECF_ASTC_5x4_UNORM_BLOCK:
        case ECF_ASTC_5x4_SRGB_BLOCK:
            return core::vector3d<uint32_t>(5u, 4u, 1u);
        case ECF_ASTC_5x5_UNORM_BLOCK:
        case ECF_ASTC_5x5_SRGB_BLOCK:
            return core::vector3d<uint32_t>(5u, 5u, 1u);
        case ECF_ASTC_6x5_UNORM_BLOCK:
        case ECF_ASTC_6x5_SRGB_BLOCK:
            return core::vector3d<uint32_t>(6u, 5u, 1u);
        case ECF_ASTC_6x6_UNORM_BLOCK:
        case ECF_ASTC_6x6_SRGB_BLOCK:
            return core::vector3d<uint32_t>(6u, 6u, 1u);
        case ECF_ASTC_8x5_UNORM_BLOCK:
        case ECF_ASTC_8x5_SRGB_BLOCK:
            return core::vector3d<uint32_t>(8u, 5u, 1u);
        case ECF_ASTC_8x6_UNORM_BLOCK:
        case ECF_ASTC_8x6_SRGB_BLOCK:
            return core::vector3d<uint32_t>(8u, 6u, 1u);
        case ECF_ASTC_8x8_UNORM_BLOCK:
        case ECF_ASTC_8x8_SRGB_BLOCK:
            return core::vector3d<uint32_t>(8u, 8u, 1u);
        case ECF_ASTC_10x5_UNORM_BLOCK:
        case ECF_ASTC_10x5_SRGB_BLOCK:
            return core::vector3d<uint32_t>(10u, 5u, 1u);
        case ECF_ASTC_10x6_UNORM_BLOCK:
        case ECF_ASTC_10x6_SRGB_BLOCK:
            return core::vector3d<uint32_t>(10u, 6u, 1u);
        case ECF_ASTC_10x8_UNORM_BLOCK:
        case ECF_ASTC_10x8_SRGB_BLOCK:
            return core::vector3d<uint32_t>(10u, 8u, 1u);
        case ECF_ASTC_10x10_UNORM_BLOCK:
        case ECF_ASTC_10x10_SRGB_BLOCK:
            return core::vector3d<uint32_t>(10u, 10u, 1u);
        case ECF_ASTC_12x10_UNORM_BLOCK:
        case ECF_ASTC_12x10_SRGB_BLOCK:
            return core::vector3d<uint32_t>(12u, 10u, 1u);
        case ECF_ASTC_12x12_UNORM_BLOCK:
        case ECF_ASTC_12x12_SRGB_BLOCK:
            return core::vector3d<uint32_t>(12u, 12u, 1u);
        default:
            return core::vector3d<uint32_t>(1u);
        }
    }

    template<ECOLOR_FORMAT cf>
    constexpr bool isSignedFormat()
    {
        return impl::is_any_of <
            cf,
            ECF_R8_SNORM,
            ECF_R8_SSCALED,
            ECF_R8_SINT,
            ECF_R8G8_SNORM,
            ECF_R8G8_SSCALED,
            ECF_R8G8_SINT,
            ECF_R8G8B8_SNORM,
            ECF_R8G8B8_SSCALED,
            ECF_R8G8B8_SINT,
            ECF_B8G8R8_SNORM,
            ECF_B8G8R8_SSCALED,
            ECF_B8G8R8_SINT,
            ECF_R8G8B8A8_SNORM,
            ECF_R8G8B8A8_SSCALED,
            ECF_R8G8B8A8_SINT,
            ECF_B8G8R8A8_SNORM,
            ECF_B8G8R8A8_SSCALED,
            ECF_B8G8R8A8_SINT,
            ECF_A8B8G8R8_SNORM_PACK32,
            ECF_A8B8G8R8_SSCALED_PACK32,
            ECF_A8B8G8R8_SINT_PACK32,
            ECF_A2R10G10B10_SNORM_PACK32,
            ECF_A2R10G10B10_SSCALED_PACK32,
            ECF_A2R10G10B10_SINT_PACK32,
            ECF_A2B10G10R10_SNORM_PACK32,
            ECF_A2B10G10R10_SSCALED_PACK32,
            ECF_A2B10G10R10_SINT_PACK32,
            ECF_R16_SNORM,
            ECF_R16_SSCALED,
            ECF_R16_SINT,
            ECF_R16G16_SNORM,
            ECF_R16G16_SSCALED,
            ECF_R16G16_SINT,
            ECF_R16G16B16_SNORM,
            ECF_R16G16B16_SSCALED,
            ECF_R16G16B16_SINT,
            ECF_R16G16B16A16_SNORM,
            ECF_R16G16B16A16_SSCALED,
            ECF_R16G16B16A16_SINT,
            ECF_R32_SINT,
            ECF_R32G32_SINT,
            ECF_R32G32B32_SINT,
            ECF_R32G32B32A32_SINT,
            ECF_R64_SINT,
            ECF_R64G64_SINT,
            ECF_R64G64B64_SINT,
            ECF_R64G64B64A64_SINT,
            ECF_R16G16B16_SFLOAT,
            ECF_R32G32B32_SFLOAT,
            ECF_R64_SFLOAT,
            ECF_R64G64_SFLOAT,
            ECF_R64G64B64_SFLOAT,
            ECF_R64G64B64A64_SFLOAT
        > ::value;
    }
    template<ECOLOR_FORMAT cf>
    constexpr bool isIntegerFormat()
    {
        return impl::is_any_of <
            cf,
            ECF_R8_SINT,
            ECF_R8_SRGB,
            ECF_R8G8_SINT,
            ECF_R8G8_SRGB,
            ECF_R8G8B8_SINT,
            ECF_R8G8B8_SRGB,
            ECF_B8G8R8_UINT,
            ECF_B8G8R8_SINT,
            ECF_B8G8R8_SRGB,
            ECF_R8G8B8A8_SINT,
            ECF_R8G8B8A8_SRGB,
            ECF_B8G8R8A8_UINT,
            ECF_B8G8R8A8_SINT,
            ECF_B8G8R8A8_SRGB,
            ECF_A8B8G8R8_UINT_PACK32,
            ECF_A8B8G8R8_SINT_PACK32,
            ECF_A8B8G8R8_SRGB_PACK32,
            ECF_A2R10G10B10_UINT_PACK32,
            ECF_A2R10G10B10_SINT_PACK32,
            ECF_A2B10G10R10_UINT_PACK32,
            ECF_A2B10G10R10_SINT_PACK32,
            ECF_R16_UINT,
            ECF_R16_SINT,
            ECF_R16G16_UINT,
            ECF_R16G16_SINT,
            ECF_R16G16B16_UINT,
            ECF_R16G16B16_SINT,
            ECF_R16G16B16A16_UINT,
            ECF_R16G16B16A16_SINT,
            ECF_R32_UINT,
            ECF_R32_SINT,
            ECF_R32G32_UINT,
            ECF_R32G32_SINT,
            ECF_R32G32B32_UINT,
            ECF_R32G32B32_SINT,
            ECF_R32G32B32A32_UINT,
            ECF_R32G32B32A32_SINT,
            ECF_R64_UINT,
            ECF_R64_SINT,
            ECF_R64G64_UINT,
            ECF_R64G64_SINT,
            ECF_R64G64B64_UINT,
            ECF_R64G64B64_SINT,
            ECF_R64G64B64A64_UINT,
            ECF_R64G64B64A64_SINT
        > ::value;
    }
    template<ECOLOR_FORMAT cf>
    constexpr bool isFloatingPointFormat()
    {
        return impl::is_any_of<
            cf,
            ECF_R16_SFLOAT,
            ECF_R16G16_SFLOAT,
            ECF_R16G16B16_SFLOAT,
            ECF_R16G16B16A16_SFLOAT,
            ECF_R32_SFLOAT,
            ECF_R32G32_SFLOAT,
            ECF_R32G32B32_SFLOAT,
            ECF_R32G32B32A32_SFLOAT,
            ECF_R64_SFLOAT,
            ECF_R64G64_SFLOAT,
            ECF_R64G64B64_SFLOAT,
            ECF_R64G64B64A64_SFLOAT,
            ECF_B10G11R11_UFLOAT_PACK32,
            ECF_E5B9G9R9_UFLOAT_PACK32,
            ECF_E5B9G9R9_UFLOAT_PACK32
        >::value;
    }
    template<ECOLOR_FORMAT cf>
    constexpr bool isNormalizedFormat()
    {
        return impl::is_any_of <
            cf,
            ECF_R4G4_UNORM_PACK8,
            ECF_R4G4B4A4_UNORM_PACK16,
            ECF_B4G4R4A4_UNORM_PACK16,
            ECF_R5G6B5_UNORM_PACK16,
            ECF_B5G6R5_UNORM_PACK16,
            ECF_R5G5B5A1_UNORM_PACK16,
            ECF_B5G5R5A1_UNORM_PACK16,
            ECF_A1R5G5B5_UNORM_PACK16,
            ECF_R8_UNORM,
            ECF_R8_SNORM,
            ECF_R8_USCALED,
            ECF_R8_SSCALED,
            ECF_R8G8_UNORM,
            ECF_R8G8_SNORM,
            ECF_R8G8_USCALED,
            ECF_R8G8_SSCALED,
            ECF_R8G8B8_UNORM,
            ECF_R8G8B8_SNORM,
            ECF_R8G8B8_USCALED,
            ECF_R8G8B8_SSCALED,
            ECF_B8G8R8_UNORM,
            ECF_B8G8R8_SNORM,
            ECF_B8G8R8_USCALED,
            ECF_B8G8R8_SSCALED,
            ECF_R8G8B8A8_UNORM,
            ECF_R8G8B8A8_SNORM,
            ECF_R8G8B8A8_USCALED,
            ECF_R8G8B8A8_SSCALED,
            ECF_B8G8R8A8_UNORM,
            ECF_B8G8R8A8_SNORM,
            ECF_B8G8R8A8_USCALED,
            ECF_B8G8R8A8_SSCALED,
            ECF_A8B8G8R8_UNORM_PACK32,
            ECF_A8B8G8R8_SNORM_PACK32,
            ECF_A8B8G8R8_USCALED_PACK32,
            ECF_A8B8G8R8_SSCALED_PACK32,
            ECF_A2R10G10B10_UNORM_PACK32,
            ECF_A2R10G10B10_SNORM_PACK32,
            ECF_A2R10G10B10_USCALED_PACK32,
            ECF_A2R10G10B10_SSCALED_PACK32,
            ECF_A2B10G10R10_UNORM_PACK32,
            ECF_A2B10G10R10_SNORM_PACK32,
            ECF_A2B10G10R10_USCALED_PACK32,
            ECF_A2B10G10R10_SSCALED_PACK32,
            ECF_R16_UNORM,
            ECF_R16_SNORM,
            ECF_R16_USCALED,
            ECF_R16_SSCALED,
            ECF_R16G16_UNORM,
            ECF_R16G16_SNORM,
            ECF_R16G16_USCALED,
            ECF_R16G16_SSCALED,
            ECF_R16G16B16_UNORM,
            ECF_R16G16B16_SNORM,
            ECF_R16G16B16_USCALED,
            ECF_R16G16B16_SSCALED,
            ECF_R16G16B16A16_UNORM,
            ECF_R16G16B16A16_SNORM,
            ECF_R16G16B16A16_USCALED,
            ECF_BC1_RGB_UNORM_BLOCK,
            ECF_BC1_RGB_SRGB_BLOCK,
            ECF_BC1_RGBA_UNORM_BLOCK,
            ECF_BC1_RGBA_SRGB_BLOCK,
            ECF_BC2_UNORM_BLOCK,
            ECF_BC2_SRGB_BLOCK,
            ECF_BC3_UNORM_BLOCK,
            ECF_BC3_SRGB_BLOCK,
            ECF_BC3_SRGB_BLOCK,
            ECF_ASTC_4x4_UNORM_BLOCK,
            ECF_ASTC_4x4_SRGB_BLOCK,
            ECF_ASTC_5x4_UNORM_BLOCK,
            ECF_ASTC_5x4_SRGB_BLOCK,
            ECF_ASTC_5x5_UNORM_BLOCK,
            ECF_ASTC_5x5_SRGB_BLOCK,
            ECF_ASTC_6x5_UNORM_BLOCK,
            ECF_ASTC_6x5_SRGB_BLOCK,
            ECF_ASTC_6x6_UNORM_BLOCK,
            ECF_ASTC_6x6_SRGB_BLOCK,
            ECF_ASTC_8x5_UNORM_BLOCK,
            ECF_ASTC_8x5_SRGB_BLOCK,
            ECF_ASTC_8x6_UNORM_BLOCK,
            ECF_ASTC_8x6_SRGB_BLOCK,
            ECF_ASTC_8x8_UNORM_BLOCK,
            ECF_ASTC_8x8_SRGB_BLOCK,
            ECF_ASTC_10x5_UNORM_BLOCK,
            ECF_ASTC_10x5_SRGB_BLOCK,
            ECF_ASTC_10x6_UNORM_BLOCK,
            ECF_ASTC_10x6_SRGB_BLOCK,
            ECF_ASTC_10x8_UNORM_BLOCK,
            ECF_ASTC_10x8_SRGB_BLOCK,
            ECF_ASTC_10x10_UNORM_BLOCK,
            ECF_ASTC_10x10_SRGB_BLOCK,
            ECF_ASTC_12x10_UNORM_BLOCK,
            ECF_ASTC_12x10_SRGB_BLOCK,
            ECF_ASTC_12x12_UNORM_BLOCK,
            ECF_ASTC_12x12_SRGB_BLOCK
        > ::value;
    }
    template<ECOLOR_FORMAT cf>
    constexpr bool isScaledFormat()
    {
        return impl::is_any_of <
            cf,
            ECF_R8_USCALED,
            ECF_R8_SSCALED,
            ECF_R8G8_USCALED,
            ECF_R8G8_SSCALED,
            ECF_R8G8B8_USCALED,
            ECF_R8G8B8_SSCALED,
            ECF_B8G8R8_USCALED,
            ECF_B8G8R8_SSCALED,
            ECF_R8G8B8A8_USCALED,
            ECF_R8G8B8A8_SSCALED,
            ECF_B8G8R8A8_USCALED,
            ECF_B8G8R8A8_SSCALED,
            ECF_A8B8G8R8_USCALED_PACK32,
            ECF_A8B8G8R8_SSCALED_PACK32,
            ECF_A2R10G10B10_USCALED_PACK32,
            ECF_A2R10G10B10_SSCALED_PACK32,
            ECF_A2B10G10R10_USCALED_PACK32,
            ECF_A2B10G10R10_SSCALED_PACK32,
            ECF_R16_USCALED,
            ECF_R16_SSCALED,
            ECF_R16G16_USCALED,
            ECF_R16G16_SSCALED,
            ECF_R16G16B16_USCALED,
            ECF_R16G16B16_SSCALED,
            ECF_R16G16B16A16_USCALED,
            ECF_R16G16B16A16_SSCALED,
            ECF_R16G16B16A16_SSCALED
        > ::value;
    }
    template<ECOLOR_FORMAT cf>
    constexpr bool isSRGBFormat()
    {
        return impl::is_any_of<
            cf,
            ECF_R8_SRGB,
            ECF_R8G8_SRGB,
            ECF_R8G8B8_SRGB,
            ECF_B8G8R8_SRGB,
            ECF_R8G8B8A8_SRGB,
            ECF_B8G8R8A8_SRGB,
            ECF_A8B8G8R8_SRGB_PACK32,
            ECF_ASTC_4x4_SRGB_BLOCK,
            ECF_ASTC_5x4_SRGB_BLOCK,
            ECF_ASTC_5x5_SRGB_BLOCK,
            ECF_ASTC_6x5_SRGB_BLOCK,
            ECF_ASTC_6x6_SRGB_BLOCK,
            ECF_ASTC_8x5_SRGB_BLOCK,
            ECF_ASTC_8x6_SRGB_BLOCK,
            ECF_ASTC_8x8_SRGB_BLOCK,
            ECF_ASTC_10x5_SRGB_BLOCK,
            ECF_ASTC_10x6_SRGB_BLOCK,
            ECF_ASTC_10x8_SRGB_BLOCK,
            ECF_ASTC_10x10_SRGB_BLOCK,
            ECF_ASTC_12x10_SRGB_BLOCK,
            ECF_ASTC_12x12_SRGB_BLOCK,
            ECF_ASTC_12x12_SRGB_BLOCK
        >::value;
    }
    template<ECOLOR_FORMAT cf>
    constexpr bool isBlockCompressionFormat()
    {
        return impl::is_any_of<
            cf,
            ECF_BC1_RGB_UNORM_BLOCK,
            ECF_BC1_RGB_SRGB_BLOCK,
            ECF_BC1_RGBA_UNORM_BLOCK,
            ECF_BC1_RGBA_SRGB_BLOCK,
            ECF_BC2_UNORM_BLOCK,
            ECF_BC2_SRGB_BLOCK,
            ECF_BC3_UNORM_BLOCK,
            ECF_BC3_SRGB_BLOCK,
            ECF_ASTC_4x4_UNORM_BLOCK,
            ECF_ASTC_4x4_SRGB_BLOCK,
            ECF_ASTC_5x4_UNORM_BLOCK,
            ECF_ASTC_5x4_SRGB_BLOCK,
            ECF_ASTC_5x5_UNORM_BLOCK,
            ECF_ASTC_5x5_SRGB_BLOCK,
            ECF_ASTC_6x5_UNORM_BLOCK,
            ECF_ASTC_6x5_SRGB_BLOCK,
            ECF_ASTC_6x6_UNORM_BLOCK,
            ECF_ASTC_6x6_SRGB_BLOCK,
            ECF_ASTC_8x5_UNORM_BLOCK,
            ECF_ASTC_8x5_SRGB_BLOCK,
            ECF_ASTC_8x6_UNORM_BLOCK,
            ECF_ASTC_8x6_SRGB_BLOCK,
            ECF_ASTC_8x8_UNORM_BLOCK,
            ECF_ASTC_8x8_SRGB_BLOCK,
            ECF_ASTC_10x5_UNORM_BLOCK,
            ECF_ASTC_10x5_SRGB_BLOCK,
            ECF_ASTC_10x6_UNORM_BLOCK,
            ECF_ASTC_10x6_SRGB_BLOCK,
            ECF_ASTC_10x8_UNORM_BLOCK,
            ECF_ASTC_10x8_SRGB_BLOCK,
            ECF_ASTC_10x10_UNORM_BLOCK,
            ECF_ASTC_10x10_SRGB_BLOCK,
            ECF_ASTC_12x10_UNORM_BLOCK,
            ECF_ASTC_12x10_SRGB_BLOCK,
            ECF_ASTC_12x12_UNORM_BLOCK,
            ECF_ASTC_12x12_SRGB_BLOCK
        >::value;
    }
    template<ECOLOR_FORMAT cf>
    constexpr bool isPlanarFormat()
    {
        return impl::is_any_of<
            cf,
            ECF_G8_B8_R8_3PLANE_420_UNORM,
            ECF_G8_B8R8_2PLANE_420_UNORM,
            ECF_G8_B8_R8_3PLANE_422_UNORM,
            ECF_G8_B8R8_2PLANE_422_UNORM,
            ECF_G8_B8_R8_3PLANE_444_UNORM,
            ECF_G8_B8_R8_3PLANE_444_UNORM
        >::value;
    }

    inline void getHorizontalReductionFactorPerPlane(ECOLOR_FORMAT _planarFmt, uint32_t _reductionFactor[4])
    {
        switch (_planarFmt)
        {
        case ECF_G8_B8_R8_3PLANE_420_UNORM:
        case ECF_G8_B8_R8_3PLANE_422_UNORM:
            _reductionFactor[0] = 1u;
            _reductionFactor[1] = _reductionFactor[2] = 2u;
            return;
        case ECF_G8_B8R8_2PLANE_420_UNORM:
        case ECF_G8_B8R8_2PLANE_422_UNORM:
            _reductionFactor[0] = 1u;
            _reductionFactor[1] = 2u;
            return;
        case ECF_G8_B8_R8_3PLANE_444_UNORM:
            _reductionFactor[0] = _reductionFactor[1] = _reductionFactor[2] = 1u;
            return;
        }
    }
    inline void getVerticalReductionFactorPerPlane(ECOLOR_FORMAT _planarFmt, uint32_t _reductionFactor[4])
    {
        switch (_planarFmt)
        {
        case ECF_G8_B8_R8_3PLANE_420_UNORM:
            _reductionFactor[0] = 1u;
            _reductionFactor[1] = _reductionFactor[2] = 2u;
            return;
        case ECF_G8_B8R8_2PLANE_420_UNORM:
            _reductionFactor[0] = 1u;
            _reductionFactor[1] = 2u;
            return;
        case ECF_G8_B8_R8_3PLANE_422_UNORM:
        case ECF_G8_B8R8_2PLANE_422_UNORM:
        case ECF_G8_B8_R8_3PLANE_444_UNORM:
            _reductionFactor[0] = _reductionFactor[1] = _reductionFactor[2] = 1u;
            return;
        }
    }
    inline void getChannelsPerPlane(ECOLOR_FORMAT _planarFmt, uint32_t _chCnt[4])
    {
        switch (_planarFmt)
        {
        case ECF_G8_B8_R8_3PLANE_420_UNORM:
        case ECF_G8_B8_R8_3PLANE_422_UNORM:
        case ECF_G8_B8_R8_3PLANE_444_UNORM:
            _chCnt[0] = _chCnt[1] = _chCnt[2] = 1u;
            return;
        case ECF_G8_B8R8_2PLANE_420_UNORM:
        case ECF_G8_B8R8_2PLANE_422_UNORM:
            _chCnt[0] = 1u;
            _chCnt[1] = 2u;
            return;
        }
    }

    inline bool isSignedFormat(ECOLOR_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case ECF_R8_SNORM:
        case ECF_R8_SSCALED:
        case ECF_R8_SINT:
        case ECF_R8G8_SNORM:
        case ECF_R8G8_SSCALED:
        case ECF_R8G8_SINT:
        case ECF_R8G8B8_SNORM:
        case ECF_R8G8B8_SSCALED:
        case ECF_R8G8B8_SINT:
        case ECF_B8G8R8_SNORM:
        case ECF_B8G8R8_SSCALED:
        case ECF_B8G8R8_SINT:
        case ECF_R8G8B8A8_SNORM:
        case ECF_R8G8B8A8_SSCALED:
        case ECF_R8G8B8A8_SINT:
        case ECF_B8G8R8A8_SNORM:
        case ECF_B8G8R8A8_SSCALED:
        case ECF_B8G8R8A8_SINT:
        case ECF_A8B8G8R8_SNORM_PACK32:
        case ECF_A8B8G8R8_SSCALED_PACK32:
        case ECF_A8B8G8R8_SINT_PACK32:
        case ECF_A2R10G10B10_SNORM_PACK32:
        case ECF_A2R10G10B10_SSCALED_PACK32:
        case ECF_A2R10G10B10_SINT_PACK32:
        case ECF_A2B10G10R10_SNORM_PACK32:
        case ECF_A2B10G10R10_SSCALED_PACK32:
        case ECF_A2B10G10R10_SINT_PACK32:
        case ECF_R16_SNORM:
        case ECF_R16_SSCALED:
        case ECF_R16_SINT:
        case ECF_R16G16_SNORM:
        case ECF_R16G16_SSCALED:
        case ECF_R16G16_SINT:
        case ECF_R16G16B16_SNORM:
        case ECF_R16G16B16_SSCALED:
        case ECF_R16G16B16_SINT:
        case ECF_R16G16B16A16_SNORM:
        case ECF_R16G16B16A16_SSCALED:
        case ECF_R16G16B16A16_SINT:
        case ECF_R32_SINT:
        case ECF_R32G32_SINT:
        case ECF_R32G32B32_SINT:
        case ECF_R32G32B32A32_SINT:
        case ECF_R64_SINT:
        case ECF_R64G64_SINT:
        case ECF_R64G64B64_SINT:
        case ECF_R64G64B64A64_SINT:
        case ECF_R16G16B16_SFLOAT:
        case ECF_R32G32B32_SFLOAT:
        case ECF_R64_SFLOAT:
        case ECF_R64G64_SFLOAT:
        case ECF_R64G64B64_SFLOAT:
        case ECF_R64G64B64A64_SFLOAT:
            return true;
        default: return false;
        }
    }
    inline bool isIntegerFormat(ECOLOR_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case ECF_R8_SINT:
        case ECF_R8_SRGB:
        case ECF_R8G8_SINT:
        case ECF_R8G8_SRGB:
        case ECF_R8G8B8_SINT:
        case ECF_R8G8B8_SRGB:
        case ECF_B8G8R8_UINT:
        case ECF_B8G8R8_SINT:
        case ECF_B8G8R8_SRGB:
        case ECF_R8G8B8A8_SINT:
        case ECF_R8G8B8A8_SRGB:
        case ECF_B8G8R8A8_UINT:
        case ECF_B8G8R8A8_SINT:
        case ECF_B8G8R8A8_SRGB:
        case ECF_A8B8G8R8_UINT_PACK32:
        case ECF_A8B8G8R8_SINT_PACK32:
        case ECF_A8B8G8R8_SRGB_PACK32:
        case ECF_A2R10G10B10_UINT_PACK32:
        case ECF_A2R10G10B10_SINT_PACK32:
        case ECF_A2B10G10R10_UINT_PACK32:
        case ECF_A2B10G10R10_SINT_PACK32:
        case ECF_R16_UINT:
        case ECF_R16_SINT:
        case ECF_R16G16_UINT:
        case ECF_R16G16_SINT:
        case ECF_R16G16B16_UINT:
        case ECF_R16G16B16_SINT:
        case ECF_R16G16B16A16_UINT:
        case ECF_R16G16B16A16_SINT:
        case ECF_R32_UINT:
        case ECF_R32_SINT:
        case ECF_R32G32_UINT:
        case ECF_R32G32_SINT:
        case ECF_R32G32B32_UINT:
        case ECF_R32G32B32_SINT:
        case ECF_R32G32B32A32_UINT:
        case ECF_R32G32B32A32_SINT:
        case ECF_R64_UINT:
        case ECF_R64_SINT:
        case ECF_R64G64_UINT:
        case ECF_R64G64_SINT:
        case ECF_R64G64B64_UINT:
        case ECF_R64G64B64_SINT:
        case ECF_R64G64B64A64_UINT:
        case ECF_R64G64B64A64_SINT:
            return true;
        default: return false;
        }
    }
    inline bool isFloatingPointFormat(ECOLOR_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case ECF_R16_SFLOAT:
        case ECF_R16G16_SFLOAT:
        case ECF_R16G16B16_SFLOAT:
        case ECF_R16G16B16A16_SFLOAT:
        case ECF_R32_SFLOAT:
        case ECF_R32G32_SFLOAT:
        case ECF_R32G32B32_SFLOAT:
        case ECF_R32G32B32A32_SFLOAT:
        case ECF_R64_SFLOAT:
        case ECF_R64G64_SFLOAT:
        case ECF_R64G64B64_SFLOAT:
        case ECF_R64G64B64A64_SFLOAT:
        case ECF_B10G11R11_UFLOAT_PACK32:
        case ECF_E5B9G9R9_UFLOAT_PACK32:
            return true;
        default: return false;
        }
    }
    //! Note that scaled formats are subset of normalized formats
    inline bool isNormalizedFormat(ECOLOR_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case ECF_R4G4_UNORM_PACK8:
        case ECF_R4G4B4A4_UNORM_PACK16:
        case ECF_B4G4R4A4_UNORM_PACK16:
        case ECF_R5G6B5_UNORM_PACK16:
        case ECF_B5G6R5_UNORM_PACK16:
        case ECF_R5G5B5A1_UNORM_PACK16:
        case ECF_B5G5R5A1_UNORM_PACK16:
        case ECF_A1R5G5B5_UNORM_PACK16:
        case ECF_R8_UNORM:
        case ECF_R8_SNORM:
        case ECF_R8_USCALED:
        case ECF_R8_SSCALED:
        case ECF_R8G8_UNORM:
        case ECF_R8G8_SNORM:
        case ECF_R8G8_USCALED:
        case ECF_R8G8_SSCALED:
        case ECF_R8G8B8_UNORM:
        case ECF_R8G8B8_SNORM:
        case ECF_R8G8B8_USCALED:
        case ECF_R8G8B8_SSCALED:
        case ECF_B8G8R8_UNORM:
        case ECF_B8G8R8_SNORM:
        case ECF_B8G8R8_USCALED:
        case ECF_B8G8R8_SSCALED:
        case ECF_R8G8B8A8_UNORM:
        case ECF_R8G8B8A8_SNORM:
        case ECF_R8G8B8A8_USCALED:
        case ECF_R8G8B8A8_SSCALED:
        case ECF_B8G8R8A8_UNORM:
        case ECF_B8G8R8A8_SNORM:
        case ECF_B8G8R8A8_USCALED:
        case ECF_B8G8R8A8_SSCALED:
        case ECF_A8B8G8R8_UNORM_PACK32:
        case ECF_A8B8G8R8_SNORM_PACK32:
        case ECF_A8B8G8R8_USCALED_PACK32:
        case ECF_A8B8G8R8_SSCALED_PACK32:
        case ECF_A2R10G10B10_UNORM_PACK32:
        case ECF_A2R10G10B10_SNORM_PACK32:
        case ECF_A2R10G10B10_USCALED_PACK32:
        case ECF_A2R10G10B10_SSCALED_PACK32:
        case ECF_A2B10G10R10_UNORM_PACK32:
        case ECF_A2B10G10R10_SNORM_PACK32:
        case ECF_A2B10G10R10_USCALED_PACK32:
        case ECF_A2B10G10R10_SSCALED_PACK32:
        case ECF_R16_UNORM:
        case ECF_R16_SNORM:
        case ECF_R16_USCALED:
        case ECF_R16_SSCALED:
        case ECF_R16G16_UNORM:
        case ECF_R16G16_SNORM:
        case ECF_R16G16_USCALED:
        case ECF_R16G16_SSCALED:
        case ECF_R16G16B16_UNORM:
        case ECF_R16G16B16_SNORM:
        case ECF_R16G16B16_USCALED:
        case ECF_R16G16B16_SSCALED:
        case ECF_R16G16B16A16_UNORM:
        case ECF_R16G16B16A16_SNORM:
        case ECF_R16G16B16A16_USCALED:
        case ECF_R16G16B16A16_SSCALED:
        case ECF_BC1_RGB_UNORM_BLOCK:
        case ECF_BC1_RGB_SRGB_BLOCK:
        case ECF_BC1_RGBA_UNORM_BLOCK:
        case ECF_BC1_RGBA_SRGB_BLOCK:
        case ECF_BC2_UNORM_BLOCK:
        case ECF_BC2_SRGB_BLOCK:
        case ECF_BC3_UNORM_BLOCK:
        case ECF_BC3_SRGB_BLOCK:
        case ECF_ASTC_4x4_UNORM_BLOCK:
        case ECF_ASTC_4x4_SRGB_BLOCK:
        case ECF_ASTC_5x4_UNORM_BLOCK:
        case ECF_ASTC_5x4_SRGB_BLOCK:
        case ECF_ASTC_5x5_UNORM_BLOCK:
        case ECF_ASTC_5x5_SRGB_BLOCK:
        case ECF_ASTC_6x5_UNORM_BLOCK:
        case ECF_ASTC_6x5_SRGB_BLOCK:
        case ECF_ASTC_6x6_UNORM_BLOCK:
        case ECF_ASTC_6x6_SRGB_BLOCK:
        case ECF_ASTC_8x5_UNORM_BLOCK:
        case ECF_ASTC_8x5_SRGB_BLOCK:
        case ECF_ASTC_8x6_UNORM_BLOCK:
        case ECF_ASTC_8x6_SRGB_BLOCK:
        case ECF_ASTC_8x8_UNORM_BLOCK:
        case ECF_ASTC_8x8_SRGB_BLOCK:
        case ECF_ASTC_10x5_UNORM_BLOCK:
        case ECF_ASTC_10x5_SRGB_BLOCK:
        case ECF_ASTC_10x6_UNORM_BLOCK:
        case ECF_ASTC_10x6_SRGB_BLOCK:
        case ECF_ASTC_10x8_UNORM_BLOCK:
        case ECF_ASTC_10x8_SRGB_BLOCK:
        case ECF_ASTC_10x10_UNORM_BLOCK:
        case ECF_ASTC_10x10_SRGB_BLOCK:
        case ECF_ASTC_12x10_UNORM_BLOCK:
        case ECF_ASTC_12x10_SRGB_BLOCK:
        case ECF_ASTC_12x12_UNORM_BLOCK:
        case ECF_ASTC_12x12_SRGB_BLOCK:
            return true;
        default: return false;
        }
    }
    //SCALED implies NORMALIZED!
    inline bool isScaledFormat(ECOLOR_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case ECF_R8_USCALED:
        case ECF_R8_SSCALED:
        case ECF_R8G8_USCALED:
        case ECF_R8G8_SSCALED:
        case ECF_R8G8B8_USCALED:
        case ECF_R8G8B8_SSCALED:
        case ECF_B8G8R8_USCALED:
        case ECF_B8G8R8_SSCALED:
        case ECF_R8G8B8A8_USCALED:
        case ECF_R8G8B8A8_SSCALED:
        case ECF_B8G8R8A8_USCALED:
        case ECF_B8G8R8A8_SSCALED:
        case ECF_A8B8G8R8_USCALED_PACK32:
        case ECF_A8B8G8R8_SSCALED_PACK32:
        case ECF_A2R10G10B10_USCALED_PACK32:
        case ECF_A2R10G10B10_SSCALED_PACK32:
        case ECF_A2B10G10R10_USCALED_PACK32:
        case ECF_A2B10G10R10_SSCALED_PACK32:
        case ECF_R16_USCALED:
        case ECF_R16_SSCALED:
        case ECF_R16G16_USCALED:
        case ECF_R16G16_SSCALED:
        case ECF_R16G16B16_USCALED:
        case ECF_R16G16B16_SSCALED:
        case ECF_R16G16B16A16_USCALED:
        case ECF_R16G16B16A16_SSCALED:
            return true;
        default: return false;
        }
    }
    inline bool isSRGBFormat(ECOLOR_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case ECF_R8_SRGB:
        case ECF_R8G8_SRGB:
        case ECF_R8G8B8_SRGB:
        case ECF_B8G8R8_SRGB:
        case ECF_R8G8B8A8_SRGB:
        case ECF_B8G8R8A8_SRGB:
        case ECF_A8B8G8R8_SRGB_PACK32:
        case ECF_BC1_RGB_SRGB_BLOCK:
        case ECF_BC1_RGBA_SRGB_BLOCK:
        case ECF_BC2_SRGB_BLOCK:
        case ECF_BC3_SRGB_BLOCK:
        case ECF_ASTC_4x4_SRGB_BLOCK:
        case ECF_ASTC_5x4_SRGB_BLOCK:
        case ECF_ASTC_5x5_SRGB_BLOCK:
        case ECF_ASTC_6x5_SRGB_BLOCK:
        case ECF_ASTC_6x6_SRGB_BLOCK:
        case ECF_ASTC_8x5_SRGB_BLOCK:
        case ECF_ASTC_8x6_SRGB_BLOCK:
        case ECF_ASTC_8x8_SRGB_BLOCK:
        case ECF_ASTC_10x5_SRGB_BLOCK:
        case ECF_ASTC_10x6_SRGB_BLOCK:
        case ECF_ASTC_10x8_SRGB_BLOCK:
        case ECF_ASTC_10x10_SRGB_BLOCK:
        case ECF_ASTC_12x10_SRGB_BLOCK:
        case ECF_ASTC_12x12_SRGB_BLOCK:
            return true;
        default: return false;
        }
    }
    inline bool isBlockCompressionFormat(ECOLOR_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case ECF_BC1_RGB_UNORM_BLOCK:
        case ECF_BC1_RGB_SRGB_BLOCK:
        case ECF_BC1_RGBA_UNORM_BLOCK:
        case ECF_BC1_RGBA_SRGB_BLOCK:
        case ECF_BC2_UNORM_BLOCK:
        case ECF_BC2_SRGB_BLOCK:
        case ECF_BC3_UNORM_BLOCK:
        case ECF_BC3_SRGB_BLOCK:
            return true;
        default: return false;
        }
    }
    inline bool isPlanarFormat(ECOLOR_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case ECF_G8_B8_R8_3PLANE_420_UNORM:
        case ECF_G8_B8R8_2PLANE_420_UNORM:
        case ECF_G8_B8_R8_3PLANE_422_UNORM:
        case ECF_G8_B8R8_2PLANE_422_UNORM:
        case ECF_G8_B8_R8_3PLANE_444_UNORM:
            return true;
        default: return false;
        }
    }
	
}} //irr::video

#endif //__IRR_E_COLOR_H_INCLUDED__