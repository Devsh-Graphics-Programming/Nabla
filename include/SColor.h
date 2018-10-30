// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __COLOR_H_INCLUDED__
#define __COLOR_H_INCLUDED__

#include <type_traits>
#include "vectorSIMD.h"
#include "coreutil.h"

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
        ECF_R8_UINT,//new
        ECF_R8_SINT,
        ECF_R8_SRGB,
        ECF_R8G8_UNORM,
        ECF_R8G8_SNORM,
        ECF_R8G8_USCALED,
        ECF_R8G8_SSCALED,
        ECF_R8G8_UINT,//new
        ECF_R8G8_SINT,
        ECF_R8G8_SRGB,
        ECF_R8G8B8_UNORM,
        ECF_R8G8B8_SNORM,
        ECF_R8G8B8_USCALED,
        ECF_R8G8B8_SSCALED,
        ECF_R8G8B8_UINT,//new
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
        ECF_R8G8B8A8_UINT,//new
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
        ECF_R16G16B16A16_SFLOAT,//new
        ECF_R32_UINT,
        ECF_R32_SINT,
        ECF_R32_SFLOAT,//new
        ECF_R32G32_UINT,
        ECF_R32G32_SINT,
        ECF_R32G32_SFLOAT,//new
        ECF_R32G32B32_UINT,
        ECF_R32G32B32_SINT,
        ECF_R32G32B32_SFLOAT,
        ECF_R32G32B32A32_UINT,
        ECF_R32G32B32A32_SINT,
        ECF_R32G32B32A32_SFLOAT,//new
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
        ECF_B10G11R11_UFLOAT_PACK32,//new
        ECF_E5B9G9R9_UFLOAT_PACK32,//new
        ECF_D16_UNORM,
        ECF_X8_D24_UNORM_PACK32,
        ECF_D16_UNORM_S8_UINT,
        ECF_D24_UNORM_S8_UINT,
        ECF_D32_SFLOAT_S8_UINT,

        //! Block Compression Formats!
        ECF_BC1_RGB_UNORM_BLOCK,//new
        ECF_BC1_RGB_SRGB_BLOCK,
        ECF_BC1_RGBA_UNORM_BLOCK,//new
        ECF_BC1_RGBA_SRGB_BLOCK,
        ECF_BC2_UNORM_BLOCK,//new
        ECF_BC2_SRGB_BLOCK,
        ECF_BC3_UNORM_BLOCK,//new
        ECF_BC3_SRGB_BLOCK,

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
        default: return 0;
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
            ECF_R16G16B16_SFLOAT,
            ECF_R32G32B32_SFLOAT,
            ECF_R64_SFLOAT,
            ECF_R64G64_SFLOAT,
            ECF_R64G64B64_SFLOAT,
            ECF_R64G64B64A64_SFLOAT
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
            ECF_R16G16B16A16_SSCALED
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
            ECF_A8B8G8R8_SRGB_PACK32
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
            ECF_BC3_SRGB_BLOCK
        >::value;
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

    template<ECOLOR_FORMAT fmt, typename T>
    inline typename 
    std::enable_if<
        true,//std::is_same<T, double>::value && isScaledFormat<fmt>(),
        void
    >::type
    decodePixels(const void* _pix, T* _output, uint64_t _scale);

    template<ECOLOR_FORMAT fmt, typename T>
    inline typename 
    std::enable_if<
        true,//std::is_same<T, double>::value && isScaledFormat<fmt>(),
        void
    >::type
    encodePixels(void* _pix, const T* _input, uint64_t _scale);

    template<ECOLOR_FORMAT fmt, typename T>
    inline typename 
    std::enable_if<
        true,//std::is_same<T, double>::value || std::is_same<T, uint64_t>::value || std::is_same<T, int64_t>::value,
        void
    >::type
    decodePixels(const void* _pix, T* _output);

    template<ECOLOR_FORMAT fmt, typename T>
    inline typename 
    std::enable_if<
        true,//std::is_same<T, double>::value || std::is_same<T, uint64_t>::value || std::is_same<T, int64_t>::value,
        void
    >::type
    encodePixels(void* _pix, const T* _input);

    template<ECOLOR_FORMAT fmt, typename T>
    inline typename
    std::enable_if<
        true,//std::is_same<T, double>::value && isBlockCompressionFormat<fmt>(),
        void
    >::type
    decodePixels(const void* _pix, T* _output, uint32_t _x, uint32_t _y);

    template<>
    inline void decodePixels<ECF_A1R5G5B5, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[0] = (pix & 0x1fu);
        _output[1] = ((pix>>5) & 0x1fu);
        _output[2] = ((pix>>10) & 0x1fu);
        _output[3] = pix >> 15;
    }

    template<>
    inline void decodePixels<ECF_R5G6B5, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0x1fULL);
        _output[1] = ((pix >> 5) & 0x3fULL);
        _output[2] = ((pix >> 11) & 0x1fULL);
    }

    /*
    namespace impl
    {
        template<typename T>
        inline void decode11_11_10f(const void* _pix, T* _output)
        {
            using fptr = float(*)(uint32_t);
            fptr f[3]{ &core::unpack11bitFloat, &core::unpack11bitFloat, &core::unpack10bitFloat };

            const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
            for (uint32_t i = 0u; i < 3u; ++i)
            {
                _output[i] = f[i](pix >> (11 * i));
            }
        }
        template<typename T>
        inline void encode11_11_10f(void* _pix, const T* _input)
        {
            using fptr = uint32_t(*)(float);
            fptr f[3]{ &core::to11bitFloat, &core::to11bitFloat, &core::to10bitFloat };

            uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
            pix = 0u;
            for (uint32_t i = 0u; i < 3u; ++i)
            {
                pix |= (f[i](_input[i]) << (11*i));
            }
        }
    }
    template<>
    inline void decodePixels<ECF_B10G11R11_UFLOAT_PACK32, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode11_11_10f(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_B10G11R11_UFLOAT_PACK32, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode11_11_10f(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_B10G11R11_UFLOAT_PACK32, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode11_11_10f(_pix, _output);
    }
    template<>
    inline void encodePixels<ECF_B10G11R11_UFLOAT_PACK32, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode11_11_10f(_pix, _input);
    }
    template<>
    inline void encodePixels<ECF_B10G11R11_UFLOAT_PACK32, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode11_11_10f(_pix, _input);
    }
    template<>
    inline void encodePixels<ECF_B10G11R11_UFLOAT_PACK32, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode11_11_10f(_pix, _input);
    }

    namespace impl
    {
        template<typename T>
        inline void decode16f(const void* _pix, T* _output, uint32_t _chCnt)
        {
            using fptr = float(*)(uint16_t);

            fptr f = &core::Float16Compressor::decompress;

            const uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
            for (uint32_t i = 0u; i < _chCnt; ++i)
                _output[i] = f(pix >> (i*16));
        }
        template<typename T>
        inline void encode16f(void* _pix, const T* _input, uint32_t _chCnt)
        {
            using fptr = uint16_t(*)(float);

            fptr f = &core::Float16Compressor::compress;

            const uint16_t* pix = reinterpret_cast<uint16_t*>(_pix);
            for (uint32_t i = 0u; i < _chCnt; ++i)
            {
                pix[i] = f(_input[i]);
            }
        }
    }

    template<>
    inline void decodePixels<ECF_R16_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R16_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R16_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R16G16_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R16G16_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R16G16_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R16G16B16_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R16G16B16_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R16G16B16_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R16G16B16A16_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 4u);
    }
    template<>
    inline void decodePixels<ECF_R16G16B16A16_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 4u);
    }
    template<>
    inline void decodePixels<ECF_R16G16B16A16_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode16f(_pix, _output, 4u);
    }

    template<>
    inline void encodePixels<ECF_R16_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R16_SFLOAT, int64_t>(void* _pix, const int64_t* _input,  uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R16_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input,  uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R16G16_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R16G16_SFLOAT, int64_t>(void* _pix, const int64_t* _input,  uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R16G16_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input,  uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R16G16B16_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R16G16B16_SFLOAT, int64_t>(void* _pix, const int64_t* _input,  uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R16G16B16_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input,  uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R16G16B16A16_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 4u);
    }
    template<>
    inline void encodePixels<ECF_R16G16B16A16_SFLOAT, int64_t>(void* _pix, const int64_t* _input,  uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 4u);
    }
    template<>
    inline void encodePixels<ECF_R16G16B16A16_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input,  uint64_t _scale)
    {
        impl::encode16f(_pix, _input, 4u);
    }

    namespace impl
    {
        template<typename T>
        inline void decode32f(const void* _pix, T* _output, uint32_t _chCnt)
        {
            const float* pix = reinterpret_cast<const float*>(_pix);
            for (uint32_t i = 0u; i < _chCnt; ++i)
                _output[i] = pix[i];
        }
        template<typename T>
        inline void encode32f(void* _pix, const T* _input, uint32_t _chCnt)
        {
            const float* pix = reinterpret_cast<const float*>(_pix);
            for (uint32_t i = 0u; i < _chCnt; ++i)
                pix[i] = _input[i];
        }
    }

    template<>
    inline void decodePixels<ECF_R32_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R32_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R32_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R32G32_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R32G32_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R32G32_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32A32_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 4u);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32A32_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 4u);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32A32_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode32f(_pix, _output, 4u);
    }

    template<>
    inline void encodePixels<ECF_R32_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R32_SFLOAT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R32_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R32G32_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R32G32_SFLOAT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R32G32_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R32G32B32_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R32G32B32_SFLOAT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R32G32B32_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R32G32B32A32_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 4u);
    }
    template<>
    inline void encodePixels<ECF_R32G32B32A32_SFLOAT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 4u);
    }
    template<>
    inline void encodePixels<ECF_R32G32B32A32_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode32f(_pix, _input, 4u);
    }

    namespace impl
    {
        template<typename T>
        inline void decode64f(const void* _pix, T* _output, uint32_t _chCnt)
        {
            const double* pix = reinterpret_cast<const double*>(_pix);
            for (uint32_t i = 0u; i < _chCnt; ++i)
                _output[i] = pix[i];
        }
        template<typename T>
        inline void encode64f(void* _pix, const T* _input, uint32_t _chCnt)
        {
            const double* pix = reinterpret_cast<const double*>(_pix);
            for (uint32_t i = 0u; i < _chCnt; ++i)
                pix[i] = _input[i];
        }
    }

    template<>
    inline void decodePixels<ECF_R64_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R64_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R64_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R64G64_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R64G64_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R64G64_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64A64_SFLOAT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 4u);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64A64_SFLOAT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 4u);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64A64_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode64f(_pix, _output, 4u);
    }

    template<>
    inline void encodePixels<ECF_R64_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R64_SFLOAT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R64_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R64G64_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R64G64_SFLOAT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R64G64_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R64G64B64_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R64G64B64_SFLOAT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R64G64B64_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R64G64B64A64_SFLOAT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 4u);
    }
    template<>
    inline void encodePixels<ECF_R64G64B64A64_SFLOAT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 4u);
    }
    template<>
    inline void encodePixels<ECF_R64G64B64A64_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode64f(_pix, _input, 4u);
    }

    namespace impl
    {
        template<typename T>
        inline void decode8ui(const void* _pix, T* _output, uint32_t _chCnt)
        {
            const uint8_t* pix = reinterpret_cast<const uint8_t*>(_pix);
            for (uint32_t i = 0u; i < _chCnt; ++i)
                _output[i] = pix[i];
        }
        template<typename T>
        inline void encode8ui(void* _pix, const T* _input, uint32_t _chCnt)
        {
            uint8_t* pix = reinterpret_cast<uint8_t*>(_pix);
            for (uint32_t i = 0u; i < _chCnt; ++i)
                _input[i] = pix[i];
        }
    }

    template<>
    inline void decodePixels<ECF_R8_UINT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R8_UINT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R8_UINT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 1u);
    }
    template<>
    inline void decodePixels<ECF_R8G8_UINT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R8G8_UINT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R8G8_UINT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 2u);
    }
    template<>
    inline void decodePixels<ECF_R8G8B8_UINT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R8G8B8_UINT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R8G8B8_UINT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 3u);
    }
    template<>
    inline void decodePixels<ECF_R8G8B8A8_UINT, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 4u);
    }
    template<>
    inline void decodePixels<ECF_R8G8B8A8_UINT, int64_t>(const void* _pix, int64_t* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 4u);
    }
    template<>
    inline void decodePixels<ECF_R8G8B8A8_UINT, uint64_t>(const void* _pix, uint64_t* _output, uint64_t _scale)
    {
        impl::decode8ui(_pix, _output, 4u);
    }

    template<>
    inline void encodePixels<ECF_R8_UINT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R8_UINT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R8_UINT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 1u);
    }
    template<>
    inline void encodePixels<ECF_R8G8_UINT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R8G8_UINT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R8G8_UINT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 2u);
    }
    template<>
    inline void encodePixels<ECF_R8G8B8_UINT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R8G8B8_UINT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R8G8B8_UINT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 3u);
    }
    template<>
    inline void encodePixels<ECF_R8G8B8A8_UINT, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 4u);
    }
    template<>
    inline void encodePixels<ECF_R8G8B8A8_UINT, int64_t>(void* _pix, const int64_t* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 4u);
    }
    template<>
    inline void encodePixels<ECF_R8G8B8A8_UINT, uint64_t>(void* _pix, const uint64_t* _input, uint64_t _scale)
    {
        impl::encode8ui(_pix, _input, 4u);
    }
    */

    //integer formats decode/encode
    template<>
    inline void decodePixels<ECF_R4G4_UNORM_PACK8, double>(const void* _pix, double* _output)
    {
        const uint8_t& pix = reinterpret_cast<const uint8_t*>(_pix)[0];
        _output[1] = ((pix >> 0) & 0xfULL) / 15.;
        _output[0] = ((pix >> 4) & 0xfULL) / 15.;
    }
    template<>
    inline void encodePixels<ECF_R4G4_UNORM_PACK8, double>(void* _pix, const double* _input)
    {
        uint8_t& pix = reinterpret_cast<uint8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xfULL;
            pix &= (~(mask << 0));
            double inp = _input[1];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint8_t mask = 0xfULL;
            pix &= (~(mask << 4));
            double inp = _input[0];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 4);
        }
    }
    template<>
    inline void decodePixels<ECF_R4G4B4A4_UNORM_PACK16, double>(const void* _pix, double* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[3] = ((pix >> 0) & 0xfULL) / 15.;
        _output[2] = ((pix >> 4) & 0xfULL) / 15.;
        _output[1] = ((pix >> 8) & 0xfULL) / 15.;
        _output[0] = ((pix >> 12) & 0xfULL) / 15.;
    }
    template<>
    inline void encodePixels<ECF_R4G4B4A4_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 0));
            double inp = _input[3];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 4));
            double inp = _input[2];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 4);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 12));
            double inp = _input[0];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 12);
        }

    }
    template<>
    inline void decodePixels<ECF_B4G4R4A4_UNORM_PACK16, double>(const void* _pix, double* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[3] = ((pix >> 0) & 0xfULL) / 15.;
        _output[0] = ((pix >> 4) & 0xfULL) / 15.;
        _output[1] = ((pix >> 8) & 0xfULL) / 15.;
        _output[2] = ((pix >> 12) & 0xfULL) / 15.;
    }
    template<>
    inline void encodePixels<ECF_B4G4R4A4_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 0));
            double inp = _input[3];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 4));
            double inp = _input[0];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 4);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 12));
            double inp = _input[2];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 12);
        }

    }
    template<>
    inline void decodePixels<ECF_R5G6B5_UNORM_PACK16, double>(const void* _pix, double* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0x1fULL) / 31.;
        _output[1] = ((pix >> 5) & 0x3fULL) / 63.;
        _output[0] = ((pix >> 11) & 0x1fULL) / 31.;
    }
    template<>
    inline void encodePixels<ECF_R5G6B5_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0x3fULL;
            pix &= (~(mask << 5));
            double inp = _input[1];
            inp *= 63.;
            pix |= ((uint64_t(inp) & mask) << 5);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 11));
            double inp = _input[0];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 11);
        }

    }
    template<>
    inline void decodePixels<ECF_B5G6R5_UNORM_PACK16, double>(const void* _pix, double* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0x1fULL) / 31.;
        _output[1] = ((pix >> 5) & 0x3fULL) / 63.;
        _output[2] = ((pix >> 11) & 0x1fULL) / 31.;
    }
    template<>
    inline void encodePixels<ECF_B5G6R5_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0x3fULL;
            pix &= (~(mask << 5));
            double inp = _input[1];
            inp *= 63.;
            pix |= ((uint64_t(inp) & mask) << 5);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 11));
            double inp = _input[2];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 11);
        }

    }
    template<>
    inline void decodePixels<ECF_R5G5B5A1_UNORM_PACK16, double>(const void* _pix, double* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[3] = ((pix >> 0) & 0x1ULL) / 1.;
        _output[2] = ((pix >> 1) & 0x1fULL) / 31.;
        _output[1] = ((pix >> 6) & 0x1fULL) / 31.;
        _output[0] = ((pix >> 11) & 0x1fULL) / 31.;
    }
    template<>
    inline void encodePixels<ECF_R5G5B5A1_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0x1ULL;
            pix &= (~(mask << 0));
            double inp = _input[3];
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 1));
            double inp = _input[2];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 1);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 6));
            double inp = _input[1];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 6);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 11));
            double inp = _input[0];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 11);
        }

    }
    template<>
    inline void decodePixels<ECF_B5G5R5A1_UNORM_PACK16, double>(const void* _pix, double* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[3] = ((pix >> 0) & 0x1ULL) / 1.;
        _output[0] = ((pix >> 1) & 0x1fULL) / 31.;
        _output[1] = ((pix >> 6) & 0x1fULL) / 31.;
        _output[2] = ((pix >> 11) & 0x1fULL) / 31.;
    }
    template<>
    inline void encodePixels<ECF_B5G5R5A1_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0x1ULL;
            pix &= (~(mask << 0));
            double inp = _input[3];
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 1));
            double inp = _input[0];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 1);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 6));
            double inp = _input[1];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 6);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 11));
            double inp = _input[2];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 11);
        }

    }
    template<>
    inline void decodePixels<ECF_A1R5G5B5_UNORM_PACK16, double>(const void* _pix, double* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0x1fULL) / 31.;
        _output[1] = ((pix >> 5) & 0x1fULL) / 31.;
        _output[0] = ((pix >> 10) & 0x1fULL) / 31.;
        _output[3] = ((pix >> 15) & 0x1ULL) / 1.;
    }
    template<>
    inline void encodePixels<ECF_A1R5G5B5_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 5));
            double inp = _input[1];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 5);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 10));
            double inp = _input[0];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const uint16_t mask = 0x1ULL;
            pix &= (~(mask << 15));
            double inp = _input[3];
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 15);
        }

    }
    template<>
    inline void decodePixels<ECF_R8_UNORM, double>(const void* _pix, double* _output)
    {
        const uint8_t& pix = reinterpret_cast<const uint8_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255.;
    }
    template<>
    inline void encodePixels<ECF_R8_UNORM, double>(void* _pix, const double* _input)
    {
        uint8_t& pix = reinterpret_cast<uint8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R8_SNORM, double>(const void* _pix, double* _output)
    {
        const int8_t& pix = reinterpret_cast<const int8_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127.;
    }
    template<>
    inline void encodePixels<ECF_R8_SNORM, double>(void* _pix, const double* _input)
    {
        int8_t& pix = reinterpret_cast<int8_t*>(_pix)[0];
        {
            const int8_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R8_USCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint8_t& pix = reinterpret_cast<const uint8_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R8_USCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint8_t& pix = reinterpret_cast<uint8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R8_SSCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int8_t& pix = reinterpret_cast<const int8_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R8_SSCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int8_t& pix = reinterpret_cast<int8_t*>(_pix)[0];
        {
            const int8_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R8_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint8_t& pix = reinterpret_cast<const uint8_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL);
    }
    template<>
    inline void encodePixels<ECF_R8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint8_t& pix = reinterpret_cast<uint8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R8_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int8_t& pix = reinterpret_cast<const int8_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL);
    }
    template<>
    inline void encodePixels<ECF_R8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int8_t& pix = reinterpret_cast<int8_t*>(_pix)[0];
        {
            const int8_t mask = 0xffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R8_SRGB, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint8_t& pix = reinterpret_cast<const uint8_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        {
            double lin = _output[0] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[0] = lin * 255.;
        };
    }
    template<>
    inline void encodePixels<ECF_R8_SRGB, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint8_t& pix = reinterpret_cast<uint8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8_UNORM, double>(const void* _pix, double* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
    }
    template<>
    inline void encodePixels<ECF_R8G8_UNORM, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8_SNORM, double>(const void* _pix, double* _output)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
    }
    template<>
    inline void encodePixels<ECF_R8G8_SNORM, double>(void* _pix, const double* _input)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const int16_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int16_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8_USCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R8G8_USCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8_SSCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R8G8_SSCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const int16_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int16_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
    }
    template<>
    inline void encodePixels<ECF_R8G8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
    }
    template<>
    inline void encodePixels<ECF_R8G8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const int16_t mask = 0xffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int16_t mask = 0xffLL;
            pix &= (~(mask << 8));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8_SRGB, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        {
            double lin = _output[0] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[0] = lin * 255.;
        };
        _output[1] = ((pix >> 8) & 0xffULL);
        {
            double lin = _output[1] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[1] = lin * 255.;
        };
    }
    template<>
    inline void encodePixels<ECF_R8G8_SRGB, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 8);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8_UNORM, double>(const void* _pix, double* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
        _output[2] = ((pix >> 16) & 0xffULL) / 255.;
    }
    template<>
    inline void encodePixels<ECF_R8G8B8_UNORM, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8_SNORM, double>(const void* _pix, double* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
        _output[2] = ((pix >> 16) & 0xffLL) / 127.;
    }
    template<>
    inline void encodePixels<ECF_R8G8B8_SNORM, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8_USCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
        _output[2] = ((pix >> 16) & 0xffULL) / 255. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R8G8B8_USCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8_SSCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
        _output[2] = ((pix >> 16) & 0xffLL) / 127. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R8G8B8_SSCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
        _output[2] = ((pix >> 16) & 0xffULL);
    }
    template<>
    inline void encodePixels<ECF_R8G8B8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
        _output[2] = ((pix >> 16) & 0xffLL);
    }
    template<>
    inline void encodePixels<ECF_R8G8B8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8_SRGB, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        {
            double lin = _output[0] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[0] = lin * 255.;
        };
        _output[1] = ((pix >> 8) & 0xffULL);
        {
            double lin = _output[1] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[1] = lin * 255.;
        };
        _output[2] = ((pix >> 16) & 0xffULL);
        {
            double lin = _output[2] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[2] = lin * 255.;
        };
    }
    template<>
    inline void encodePixels<ECF_R8G8B8_SRGB, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[2];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8_UNORM, double>(const void* _pix, double* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
        _output[0] = ((pix >> 16) & 0xffULL) / 255.;
    }
    template<>
    inline void encodePixels<ECF_B8G8R8_UNORM, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8_SNORM, double>(const void* _pix, double* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
        _output[0] = ((pix >> 16) & 0xffLL) / 127.;
    }
    template<>
    inline void encodePixels<ECF_B8G8R8_SNORM, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8_USCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
        _output[0] = ((pix >> 16) & 0xffULL) / 255. * _scale;
    }
    template<>
    inline void encodePixels<ECF_B8G8R8_USCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8_SSCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
        _output[0] = ((pix >> 16) & 0xffLL) / 127. * _scale;
    }
    template<>
    inline void encodePixels<ECF_B8G8R8_SSCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
        _output[0] = ((pix >> 16) & 0xffULL);
    }
    template<>
    inline void encodePixels<ECF_B8G8R8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
        _output[0] = ((pix >> 16) & 0xffLL);
    }
    template<>
    inline void encodePixels<ECF_B8G8R8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8_SRGB, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffULL);
        {
            double lin = _output[2] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[2] = lin * 255.;
        };
        _output[1] = ((pix >> 8) & 0xffULL);
        {
            double lin = _output[1] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[1] = lin * 255.;
        };
        _output[0] = ((pix >> 16) & 0xffULL);
        {
            double lin = _output[0] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[0] = lin * 255.;
        };
    }
    template<>
    inline void encodePixels<ECF_B8G8R8_SRGB, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[2];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[0];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8A8_UNORM, double>(const void* _pix, double* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
        _output[2] = ((pix >> 16) & 0xffULL) / 255.;
        _output[3] = ((pix >> 24) & 0xffULL) / 255.;
    }
    template<>
    inline void encodePixels<ECF_R8G8B8A8_UNORM, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8A8_SNORM, double>(const void* _pix, double* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
        _output[2] = ((pix >> 16) & 0xffLL) / 127.;
        _output[3] = ((pix >> 24) & 0xffLL) / 127.;
    }
    template<>
    inline void encodePixels<ECF_R8G8B8A8_SNORM, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8A8_USCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
        _output[2] = ((pix >> 16) & 0xffULL) / 255. * _scale;
        _output[3] = ((pix >> 24) & 0xffULL) / 255. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R8G8B8A8_USCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8A8_SSCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
        _output[2] = ((pix >> 16) & 0xffLL) / 127. * _scale;
        _output[3] = ((pix >> 24) & 0xffLL) / 127. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R8G8B8A8_SSCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8A8_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
        _output[2] = ((pix >> 16) & 0xffULL);
        _output[3] = ((pix >> 24) & 0xffULL);
    }
    template<>
    inline void encodePixels<ECF_R8G8B8A8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8A8_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
        _output[2] = ((pix >> 16) & 0xffLL);
        _output[3] = ((pix >> 24) & 0xffLL);
    }
    template<>
    inline void encodePixels<ECF_R8G8B8A8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_R8G8B8A8_SRGB, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        {
            double lin = _output[0] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[0] = lin * 255.;
        };
        _output[1] = ((pix >> 8) & 0xffULL);
        {
            double lin = _output[1] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[1] = lin * 255.;
        };
        _output[2] = ((pix >> 16) & 0xffULL);
        {
            double lin = _output[2] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[2] = lin * 255.;
        };
        _output[3] = ((pix >> 24) & 0xffULL);
        {
            double lin = _output[3] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[3] = lin * 255.;
        };
    }
    template<>
    inline void encodePixels<ECF_R8G8B8A8_SRGB, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[2];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            uint64_t inp = _input[3];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8A8_UNORM, double>(const void* _pix, double* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
        _output[0] = ((pix >> 16) & 0xffULL) / 255.;
        _output[3] = ((pix >> 24) & 0xffULL) / 255.;
    }
    template<>
    inline void encodePixels<ECF_B8G8R8A8_UNORM, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8A8_SNORM, double>(const void* _pix, double* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
        _output[0] = ((pix >> 16) & 0xffLL) / 127.;
        _output[3] = ((pix >> 24) & 0xffLL) / 127.;
    }
    template<>
    inline void encodePixels<ECF_B8G8R8A8_SNORM, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8A8_USCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
        _output[0] = ((pix >> 16) & 0xffULL) / 255. * _scale;
        _output[3] = ((pix >> 24) & 0xffULL) / 255. * _scale;
    }
    template<>
    inline void encodePixels<ECF_B8G8R8A8_USCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8A8_SSCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
        _output[0] = ((pix >> 16) & 0xffLL) / 127. * _scale;
        _output[3] = ((pix >> 24) & 0xffLL) / 127. * _scale;
    }
    template<>
    inline void encodePixels<ECF_B8G8R8A8_SSCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8A8_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
        _output[0] = ((pix >> 16) & 0xffULL);
        _output[3] = ((pix >> 24) & 0xffULL);
    }
    template<>
    inline void encodePixels<ECF_B8G8R8A8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8A8_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
        _output[0] = ((pix >> 16) & 0xffLL);
        _output[3] = ((pix >> 24) & 0xffLL);
    }
    template<>
    inline void encodePixels<ECF_B8G8R8A8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_B8G8R8A8_SRGB, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0xffULL);
        {
            double lin = _output[2] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[2] = lin * 255.;
        };
        _output[1] = ((pix >> 8) & 0xffULL);
        {
            double lin = _output[1] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[1] = lin * 255.;
        };
        _output[0] = ((pix >> 16) & 0xffULL);
        {
            double lin = _output[0] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[0] = lin * 255.;
        };
        _output[3] = ((pix >> 24) & 0xffULL);
        {
            double lin = _output[3] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[3] = lin * 255.;
        };
    }
    template<>
    inline void encodePixels<ECF_B8G8R8A8_SRGB, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[2];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[0];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            uint64_t inp = _input[3];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_A8B8G8R8_UNORM_PACK32, double>(const void* _pix, double* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
        _output[2] = ((pix >> 16) & 0xffULL) / 255.;
        _output[3] = ((pix >> 24) & 0xffULL) / 255.;
    }
    template<>
    inline void encodePixels<ECF_A8B8G8R8_UNORM_PACK32, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_A8B8G8R8_SNORM_PACK32, double>(const void* _pix, double* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
        _output[2] = ((pix >> 16) & 0xffLL) / 127.;
        _output[3] = ((pix >> 24) & 0xffLL) / 127.;
    }
    template<>
    inline void encodePixels<ECF_A8B8G8R8_SNORM_PACK32, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_A8B8G8R8_USCALED_PACK32, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
        _output[2] = ((pix >> 16) & 0xffULL) / 255. * _scale;
        _output[3] = ((pix >> 24) & 0xffULL) / 255. * _scale;
    }
    template<>
    inline void encodePixels<ECF_A8B8G8R8_USCALED_PACK32, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp /= _scale;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_A8B8G8R8_SSCALED_PACK32, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
        _output[2] = ((pix >> 16) & 0xffLL) / 127. * _scale;
        _output[3] = ((pix >> 24) & 0xffLL) / 127. * _scale;
    }
    template<>
    inline void encodePixels<ECF_A8B8G8R8_SSCALED_PACK32, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp /= _scale;
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_A8B8G8R8_UINT_PACK32, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
        _output[2] = ((pix >> 16) & 0xffULL);
        _output[3] = ((pix >> 24) & 0xffULL);
    }
    template<>
    inline void encodePixels<ECF_A8B8G8R8_UINT_PACK32, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_A8B8G8R8_SINT_PACK32, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
        _output[2] = ((pix >> 16) & 0xffLL);
        _output[3] = ((pix >> 24) & 0xffLL);
    }
    template<>
    inline void encodePixels<ECF_A8B8G8R8_SINT_PACK32, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_A8B8G8R8_SRGB_PACK32, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        {
            double lin = _output[0] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[0] = lin * 255.;
        };
        _output[1] = ((pix >> 8) & 0xffULL);
        {
            double lin = _output[1] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[1] = lin * 255.;
        };
        _output[2] = ((pix >> 16) & 0xffULL);
        {
            double lin = _output[2] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[2] = lin * 255.;
        };
        _output[3] = ((pix >> 24) & 0xffULL);
        {
            double lin = _output[3] / 255.;
            if (lin <= 0.4045) lin /= 12.92;
            else lin = pow((lin + 0.055) / 1.055, 2.4);
            _output[3] = lin * 255.;
        };
    }
    template<>
    inline void encodePixels<ECF_A8B8G8R8_SRGB_PACK32, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[2];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            uint64_t inp = _input[3];
            double s = inp / 255.;
            if (s <= 0.0031308) s *= 12.92;
            else s = 1.055 * pow(s, 1. / 2.4) - 0.055;
            inp = s * 255;
            pix |= ((inp & mask) << 24);
        }

    }
    template<>
    inline void decodePixels<ECF_A2R10G10B10_UNORM_PACK32, double>(const void* _pix, double* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0x3ffULL) / 1023.;
        _output[1] = ((pix >> 10) & 0x3ffULL) / 1023.;
        _output[0] = ((pix >> 20) & 0x3ffULL) / 1023.;
        _output[3] = ((pix >> 30) & 0x3ULL) / 3.;
    }
    template<>
    inline void encodePixels<ECF_A2R10G10B10_UNORM_PACK32, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            double inp = _input[0];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp *= 3.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2R10G10B10_SNORM_PACK32, double>(const void* _pix, double* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0x3ffLL) / 511.;
        _output[1] = ((pix >> 10) & 0x3ffLL) / 511.;
        _output[0] = ((pix >> 20) & 0x3ffLL) / 511.;
        _output[3] = ((pix >> 30) & 0x3LL) / 1.;
    }
    template<>
    inline void encodePixels<ECF_A2R10G10B10_SNORM_PACK32, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            double inp = _input[0];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2R10G10B10_USCALED_PACK32, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0x3ffULL) / 1023. * _scale;
        _output[1] = ((pix >> 10) & 0x3ffULL) / 1023. * _scale;
        _output[0] = ((pix >> 20) & 0x3ffULL) / 1023. * _scale;
        _output[3] = ((pix >> 30) & 0x3ULL) / 3. * _scale;
    }
    template<>
    inline void encodePixels<ECF_A2R10G10B10_USCALED_PACK32, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp /= _scale;
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp /= _scale;
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            double inp = _input[0];
            inp /= _scale;
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp /= _scale;
            inp *= 3.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2R10G10B10_SSCALED_PACK32, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0x3ffLL) / 511. * _scale;
        _output[1] = ((pix >> 10) & 0x3ffLL) / 511. * _scale;
        _output[0] = ((pix >> 20) & 0x3ffLL) / 511. * _scale;
        _output[3] = ((pix >> 30) & 0x3LL) / 1. * _scale;
    }
    template<>
    inline void encodePixels<ECF_A2R10G10B10_SSCALED_PACK32, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp /= _scale;
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp /= _scale;
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            double inp = _input[0];
            inp /= _scale;
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp /= _scale;
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2R10G10B10_UINT_PACK32, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0x3ffULL);
        _output[1] = ((pix >> 10) & 0x3ffULL);
        _output[0] = ((pix >> 20) & 0x3ffULL);
        _output[3] = ((pix >> 30) & 0x3ULL);
    }
    template<>
    inline void encodePixels<ECF_A2R10G10B10_UINT_PACK32, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2R10G10B10_SINT_PACK32, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[2] = ((pix >> 0) & 0x3ffLL);
        _output[1] = ((pix >> 10) & 0x3ffLL);
        _output[0] = ((pix >> 20) & 0x3ffLL);
        _output[3] = ((pix >> 30) & 0x3LL);
    }
    template<>
    inline void encodePixels<ECF_A2R10G10B10_SINT_PACK32, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2B10G10R10_UNORM_PACK32, double>(const void* _pix, double* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0x3ffULL) / 1023.;
        _output[1] = ((pix >> 10) & 0x3ffULL) / 1023.;
        _output[2] = ((pix >> 20) & 0x3ffULL) / 1023.;
        _output[3] = ((pix >> 30) & 0x3ULL) / 3.;
    }
    template<>
    inline void encodePixels<ECF_A2B10G10R10_UNORM_PACK32, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            double inp = _input[2];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp *= 3.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2B10G10R10_SNORM_PACK32, double>(const void* _pix, double* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0x3ffLL) / 511.;
        _output[1] = ((pix >> 10) & 0x3ffLL) / 511.;
        _output[2] = ((pix >> 20) & 0x3ffLL) / 511.;
        _output[3] = ((pix >> 30) & 0x3LL) / 1.;
    }
    template<>
    inline void encodePixels<ECF_A2B10G10R10_SNORM_PACK32, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            double inp = _input[2];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2B10G10R10_USCALED_PACK32, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0x3ffULL) / 1023. * _scale;
        _output[1] = ((pix >> 10) & 0x3ffULL) / 1023. * _scale;
        _output[2] = ((pix >> 20) & 0x3ffULL) / 1023. * _scale;
        _output[3] = ((pix >> 30) & 0x3ULL) / 3. * _scale;
    }
    template<>
    inline void encodePixels<ECF_A2B10G10R10_USCALED_PACK32, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp /= _scale;
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            double inp = _input[2];
            inp /= _scale;
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp /= _scale;
            inp *= 3.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2B10G10R10_SSCALED_PACK32, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0x3ffLL) / 511. * _scale;
        _output[1] = ((pix >> 10) & 0x3ffLL) / 511. * _scale;
        _output[2] = ((pix >> 20) & 0x3ffLL) / 511. * _scale;
        _output[3] = ((pix >> 30) & 0x3LL) / 1. * _scale;
    }
    template<>
    inline void encodePixels<ECF_A2B10G10R10_SSCALED_PACK32, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp /= _scale;
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            double inp = _input[2];
            inp /= _scale;
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp /= _scale;
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2B10G10R10_UINT_PACK32, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0x3ffULL);
        _output[1] = ((pix >> 10) & 0x3ffULL);
        _output[2] = ((pix >> 20) & 0x3ffULL);
        _output[3] = ((pix >> 30) & 0x3ULL);
    }
    template<>
    inline void encodePixels<ECF_A2B10G10R10_UINT_PACK32, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_A2B10G10R10_SINT_PACK32, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0x3ffLL);
        _output[1] = ((pix >> 10) & 0x3ffLL);
        _output[2] = ((pix >> 20) & 0x3ffLL);
        _output[3] = ((pix >> 30) & 0x3LL);
    }
    template<>
    inline void encodePixels<ECF_A2B10G10R10_SINT_PACK32, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 30);
        }

    }
    template<>
    inline void decodePixels<ECF_R16_UNORM, double>(const void* _pix, double* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535.;
    }
    template<>
    inline void encodePixels<ECF_R16_UNORM, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R16_SNORM, double>(const void* _pix, double* _output)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767.;
    }
    template<>
    inline void encodePixels<ECF_R16_SNORM, double>(void* _pix, const double* _input)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const int16_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R16_USCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R16_USCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R16_SSCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R16_SSCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const int16_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R16_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL);
    }
    template<>
    inline void encodePixels<ECF_R16_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R16_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL);
    }
    template<>
    inline void encodePixels<ECF_R16_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const int16_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16_UNORM, double>(const void* _pix, double* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535.;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535.;
    }
    template<>
    inline void encodePixels<ECF_R16G16_UNORM, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16_SNORM, double>(const void* _pix, double* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767.;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767.;
    }
    template<>
    inline void encodePixels<ECF_R16G16_SNORM, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16_USCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535. * _scale;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R16G16_USCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp /= _scale;
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16_SSCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767. * _scale;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R16G16_SSCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp /= _scale;
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL);
        _output[1] = ((pix >> 16) & 0xffffULL);
    }
    template<>
    inline void encodePixels<ECF_R16G16_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL);
        _output[1] = ((pix >> 16) & 0xffffLL);
    }
    template<>
    inline void encodePixels<ECF_R16G16_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 16);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16_UNORM, double>(const void* _pix, double* _output)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535.;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535.;
        _output[2] = ((pix >> 32) & 0xffffULL) / 65535.;
    }
    template<>
    inline void encodePixels<ECF_R16G16B16_UNORM, double>(void* _pix, const double* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 32);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16_SNORM, double>(const void* _pix, double* _output)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767.;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767.;
        _output[2] = ((pix >> 32) & 0xffffLL) / 32767.;
    }
    template<>
    inline void encodePixels<ECF_R16G16B16_SNORM, double>(void* _pix, const double* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 32);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16_USCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535. * _scale;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535. * _scale;
        _output[2] = ((pix >> 32) & 0xffffULL) / 65535. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R16G16B16_USCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp /= _scale;
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            inp /= _scale;
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 32);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16_SSCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767. * _scale;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767. * _scale;
        _output[2] = ((pix >> 32) & 0xffffLL) / 32767. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R16G16B16_SSCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp /= _scale;
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            inp /= _scale;
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 32);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL);
        _output[1] = ((pix >> 16) & 0xffffULL);
        _output[2] = ((pix >> 32) & 0xffffULL);
    }
    template<>
    inline void encodePixels<ECF_R16G16B16_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 16);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 32));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 32);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL);
        _output[1] = ((pix >> 16) & 0xffffLL);
        _output[2] = ((pix >> 32) & 0xffffLL);
    }
    template<>
    inline void encodePixels<ECF_R16G16B16_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 16);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 32));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 32);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16A16_UNORM, double>(const void* _pix, double* _output)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535.;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535.;
        _output[2] = ((pix >> 32) & 0xffffULL) / 65535.;
        _output[3] = ((pix >> 48) & 0xffffULL) / 65535.;
    }
    template<>
    inline void encodePixels<ECF_R16G16B16A16_UNORM, double>(void* _pix, const double* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 32);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 48));
            double inp = _input[3];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 48);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16A16_SNORM, double>(const void* _pix, double* _output)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767.;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767.;
        _output[2] = ((pix >> 32) & 0xffffLL) / 32767.;
        _output[3] = ((pix >> 48) & 0xffffLL) / 32767.;
    }
    template<>
    inline void encodePixels<ECF_R16G16B16A16_SNORM, double>(void* _pix, const double* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 32);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 48));
            double inp = _input[3];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 48);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16A16_USCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535. * _scale;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535. * _scale;
        _output[2] = ((pix >> 32) & 0xffffULL) / 65535. * _scale;
        _output[3] = ((pix >> 48) & 0xffffULL) / 65535. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R16G16B16A16_USCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp /= _scale;
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            inp /= _scale;
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 32);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 48));
            double inp = _input[3];
            inp /= _scale;
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 48);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16A16_SSCALED, double>(const void* _pix, double* _output, uint64_t _scale)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767. * _scale;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767. * _scale;
        _output[2] = ((pix >> 32) & 0xffffLL) / 32767. * _scale;
        _output[3] = ((pix >> 48) & 0xffffLL) / 32767. * _scale;
    }
    template<>
    inline void encodePixels<ECF_R16G16B16A16_SSCALED, double>(void* _pix, const double* _input, uint64_t _scale)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp /= _scale;
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp /= _scale;
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            inp /= _scale;
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 32);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 48));
            double inp = _input[3];
            inp /= _scale;
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 48);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16A16_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffULL);
        _output[1] = ((pix >> 16) & 0xffffULL);
        _output[2] = ((pix >> 32) & 0xffffULL);
        _output[3] = ((pix >> 48) & 0xffffULL);
    }
    template<>
    inline void encodePixels<ECF_R16G16B16A16_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 16);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 32));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 32);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 48));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 48);
        }

    }
    template<>
    inline void decodePixels<ECF_R16G16B16A16_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffLL);
        _output[1] = ((pix >> 16) & 0xffffLL);
        _output[2] = ((pix >> 32) & 0xffffLL);
        _output[3] = ((pix >> 48) & 0xffffLL);
    }
    template<>
    inline void encodePixels<ECF_R16G16B16A16_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 16);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 32));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 32);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 48));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 48);
        }

    }
    template<>
    inline void decodePixels<ECF_R32_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffffffULL);
    }
    template<>
    inline void encodePixels<ECF_R32_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffffffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R32_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffffffLL);
    }
    template<>
    inline void encodePixels<ECF_R32_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffffffffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R32G32_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffffffULL);
        _output[1] = ((pix >> 32) & 0xffffffffULL);
    }
    template<>
    inline void encodePixels<ECF_R32G32_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffffffULL;
            pix &= (~(mask << 32));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 32);
        }

    }
    template<>
    inline void decodePixels<ECF_R32G32_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffffffLL);
        _output[1] = ((pix >> 32) & 0xffffffffLL);
    }
    template<>
    inline void encodePixels<ECF_R32G32_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffffffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int64_t mask = 0xffffffffLL;
            pix &= (~(mask << 32));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 32);
        }

    }
    template<>
    inline void decodePixels<ECF_R32G32B32_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t* pix = reinterpret_cast<const uint32_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            _output[i] = pix[i];
    }
    template<>
    inline void encodePixels<ECF_R32G32B32_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t* pix = reinterpret_cast<uint32_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = _input[i];
    }
    template<>
    inline void decodePixels<ECF_R32G32B32_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t* pix = reinterpret_cast<const int32_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            _output[i] = pix[i];
    }
    template<>
    inline void encodePixels<ECF_R32G32B32_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t* pix = reinterpret_cast<int32_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = _input[i];
    }
    template<>
    inline void decodePixels<ECF_R32G32B32A32_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint32_t* pix = reinterpret_cast<const uint32_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            _output[i] = pix[i];
    }
    template<>
    inline void encodePixels<ECF_R32G32B32A32_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t* pix = reinterpret_cast<uint32_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            pix[i] = _input[i];
    }
    template<>
    inline void decodePixels<ECF_R32G32B32A32_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int32_t* pix = reinterpret_cast<const int32_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            _output[i] = pix[i];
    }
    template<>
    inline void encodePixels<ECF_R32G32B32A32_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t* pix = reinterpret_cast<int32_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            pix[i] = _input[i];
    }
    template<>
    inline void decodePixels<ECF_R64_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffffffffffffffULL);
    }
    template<>
    inline void encodePixels<ECF_R64_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffffffffffffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R64_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix)[0];
        _output[0] = ((pix >> 0) & 0xffffffffffffffffLL);
    }
    template<>
    inline void encodePixels<ECF_R64_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffffffffffffffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
    template<>
    inline void decodePixels<ECF_R64G64_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint64_t* pix = reinterpret_cast<const uint64_t*>(_pix);
        for (uint32_t i = 0u; i < 2u; ++i)
            _output[i] = pix[i];
    }
    template<>
    inline void encodePixels<ECF_R64G64_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t* pix = reinterpret_cast<uint64_t*>(_pix);
        for (uint32_t i = 0u; i < 2u; ++i)
            pix[i] = _input[i];
    }
    template<>
    inline void decodePixels<ECF_R64G64_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int64_t* pix = reinterpret_cast<const int64_t*>(_pix);
        for (uint32_t i = 0u; i < 2u; ++i)
            _output[i] = pix[i];
    }
    template<>
    inline void encodePixels<ECF_R64G64_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t* pix = reinterpret_cast<int64_t*>(_pix);
        for (uint32_t i = 0u; i < 2u; ++i)
            pix[i] = _input[i];
    }
    template<>
    inline void decodePixels<ECF_R64G64B64_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint64_t* pix = reinterpret_cast<const uint64_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            _output[i] = pix[i];
    }
    template<>
    inline void encodePixels<ECF_R64G64B64_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t* pix = reinterpret_cast<uint64_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = _input[i];
    }
    template<>
    inline void decodePixels<ECF_R64G64B64_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int64_t* pix = reinterpret_cast<const int64_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            _output[i] = pix[i];
    }
    template<>
    inline void encodePixels<ECF_R64G64B64_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t* pix = reinterpret_cast<int64_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = _input[i];
    }
    template<>
    inline void decodePixels<ECF_R64G64B64A64_UINT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        const uint64_t* pix = reinterpret_cast<const uint64_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            _output[i] = pix[i];
    }
    template<>
    inline void encodePixels<ECF_R64G64B64A64_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t* pix = reinterpret_cast<uint64_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            pix[i] = _input[i];
    }
    template<>
    inline void decodePixels<ECF_R64G64B64A64_SINT, int64_t>(const void* _pix, int64_t* _output)
    {
        const int64_t* pix = reinterpret_cast<const int64_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            _output[i] = pix[i];
    }
    template<>
    inline void encodePixels<ECF_R64G64B64A64_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t* pix = reinterpret_cast<int64_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            pix[i] = _input[i];
    }


    //floating point formats decode/encode
    namespace impl
    {
        template<typename T>
        inline void decode_r11g11b10f(const void* _pix, T* _output)
        {
            using fptr = float(*)(uint32_t);
            fptr f[3]{ &core::unpack11bitFloat, &core::unpack11bitFloat, &core::unpack10bitFloat };

            const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
            for (uint32_t i = 0u; i < 3u; ++i)
                _output[i] = f[i](pix >> 11 * i);
        }
        template<typename T>
        inline void encode_r11g11b10f(void* _pix, const T* _input)
        {
            using fptr = uint32_t(*)(float);
            fptr f[3]{ &core::to11bitFloat, &core::to11bitFloat, &core::to10bitFloat};

            uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
            pix = 0u;
            for (uint32_t i = 0u; i < 3u; ++i)
                pix |= (f[i](_input[i]) << (11*i));
        }
    }
    template<>
    inline void decodePixels<ECF_B10G11R11_UFLOAT_PACK32, double>(const void* _pix, double* _output)
    {
        impl::decode_r11g11b10f<double>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_B10G11R11_UFLOAT_PACK32, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decode_r11g11b10f<uint64_t>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_B10G11R11_UFLOAT_PACK32, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decode_r11g11b10f<int64_t>(_pix, _output);
    }
    template<>
    inline void encodePixels<ECF_B10G11R11_UFLOAT_PACK32, double>(void* _pix, const double* _output)
    {
        impl::encode_r11g11b10f<double>(_pix, _output);
    }
    template<>
    inline void encodePixels<ECF_B10G11R11_UFLOAT_PACK32, uint64_t>(void* _pix, const uint64_t* _output)
    {
        impl::encode_r11g11b10f<uint64_t>(_pix, _output);
    }
    template<>
    inline void encodePixels<ECF_B10G11R11_UFLOAT_PACK32, int64_t>(void* _pix, const int64_t* _output)
    {
        impl::encode_r11g11b10f<int64_t>(_pix, _output);
    }
    namespace impl
    {
        template<typename T, uint32_t chCnt>
        inline void decodef16(const void* _pix, T* _output)
        {
            const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix)[0];
            for (uint32_t i = 0u; i < chCnt; ++i)
                _output[i] = core::Float16Compressor::decompress(pix >> i*16);
        }
        template<typename T, uint32_t chCnt>
        inline void encodef16(void* _pix, const T* _input)
        {
            uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
            for (uint32_t i = 0u; i < chCnt; ++i)
            {
                pix &= (~(0xffffu<<(16u*i)));
                pix |= (uint64_t(core::Float16Compressor::compress(_input[i])) << (16u*i));
            }
        }
    }
    template<>
    inline void encodePixels<ECF_R16_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef16<double, 1u>(_pix, _input);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void encodePixels<ECF_R16G16_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef16<double, 2u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R16G16B16_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef16<double, 3u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R16G16B16A16_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef16<double, 4u>(_pix, _input);
    }
    template<>
    inline void encodePixels<ECF_R16_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef16<uint64_t, 1u>(_pix, _input);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void encodePixels<ECF_R16G16_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef16<uint64_t, 2u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R16G16B16_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef16<uint64_t, 3u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R16G16B16A16_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef16<uint64_t, 4u>(_pix, _input);
    }
    template<>
    inline void encodePixels<ECF_R16_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef16<int64_t, 1u>(_pix, _input);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void encodePixels<ECF_R16G16_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef16<int64_t, 2u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R16G16B16_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef16<int64_t, 3u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R16G16B16A16_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef16<int64_t, 4u>(_pix, _input);
    }
    template<>
    inline void decodePixels<ECF_R16_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef16<double, 1u>(_pix, _output);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void decodePixels<ECF_R16G16_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef16<double, 2u>(_pix, _output);
    }
    template<> // mapped to GL_RGBA
    inline void decodePixels<ECF_R16G16B16_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef16<double, 3u>(_pix, _output);
    }
    template<> // mapped to GL_RGBA
    inline void decodePixels<ECF_R16G16B16A16_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef16<double, 4u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R16_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef16<uint64_t, 1u>(_pix, _output);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void decodePixels<ECF_R16G16_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef16<uint64_t, 2u>(_pix, _output);
    }
    template<> // mapped to GL_RGBA
    inline void decodePixels<ECF_R16G16B16_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef16<uint64_t, 3u>(_pix, _output);
    }
    template<> // mapped to GL_RGBA
    inline void decodePixels<ECF_R16G16B16A16_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef16<uint64_t, 4u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R16_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef16<int64_t, 1u>(_pix, _output);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void decodePixels<ECF_R16G16_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef16<int64_t, 2u>(_pix, _output);
    }
    template<> // mapped to GL_RGBA
    inline void decodePixels<ECF_R16G16B16_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef16<int64_t, 3u>(_pix, _output);
    }
    template<> // mapped to GL_RGBA
    inline void decodePixels<ECF_R16G16B16A16_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef16<int64_t, 4u>(_pix, _output);
    }
    namespace impl
    {
        template<typename T, uint32_t chCnt>
        inline void decodef32(const void* _pix, T* _output)
        {
            const float* pix = reinterpret_cast<const float*>(_pix);
            for (uint32_t i = 0u; i < chCnt; ++i)
                _output[i] = pix[i];
        }
        template<typename T, uint32_t chCnt>
        inline void encodef32(void* _pix, const T* _input)
        {
            float* pix = reinterpret_cast<float*>(_pix);
            for (uint32_t i = 0u; i < chCnt; ++i)
                pix[i] = _input[i];
        }
    }
    template<>
    inline void encodePixels<ECF_R32_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef32<double, 1u>(_pix, _input);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void encodePixels<ECF_R32G32_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef32<double, 2u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R32G32B32_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef32<double, 3u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R32G32B32A32_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef32<double, 4u>(_pix, _input);
    }
    template<>
    inline void encodePixels<ECF_R32_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef32<uint64_t, 1u>(_pix, _input);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void encodePixels<ECF_R32G32_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef32<uint64_t, 2u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R32G32B32_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef32<uint64_t, 3u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R32G32B32A32_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef32<uint64_t, 4u>(_pix, _input);
    }
    template<>
    inline void encodePixels<ECF_R32_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef32<int64_t, 1u>(_pix, _input);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void encodePixels<ECF_R32G32_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef32<int64_t, 2u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R32G32B32_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef32<int64_t, 3u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R32G32B32A32_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef32<int64_t, 4u>(_pix, _input);
    }
    template<>
    inline void decodePixels<ECF_R32_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef32<double, 1u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32G32_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef32<double, 2u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef32<double, 3u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32A32_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef32<double, 4u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef32<uint64_t, 1u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32G32_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef32<uint64_t, 2u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef32<uint64_t, 3u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32A32_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef32<uint64_t, 4u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef32<int64_t, 1u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32G32_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef32<int64_t, 2u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef32<int64_t, 3u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R32G32B32A32_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef32<int64_t, 4u>(_pix, _output);
    }
    namespace impl
    {
        template<typename T, uint32_t chCnt>
        inline void decodef64(const void* _pix, T* _output)
        {
            const double* pix = reinterpret_cast<const double*>(_pix);
            for (uint32_t i = 0u; i < chCnt; ++i)
                _output[i] = pix[i];
        }
        template<typename T, uint32_t chCnt>
        inline void encodef64(void* _pix, const T* _input)
        {
            double* pix = reinterpret_cast<double*>(_pix);
            for (uint32_t i = 0u; i < chCnt; ++i)
                pix[i] = _input[i];
        }
    }
    template<>
    inline void encodePixels<ECF_R64_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef64<double, 1u>(_pix, _input);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void encodePixels<ECF_R64G64_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef64<double, 2u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R64G64B64_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef64<double, 3u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R64G64B64A64_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef64<double, 4u>(_pix, _input);
    }
    template<>
    inline void encodePixels<ECF_R64_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef64<uint64_t, 1u>(_pix, _input);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void encodePixels<ECF_R64G64_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef64<uint64_t, 2u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R64G64B64_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef64<uint64_t, 3u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R64G64B64A64_SFLOAT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        impl::encodef64<uint64_t, 4u>(_pix, _input);
    }
    template<>
    inline void encodePixels<ECF_R64_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef64<int64_t, 1u>(_pix, _input);
    }
    template<> // ECF_R16G16_SFLOAT gets mapped to GL_RG
    inline void encodePixels<ECF_R64G64_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef64<int64_t, 2u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R64G64B64_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef64<int64_t, 3u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<ECF_R64G64B64A64_SFLOAT, int64_t>(void* _pix, const int64_t* _input)
    {
        impl::encodef64<int64_t, 4u>(_pix, _input);
    }
    template<>
    inline void decodePixels<ECF_R64_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef64<double, 1u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64G64_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef64<double, 2u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef64<double, 3u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64A64_SFLOAT, double>(const void* _pix, double* _output)
    {
        impl::decodef64<double, 4u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef64<uint64_t, 1u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64G64_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef64<uint64_t, 2u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef64<uint64_t, 3u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64A64_SFLOAT, uint64_t>(const void* _pix, uint64_t* _output)
    {
        impl::decodef64<uint64_t, 4u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef64<int64_t, 1u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64G64_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef64<int64_t, 2u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef64<int64_t, 3u>(_pix, _output);
    }
    template<>
    inline void decodePixels<ECF_R64G64B64A64_SFLOAT, int64_t>(const void* _pix, int64_t* _output)
    {
        impl::decodef64<int64_t, 4u>(_pix, _output);
    }

    // Block Compression formats
    namespace impl
    {
        template<typename T>
        inline void decodeBC1(const void* _pix, T* _output, uint32_t _x, uint32_t _y, bool _alpha)
        {
#include "irr/irrpack.h"
            struct {
                uint16_t c0, c1;
                uint32_t lut;
            } PACK_STRUCT col;
#include "irr/irrunpack.h"
            memcpy(&col, _pix, 8u);

            struct {
                union {
                    uint64_t r, g, b, a;
                    uint64_t c[4];
                };
            } p[4];

            uint16_t r0, g0, b0, r1, g1, b1;
            

            decodePixels<ECF_R5G6B5, uint64_t>(&col.c0, p[0].c);
            r0 = p[0].r;
            g0 = p[0].g;
            b0 = p[0].b;
            decodePixels<ECF_R5G6B5, uint64_t>(&col.c1, p[1].c);
            r1 = p[1].r;
            g1 = p[1].g;
            b1 = p[1].b;
            if (col.c0 > col.c1)
            {
                p[2].r = (2 * r0 + 1 * r1) / 3;
                p[2].g = (2 * g0 + 1 * g1) / 3;
                p[2].b = (2 * b0 + 1 * b1) / 3;
                p[2].a = 0xff;
                p[3].r = (1 * r0 + 2 * r1) / 3;
                p[3].g = (1 * g0 + 2 * g1) / 3;
                p[3].b = (1 * b0 + 2 * b1) / 3;
                p[3].a = 0xff;
            }
            else
            {
                p[2].r = (r0 + r1) / 2;
                p[2].g = (g0 + g1) / 2;
                p[2].b = (b0 + b1) / 2;
                p[2].a = 0xff;
                p[3].r = 0;
                p[3].g = 0;
                p[3].b = 0;
                p[3].a = 0;
            }

            const uint32_t idx = 4u*_y + _x;
            const uint32_t cw = 3u & (col.lut >> (2u * idx));
            for (uint32_t i = 0u; i < (_alpha ? 4u : 3u); ++i)
                _output[i] = p[cw].c[i];
        }
        template<typename T>
        inline void decodeBC2(const void* _pix, T* _output, uint32_t _x, uint32_t _y)
        {
            const uint8_t* pix = reinterpret_cast<const uint8_t*>(_pix);
            decodeBC1(pix+8, _output, _x, _y, false);

            const uint32_t idx = 4u*_y + _x;
            const uint32_t bitI = idx * 4;
            const uint32_t byI = bitI / 8u;
            const uint32_t av = 0xfu & (pix[byI] >> (bitI & 7u));
            _output[3] = av;
        }
        template<typename T>
        inline void decodeBC4(const void* _pix, T* _output, int _offset, uint32_t _x, uint32_t _y)
        {
            struct
            {
                uint8_t a0, a1;
                uint8_t lut[6];
            } b;
            uint16_t a0, a1;
            uint8_t a[8];
            memcpy(&b, _pix, sizeof(b));

            a0 = b.a0;
            a1 = b.a1;
            a[0] = (uint8_t)a0;
            a[1] = (uint8_t)a1;
            if (a0 > a1)
            {
                a[2] = (6 * a0 + 1 * a1) / 7;
                a[3] = (5 * a0 + 2 * a1) / 7;
                a[4] = (4 * a0 + 3 * a1) / 7;
                a[5] = (3 * a0 + 4 * a1) / 7;
                a[6] = (2 * a0 + 5 * a1) / 7;
                a[7] = (1 * a0 + 6 * a1) / 7;
            }
            else
            {
                a[2] = (4 * a0 + 1 * a1) / 5;
                a[3] = (3 * a0 + 2 * a1) / 5;
                a[4] = (2 * a0 + 3 * a1) / 5;
                a[5] = (1 * a0 + 4 * a1) / 5;
                a[6] = 0;
                a[7] = 0xff;
            }

            const uint32_t idx = 4u*_y + _x;

            if (idx < 8u)
            {
                int lut = int(b.lut[0]) | int(b.lut[1] << 8) | int(b.lut[2] << 16);
                int aw = 7 & (lut >> (3 * idx));
                _output[_offset] = a[aw];
            }
            else
            {
                int lut = int(b.lut[3]) | int(b.lut[4] << 8) | int(b.lut[5] << 16);
                int aw = 7 & (lut >> (3 * idx));
                _output[_offset] = a[aw];
            }
        }

        template<typename T>
        inline void SRGB2lin(T _srgb[3]);

        template<>
        inline void SRGB2lin<double>(double _srgb[3])
        {
            for (uint32_t i = 0u; i < 3u; ++i)
            {
                double& s = _srgb[i];
                if (s <= 0.04045) s /= 12.92;
                else s = std::pow((s + 0.055) / 1.055, 2.4);
            }
        }

        template<typename T>// T is int64_t or uint64_t
        inline void SRGB2lin(T _srgb[3])
        {
            double s[3] { _srgb[0]/255., _srgb[1]/255., _srgb[2]/255. };
            SRGB2lin<double>(s);
            T* lin = _srgb;
            for (uint32_t i = 0; i < 3u; ++i)
                lin[i] = s * 255.;
        }
    }

    template<>
    inline void decodePixels<ECF_BC1_RGB_UNORM_BLOCK, double>(const void* _pix, double* _output, uint32_t _x, uint32_t _y)
    {
        impl::decodeBC1<double>(_pix, _output, _x, _y, false);
        _output[0] /= 31.;
        _output[1] /= 63.;
        _output[2] /= 31.;
    }

    template<>
    inline void decodePixels<ECF_BC1_RGB_SRGB_BLOCK, double>(const void* _pix, double* _output, uint32_t _x, uint32_t _y)
    {
        decodePixels<ECF_BC1_RGB_UNORM_BLOCK, double>(_pix, _output, _x, _y);
        impl::SRGB2lin(_output);
    }

    template<>
    inline void decodePixels<ECF_BC1_RGBA_UNORM_BLOCK, double>(const void* _pix, double* _output, uint32_t _x, uint32_t _y)
    {
        impl::decodeBC1<double>(_pix, _output, _x, _y, true);
        _output[0] /= 31.;
        _output[1] /= 63.;
        _output[2] /= 31.;
    }

    template<>
    inline void decodePixels<ECF_BC1_RGBA_SRGB_BLOCK, double>(const void* _pix, double* _output, uint32_t _x, uint32_t _y)
    {
        decodePixels<ECF_BC1_RGBA_UNORM_BLOCK, double>(_pix, _output, _x, _y);
        impl::SRGB2lin(_output);
    }

    template<>
    inline void decodePixels<ECF_BC2_UNORM_BLOCK, double>(const void* _pix, double* _output, uint32_t _x, uint32_t _y)
    {
        impl::decodeBC2<double>(_pix, _output, _x, _y);
        _output[0] /= 31.;
        _output[1] /= 63.;
        _output[2] /= 31.;
        _output[3] /= 15.;
    }

    template<>
    inline void decodePixels<ECF_BC2_SRGB_BLOCK, double>(const void* _pix, double* _output, uint32_t _x, uint32_t _y)
    {
        decodePixels<ECF_BC2_UNORM_BLOCK, double>(_pix, _output, _x, _y);
        impl::SRGB2lin(_output);
    }

    template<>
    inline void decodePixels<ECF_BC3_UNORM_BLOCK, double>(const void* _pix, double* _output, uint32_t _x, uint32_t _y)
    {
        decodePixels<ECF_BC1_RGBA_UNORM_BLOCK, double>(reinterpret_cast<const uint8_t*>(_pix)+8, _output);
        impl::decodeBC4(_pix, _output, 3, _x, _y);
        _output[3] /= 255.;
    }
    
    template<>
    inline void decodePixels<ECF_BC3_SRGB_BLOCK, double>(const void* _pix, double* _output, uint32_t _x, uint32_t _y)
    {
        decodePixels<ECF_BC3_UNORM_BLOCK, double>(_pix, _output);
        impl::SRGB2lin(_output);
    }


    template<typename T>
    bool decodePixels(ECOLOR_FORMAT _fmt, const void* _pix, T* _output);
    template<typename T>
    bool decodePixels(ECOLOR_FORMAT _fmt, const void* _pix, T* _output, uint64_t _scale);

    template<typename T>
    bool encodePixels(ECOLOR_FORMAT _fmt, void* _pix, const T* _input);
    template<typename T>
    bool encodePixels(ECOLOR_FORMAT _fmt, void* _pix, const T* _input, uint64_t _scale);


    template<>
    bool decodePixels<double>(ECOLOR_FORMAT _fmt, const void* _pix, double* _output)
    {
        switch (_fmt)
        {
        case ECF_R4G4_UNORM_PACK8: decodePixels<ECF_R4G4_UNORM_PACK8, double>(_pix, _output); return true;
        case ECF_R4G4B4A4_UNORM_PACK16: decodePixels<ECF_R4G4B4A4_UNORM_PACK16, double>(_pix, _output); return true;
        case ECF_B4G4R4A4_UNORM_PACK16: decodePixels<ECF_B4G4R4A4_UNORM_PACK16, double>(_pix, _output); return true;
        case ECF_R5G6B5_UNORM_PACK16: decodePixels<ECF_R5G6B5_UNORM_PACK16, double>(_pix, _output); return true;
        case ECF_B5G6R5_UNORM_PACK16: decodePixels<ECF_B5G6R5_UNORM_PACK16, double>(_pix, _output); return true;
        case ECF_R5G5B5A1_UNORM_PACK16: decodePixels<ECF_R5G5B5A1_UNORM_PACK16, double>(_pix, _output); return true;
        case ECF_B5G5R5A1_UNORM_PACK16: decodePixels<ECF_B5G5R5A1_UNORM_PACK16, double>(_pix, _output); return true;
        case ECF_A1R5G5B5_UNORM_PACK16: decodePixels<ECF_A1R5G5B5_UNORM_PACK16, double>(_pix, _output); return true;
        case ECF_R8_UNORM: decodePixels<ECF_R8_UNORM, double>(_pix, _output); return true;
        case ECF_R8_SNORM: decodePixels<ECF_R8_SNORM, double>(_pix, _output); return true;
        case ECF_R8G8_UNORM: decodePixels<ECF_R8G8_UNORM, double>(_pix, _output); return true;
        case ECF_R8G8_SNORM: decodePixels<ECF_R8G8_SNORM, double>(_pix, _output); return true;
        case ECF_R8G8B8_UNORM: decodePixels<ECF_R8G8B8_UNORM, double>(_pix, _output); return true;
        case ECF_R8G8B8_SNORM: decodePixels<ECF_R8G8B8_SNORM, double>(_pix, _output); return true;
        case ECF_B8G8R8_UNORM: decodePixels<ECF_B8G8R8_UNORM, double>(_pix, _output); return true;
        case ECF_B8G8R8_SNORM: decodePixels<ECF_B8G8R8_SNORM, double>(_pix, _output); return true;
        case ECF_R8G8B8A8_UNORM: decodePixels<ECF_R8G8B8A8_UNORM, double>(_pix, _output); return true;
        case ECF_R8G8B8A8_SNORM: decodePixels<ECF_R8G8B8A8_SNORM, double>(_pix, _output); return true;
        case ECF_B8G8R8A8_UNORM: decodePixels<ECF_B8G8R8A8_UNORM, double>(_pix, _output); return true;
        case ECF_B8G8R8A8_SNORM: decodePixels<ECF_B8G8R8A8_SNORM, double>(_pix, _output); return true;
        case ECF_A8B8G8R8_UNORM_PACK32: decodePixels<ECF_A8B8G8R8_UNORM_PACK32, double>(_pix, _output); return true;
        case ECF_A8B8G8R8_SNORM_PACK32: decodePixels<ECF_A8B8G8R8_SNORM_PACK32, double>(_pix, _output); return true;
        case ECF_A2R10G10B10_UNORM_PACK32: decodePixels<ECF_A2R10G10B10_UNORM_PACK32, double>(_pix, _output); return true;
        case ECF_A2R10G10B10_SNORM_PACK32: decodePixels<ECF_A2R10G10B10_SNORM_PACK32, double>(_pix, _output); return true;
        case ECF_A2B10G10R10_UNORM_PACK32: decodePixels<ECF_A2B10G10R10_UNORM_PACK32, double>(_pix, _output); return true;
        case ECF_A2B10G10R10_SNORM_PACK32: decodePixels<ECF_A2B10G10R10_SNORM_PACK32, double>(_pix, _output); return true;
        case ECF_R16_UNORM: decodePixels<ECF_R16_UNORM, double>(_pix, _output); return true;
        case ECF_R16_SNORM: decodePixels<ECF_R16_SNORM, double>(_pix, _output); return true;
        case ECF_R16G16_UNORM: decodePixels<ECF_R16G16_UNORM, double>(_pix, _output); return true;
        case ECF_R16G16_SNORM: decodePixels<ECF_R16G16_SNORM, double>(_pix, _output); return true;
        case ECF_R16G16B16_UNORM: decodePixels<ECF_R16G16B16_UNORM, double>(_pix, _output); return true;
        case ECF_R16G16B16_SNORM: decodePixels<ECF_R16G16B16_SNORM, double>(_pix, _output); return true;
        case ECF_R16G16B16A16_UNORM: decodePixels<ECF_R16G16B16A16_UNORM, double>(_pix, _output); return true;
        case ECF_R16G16B16A16_SNORM: decodePixels<ECF_R16G16B16A16_SNORM, double>(_pix, _output); return true;
        default: return false;
        }
    }
    template<>
    bool decodePixels<int64_t>(ECOLOR_FORMAT _fmt, const void* _pix, int64_t* _output)
    {
        switch (_fmt)
        {
        case ECF_R8_SINT: decodePixels<ECF_R8_SINT, int64_t>(_pix, _output); return true;
        case ECF_R8G8_SINT: decodePixels<ECF_R8G8_SINT, int64_t>(_pix, _output); return true;
        case ECF_R8G8B8_SINT: decodePixels<ECF_R8G8B8_SINT, int64_t>(_pix, _output); return true;
        case ECF_B8G8R8_SINT: decodePixels<ECF_B8G8R8_SINT, int64_t>(_pix, _output); return true;
        case ECF_R8G8B8A8_SINT: decodePixels<ECF_R8G8B8A8_SINT, int64_t>(_pix, _output); return true;
        case ECF_B8G8R8A8_SINT: decodePixels<ECF_B8G8R8A8_SINT, int64_t>(_pix, _output); return true;
        case ECF_A8B8G8R8_SINT_PACK32: decodePixels<ECF_A8B8G8R8_SINT_PACK32, int64_t>(_pix, _output); return true;
        case ECF_A2R10G10B10_SINT_PACK32: decodePixels<ECF_A2R10G10B10_SINT_PACK32, int64_t>(_pix, _output); return true;
        case ECF_A2B10G10R10_SINT_PACK32: decodePixels<ECF_A2B10G10R10_SINT_PACK32, int64_t>(_pix, _output); return true;
        case ECF_R16_SINT: decodePixels<ECF_R16_SINT, int64_t>(_pix, _output); return true;
        case ECF_R16G16_SINT: decodePixels<ECF_R16G16_SINT, int64_t>(_pix, _output); return true;
        case ECF_R16G16B16_SINT: decodePixels<ECF_R16G16B16_SINT, int64_t>(_pix, _output); return true;
        case ECF_R16G16B16A16_SINT: decodePixels<ECF_R16G16B16A16_SINT, int64_t>(_pix, _output); return true;
        case ECF_R32_SINT: decodePixels<ECF_R32_SINT, int64_t>(_pix, _output); return true;
        case ECF_R32G32_SINT: decodePixels<ECF_R32G32_SINT, int64_t>(_pix, _output); return true;
        case ECF_R32G32B32_SINT: decodePixels<ECF_R32G32B32_SINT, int64_t>(_pix, _output); return true;
        case ECF_R32G32B32A32_SINT: decodePixels<ECF_R32G32B32A32_SINT, int64_t>(_pix, _output); return true;
        case ECF_R64_SINT: decodePixels<ECF_R64_SINT, int64_t>(_pix, _output); return true;
        case ECF_R64G64_SINT: decodePixels<ECF_R64G64_SINT, int64_t>(_pix, _output); return true;
        case ECF_R64G64B64_SINT: decodePixels<ECF_R64G64B64_SINT, int64_t>(_pix, _output); return true;
        case ECF_R64G64B64A64_SINT: decodePixels<ECF_R64G64B64A64_SINT, int64_t>(_pix, _output); return true;
        default: return false;
        }
    }
    template<>
    bool decodePixels<uint64_t>(ECOLOR_FORMAT _fmt, const void* _pix, uint64_t* _output)
    {
        switch (_fmt)
        {
        case ECF_R8_UINT: decodePixels<ECF_R8_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R8_SRGB: decodePixels<ECF_R8_SRGB, uint64_t>(_pix, _output); return true;
        case ECF_R8G8_UINT: decodePixels<ECF_R8G8_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R8G8_SRGB: decodePixels<ECF_R8G8_SRGB, uint64_t>(_pix, _output); return true;
        case ECF_R8G8B8_UINT: decodePixels<ECF_R8G8B8_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R8G8B8_SRGB: decodePixels<ECF_R8G8B8_SRGB, uint64_t>(_pix, _output); return true;
        case ECF_B8G8R8_UINT: decodePixels<ECF_B8G8R8_UINT, uint64_t>(_pix, _output); return true;
        case ECF_B8G8R8_SRGB: decodePixels<ECF_B8G8R8_SRGB, uint64_t>(_pix, _output); return true;
        case ECF_R8G8B8A8_UINT: decodePixels<ECF_R8G8B8A8_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R8G8B8A8_SRGB: decodePixels<ECF_R8G8B8A8_SRGB, uint64_t>(_pix, _output); return true;
        case ECF_B8G8R8A8_UINT: decodePixels<ECF_B8G8R8A8_UINT, uint64_t>(_pix, _output); return true;
        case ECF_B8G8R8A8_SRGB: decodePixels<ECF_B8G8R8A8_SRGB, uint64_t>(_pix, _output); return true;
        case ECF_A8B8G8R8_UINT_PACK32: decodePixels<ECF_A8B8G8R8_UINT_PACK32, uint64_t>(_pix, _output); return true;
        case ECF_A8B8G8R8_SRGB_PACK32: decodePixels<ECF_A8B8G8R8_SRGB_PACK32, uint64_t>(_pix, _output); return true;
        case ECF_A2R10G10B10_UINT_PACK32: decodePixels<ECF_A2R10G10B10_UINT_PACK32, uint64_t>(_pix, _output); return true;
        case ECF_A2B10G10R10_UINT_PACK32: decodePixels<ECF_A2B10G10R10_UINT_PACK32, uint64_t>(_pix, _output); return true;
        case ECF_R16_UINT: decodePixels<ECF_R16_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R16G16_UINT: decodePixels<ECF_R16G16_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R16G16B16_UINT: decodePixels<ECF_R16G16B16_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R16G16B16A16_UINT: decodePixels<ECF_R16G16B16A16_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R32_UINT: decodePixels<ECF_R32_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R32G32_UINT: decodePixels<ECF_R32G32_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R32G32B32_UINT: decodePixels<ECF_R32G32B32_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R32G32B32A32_UINT: decodePixels<ECF_R32G32B32A32_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R64_UINT: decodePixels<ECF_R64_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R64G64_UINT: decodePixels<ECF_R64G64_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R64G64B64_UINT: decodePixels<ECF_R64G64B64_UINT, uint64_t>(_pix, _output); return true;
        case ECF_R64G64B64A64_UINT: decodePixels<ECF_R64G64B64A64_UINT, uint64_t>(_pix, _output); return true;
        default: return false;
        }
    }
    template<>
    bool decodePixels<double>(ECOLOR_FORMAT _fmt, const void* _pix, double* _output, uint64_t _scale)
    {
        switch (_fmt)
        {
        case ECF_R8_USCALED: decodePixels<ECF_R8_USCALED, double>(_pix, _output, _scale); return true;
        case ECF_R8_SSCALED: decodePixels<ECF_R8_SSCALED, double>(_pix, _output, _scale); return true;
        case ECF_R8G8_USCALED: decodePixels<ECF_R8G8_USCALED, double>(_pix, _output, _scale); return true;
        case ECF_R8G8_SSCALED: decodePixels<ECF_R8G8_SSCALED, double>(_pix, _output, _scale); return true;
        case ECF_R8G8B8_USCALED: decodePixels<ECF_R8G8B8_USCALED, double>(_pix, _output, _scale); return true;
        case ECF_R8G8B8_SSCALED: decodePixels<ECF_R8G8B8_SSCALED, double>(_pix, _output, _scale); return true;
        case ECF_B8G8R8_USCALED: decodePixels<ECF_B8G8R8_USCALED, double>(_pix, _output, _scale); return true;
        case ECF_B8G8R8_SSCALED: decodePixels<ECF_B8G8R8_SSCALED, double>(_pix, _output, _scale); return true;
        case ECF_R8G8B8A8_USCALED: decodePixels<ECF_R8G8B8A8_USCALED, double>(_pix, _output, _scale); return true;
        case ECF_R8G8B8A8_SSCALED: decodePixels<ECF_R8G8B8A8_SSCALED, double>(_pix, _output, _scale); return true;
        case ECF_B8G8R8A8_USCALED: decodePixels<ECF_B8G8R8A8_USCALED, double>(_pix, _output, _scale); return true;
        case ECF_B8G8R8A8_SSCALED: decodePixels<ECF_B8G8R8A8_SSCALED, double>(_pix, _output, _scale); return true;
        case ECF_A8B8G8R8_USCALED_PACK32: decodePixels<ECF_A8B8G8R8_USCALED_PACK32, double>(_pix, _output, _scale); return true;
        case ECF_A8B8G8R8_SSCALED_PACK32: decodePixels<ECF_A8B8G8R8_SSCALED_PACK32, double>(_pix, _output, _scale); return true;
        case ECF_A2R10G10B10_USCALED_PACK32: decodePixels<ECF_A2R10G10B10_USCALED_PACK32, double>(_pix, _output, _scale); return true;
        case ECF_A2R10G10B10_SSCALED_PACK32: decodePixels<ECF_A2R10G10B10_SSCALED_PACK32, double>(_pix, _output, _scale); return true;
        case ECF_A2B10G10R10_USCALED_PACK32: decodePixels<ECF_A2B10G10R10_USCALED_PACK32, double>(_pix, _output, _scale); return true;
        case ECF_A2B10G10R10_SSCALED_PACK32: decodePixels<ECF_A2B10G10R10_SSCALED_PACK32, double>(_pix, _output, _scale); return true;
        case ECF_R16_USCALED: decodePixels<ECF_R16_USCALED, double>(_pix, _output, _scale); return true;
        case ECF_R16_SSCALED: decodePixels<ECF_R16_SSCALED, double>(_pix, _output, _scale); return true;
        case ECF_R16G16_USCALED: decodePixels<ECF_R16G16_USCALED, double>(_pix, _output, _scale); return true;
        case ECF_R16G16_SSCALED: decodePixels<ECF_R16G16_SSCALED, double>(_pix, _output, _scale); return true;
        case ECF_R16G16B16_USCALED: decodePixels<ECF_R16G16B16_USCALED, double>(_pix, _output, _scale); return true;
        case ECF_R16G16B16_SSCALED: decodePixels<ECF_R16G16B16_SSCALED, double>(_pix, _output, _scale); return true;
        case ECF_R16G16B16A16_USCALED: decodePixels<ECF_R16G16B16A16_USCALED, double>(_pix, _output, _scale); return true;
        case ECF_R16G16B16A16_SSCALED: decodePixels<ECF_R16G16B16A16_SSCALED, double>(_pix, _output, _scale); return true;
        default: return false;
        }
    }
    template<>
    bool encodePixels<double>(ECOLOR_FORMAT _fmt, void* _pix, const double* _input)
    {
        switch (_fmt)
        {
        case ECF_R4G4_UNORM_PACK8: encodePixels<ECF_R4G4_UNORM_PACK8, double>(_pix, _input); return true;
        case ECF_R4G4B4A4_UNORM_PACK16: encodePixels<ECF_R4G4B4A4_UNORM_PACK16, double>(_pix, _input); return true;
        case ECF_B4G4R4A4_UNORM_PACK16: encodePixels<ECF_B4G4R4A4_UNORM_PACK16, double>(_pix, _input); return true;
        case ECF_R5G6B5_UNORM_PACK16: encodePixels<ECF_R5G6B5_UNORM_PACK16, double>(_pix, _input); return true;
        case ECF_B5G6R5_UNORM_PACK16: encodePixels<ECF_B5G6R5_UNORM_PACK16, double>(_pix, _input); return true;
        case ECF_R5G5B5A1_UNORM_PACK16: encodePixels<ECF_R5G5B5A1_UNORM_PACK16, double>(_pix, _input); return true;
        case ECF_B5G5R5A1_UNORM_PACK16: encodePixels<ECF_B5G5R5A1_UNORM_PACK16, double>(_pix, _input); return true;
        case ECF_A1R5G5B5_UNORM_PACK16: encodePixels<ECF_A1R5G5B5_UNORM_PACK16, double>(_pix, _input); return true;
        case ECF_R8_UNORM: encodePixels<ECF_R8_UNORM, double>(_pix, _input); return true;
        case ECF_R8_SNORM: encodePixels<ECF_R8_SNORM, double>(_pix, _input); return true;
        case ECF_R8G8_UNORM: encodePixels<ECF_R8G8_UNORM, double>(_pix, _input); return true;
        case ECF_R8G8_SNORM: encodePixels<ECF_R8G8_SNORM, double>(_pix, _input); return true;
        case ECF_R8G8B8_UNORM: encodePixels<ECF_R8G8B8_UNORM, double>(_pix, _input); return true;
        case ECF_R8G8B8_SNORM: encodePixels<ECF_R8G8B8_SNORM, double>(_pix, _input); return true;
        case ECF_B8G8R8_UNORM: encodePixels<ECF_B8G8R8_UNORM, double>(_pix, _input); return true;
        case ECF_B8G8R8_SNORM: encodePixels<ECF_B8G8R8_SNORM, double>(_pix, _input); return true;
        case ECF_R8G8B8A8_UNORM: encodePixels<ECF_R8G8B8A8_UNORM, double>(_pix, _input); return true;
        case ECF_R8G8B8A8_SNORM: encodePixels<ECF_R8G8B8A8_SNORM, double>(_pix, _input); return true;
        case ECF_B8G8R8A8_UNORM: encodePixels<ECF_B8G8R8A8_UNORM, double>(_pix, _input); return true;
        case ECF_B8G8R8A8_SNORM: encodePixels<ECF_B8G8R8A8_SNORM, double>(_pix, _input); return true;
        case ECF_A8B8G8R8_UNORM_PACK32: encodePixels<ECF_A8B8G8R8_UNORM_PACK32, double>(_pix, _input); return true;
        case ECF_A8B8G8R8_SNORM_PACK32: encodePixels<ECF_A8B8G8R8_SNORM_PACK32, double>(_pix, _input); return true;
        case ECF_A2R10G10B10_UNORM_PACK32: encodePixels<ECF_A2R10G10B10_UNORM_PACK32, double>(_pix, _input); return true;
        case ECF_A2R10G10B10_SNORM_PACK32: encodePixels<ECF_A2R10G10B10_SNORM_PACK32, double>(_pix, _input); return true;
        case ECF_A2B10G10R10_UNORM_PACK32: encodePixels<ECF_A2B10G10R10_UNORM_PACK32, double>(_pix, _input); return true;
        case ECF_A2B10G10R10_SNORM_PACK32: encodePixels<ECF_A2B10G10R10_SNORM_PACK32, double>(_pix, _input); return true;
        case ECF_R16_UNORM: encodePixels<ECF_R16_UNORM, double>(_pix, _input); return true;
        case ECF_R16_SNORM: encodePixels<ECF_R16_SNORM, double>(_pix, _input); return true;
        case ECF_R16G16_UNORM: encodePixels<ECF_R16G16_UNORM, double>(_pix, _input); return true;
        case ECF_R16G16_SNORM: encodePixels<ECF_R16G16_SNORM, double>(_pix, _input); return true;
        case ECF_R16G16B16_UNORM: encodePixels<ECF_R16G16B16_UNORM, double>(_pix, _input); return true;
        case ECF_R16G16B16_SNORM: encodePixels<ECF_R16G16B16_SNORM, double>(_pix, _input); return true;
        case ECF_R16G16B16A16_UNORM: encodePixels<ECF_R16G16B16A16_UNORM, double>(_pix, _input); return true;
        case ECF_R16G16B16A16_SNORM: encodePixels<ECF_R16G16B16A16_SNORM, double>(_pix, _input); return true;
        default: return false;
        }
    }
    template<>
    bool encodePixels<int64_t>(ECOLOR_FORMAT _fmt, void* _pix, const int64_t* _input)
    {
        switch (_fmt)
        {
        case ECF_R8_SINT: encodePixels<ECF_R8_SINT, int64_t>(_pix, _input); return true;
        case ECF_R8G8_SINT: encodePixels<ECF_R8G8_SINT, int64_t>(_pix, _input); return true;
        case ECF_R8G8B8_SINT: encodePixels<ECF_R8G8B8_SINT, int64_t>(_pix, _input); return true;
        case ECF_B8G8R8_SINT: encodePixels<ECF_B8G8R8_SINT, int64_t>(_pix, _input); return true;
        case ECF_R8G8B8A8_SINT: encodePixels<ECF_R8G8B8A8_SINT, int64_t>(_pix, _input); return true;
        case ECF_B8G8R8A8_SINT: encodePixels<ECF_B8G8R8A8_SINT, int64_t>(_pix, _input); return true;
        case ECF_A8B8G8R8_SINT_PACK32: encodePixels<ECF_A8B8G8R8_SINT_PACK32, int64_t>(_pix, _input); return true;
        case ECF_A2R10G10B10_SINT_PACK32: encodePixels<ECF_A2R10G10B10_SINT_PACK32, int64_t>(_pix, _input); return true;
        case ECF_A2B10G10R10_SINT_PACK32: encodePixels<ECF_A2B10G10R10_SINT_PACK32, int64_t>(_pix, _input); return true;
        case ECF_R16_SINT: encodePixels<ECF_R16_SINT, int64_t>(_pix, _input); return true;
        case ECF_R16G16_SINT: encodePixels<ECF_R16G16_SINT, int64_t>(_pix, _input); return true;
        case ECF_R16G16B16_SINT: encodePixels<ECF_R16G16B16_SINT, int64_t>(_pix, _input); return true;
        case ECF_R16G16B16A16_SINT: encodePixels<ECF_R16G16B16A16_SINT, int64_t>(_pix, _input); return true;
        case ECF_R32_SINT: encodePixels<ECF_R32_SINT, int64_t>(_pix, _input); return true;
        case ECF_R32G32_SINT: encodePixels<ECF_R32G32_SINT, int64_t>(_pix, _input); return true;
        case ECF_R32G32B32_SINT: encodePixels<ECF_R32G32B32_SINT, int64_t>(_pix, _input); return true;
        case ECF_R32G32B32A32_SINT: encodePixels<ECF_R32G32B32A32_SINT, int64_t>(_pix, _input); return true;
        case ECF_R64_SINT: encodePixels<ECF_R64_SINT, int64_t>(_pix, _input); return true;
        case ECF_R64G64_SINT: encodePixels<ECF_R64G64_SINT, int64_t>(_pix, _input); return true;
        case ECF_R64G64B64_SINT: encodePixels<ECF_R64G64B64_SINT, int64_t>(_pix, _input); return true;
        case ECF_R64G64B64A64_SINT: encodePixels<ECF_R64G64B64A64_SINT, int64_t>(_pix, _input); return true;
        default: return false;
        }
    }
    template<>
    bool encodePixels<uint64_t>(ECOLOR_FORMAT _fmt, void* _pix, const uint64_t* _input)
    {
        switch (_fmt)
        {
        case ECF_R8_UINT: encodePixels<ECF_R8_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R8_SRGB: encodePixels<ECF_R8_SRGB, uint64_t>(_pix, _input); return true;
        case ECF_R8G8_UINT: encodePixels<ECF_R8G8_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R8G8_SRGB: encodePixels<ECF_R8G8_SRGB, uint64_t>(_pix, _input); return true;
        case ECF_R8G8B8_UINT: encodePixels<ECF_R8G8B8_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R8G8B8_SRGB: encodePixels<ECF_R8G8B8_SRGB, uint64_t>(_pix, _input); return true;
        case ECF_B8G8R8_UINT: encodePixels<ECF_B8G8R8_UINT, uint64_t>(_pix, _input); return true;
        case ECF_B8G8R8_SRGB: encodePixels<ECF_B8G8R8_SRGB, uint64_t>(_pix, _input); return true;
        case ECF_R8G8B8A8_UINT: encodePixels<ECF_R8G8B8A8_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R8G8B8A8_SRGB: encodePixels<ECF_R8G8B8A8_SRGB, uint64_t>(_pix, _input); return true;
        case ECF_B8G8R8A8_UINT: encodePixels<ECF_B8G8R8A8_UINT, uint64_t>(_pix, _input); return true;
        case ECF_B8G8R8A8_SRGB: encodePixels<ECF_B8G8R8A8_SRGB, uint64_t>(_pix, _input); return true;
        case ECF_A8B8G8R8_UINT_PACK32: encodePixels<ECF_A8B8G8R8_UINT_PACK32, uint64_t>(_pix, _input); return true;
        case ECF_A8B8G8R8_SRGB_PACK32: encodePixels<ECF_A8B8G8R8_SRGB_PACK32, uint64_t>(_pix, _input); return true;
        case ECF_A2R10G10B10_UINT_PACK32: encodePixels<ECF_A2R10G10B10_UINT_PACK32, uint64_t>(_pix, _input); return true;
        case ECF_A2B10G10R10_UINT_PACK32: encodePixels<ECF_A2B10G10R10_UINT_PACK32, uint64_t>(_pix, _input); return true;
        case ECF_R16_UINT: encodePixels<ECF_R16_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R16G16_UINT: encodePixels<ECF_R16G16_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R16G16B16_UINT: encodePixels<ECF_R16G16B16_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R16G16B16A16_UINT: encodePixels<ECF_R16G16B16A16_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R32_UINT: encodePixels<ECF_R32_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R32G32_UINT: encodePixels<ECF_R32G32_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R32G32B32_UINT: encodePixels<ECF_R32G32B32_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R32G32B32A32_UINT: encodePixels<ECF_R32G32B32A32_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R64_UINT: encodePixels<ECF_R64_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R64G64_UINT: encodePixels<ECF_R64G64_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R64G64B64_UINT: encodePixels<ECF_R64G64B64_UINT, uint64_t>(_pix, _input); return true;
        case ECF_R64G64B64A64_UINT: encodePixels<ECF_R64G64B64A64_UINT, uint64_t>(_pix, _input); return true;
        default: return false;
        }
    }
    template<>
    bool encodePixels<double>(ECOLOR_FORMAT _fmt, void* _pix, const double* _input, uint64_t _scale)
    {
        switch (_fmt)
        {
        case ECF_R8_USCALED: encodePixels<ECF_R8_USCALED, double>(_pix, _input, _scale); return true;
        case ECF_R8_SSCALED: encodePixels<ECF_R8_SSCALED, double>(_pix, _input, _scale); return true;
        case ECF_R8G8_USCALED: encodePixels<ECF_R8G8_USCALED, double>(_pix, _input, _scale); return true;
        case ECF_R8G8_SSCALED: encodePixels<ECF_R8G8_SSCALED, double>(_pix, _input, _scale); return true;
        case ECF_R8G8B8_USCALED: encodePixels<ECF_R8G8B8_USCALED, double>(_pix, _input, _scale); return true;
        case ECF_R8G8B8_SSCALED: encodePixels<ECF_R8G8B8_SSCALED, double>(_pix, _input, _scale); return true;
        case ECF_B8G8R8_USCALED: encodePixels<ECF_B8G8R8_USCALED, double>(_pix, _input, _scale); return true;
        case ECF_B8G8R8_SSCALED: encodePixels<ECF_B8G8R8_SSCALED, double>(_pix, _input, _scale); return true;
        case ECF_R8G8B8A8_USCALED: encodePixels<ECF_R8G8B8A8_USCALED, double>(_pix, _input, _scale); return true;
        case ECF_R8G8B8A8_SSCALED: encodePixels<ECF_R8G8B8A8_SSCALED, double>(_pix, _input, _scale); return true;
        case ECF_B8G8R8A8_USCALED: encodePixels<ECF_B8G8R8A8_USCALED, double>(_pix, _input, _scale); return true;
        case ECF_B8G8R8A8_SSCALED: encodePixels<ECF_B8G8R8A8_SSCALED, double>(_pix, _input, _scale); return true;
        case ECF_A8B8G8R8_USCALED_PACK32: encodePixels<ECF_A8B8G8R8_USCALED_PACK32, double>(_pix, _input, _scale); return true;
        case ECF_A8B8G8R8_SSCALED_PACK32: encodePixels<ECF_A8B8G8R8_SSCALED_PACK32, double>(_pix, _input, _scale); return true;
        case ECF_A2R10G10B10_USCALED_PACK32: encodePixels<ECF_A2R10G10B10_USCALED_PACK32, double>(_pix, _input, _scale); return true;
        case ECF_A2R10G10B10_SSCALED_PACK32: encodePixels<ECF_A2R10G10B10_SSCALED_PACK32, double>(_pix, _input, _scale); return true;
        case ECF_A2B10G10R10_USCALED_PACK32: encodePixels<ECF_A2B10G10R10_USCALED_PACK32, double>(_pix, _input, _scale); return true;
        case ECF_A2B10G10R10_SSCALED_PACK32: encodePixels<ECF_A2B10G10R10_SSCALED_PACK32, double>(_pix, _input, _scale); return true;
        case ECF_R16_USCALED: encodePixels<ECF_R16_USCALED, double>(_pix, _input, _scale); return true;
        case ECF_R16_SSCALED: encodePixels<ECF_R16_SSCALED, double>(_pix, _input, _scale); return true;
        case ECF_R16G16_USCALED: encodePixels<ECF_R16G16_USCALED, double>(_pix, _input, _scale); return true;
        case ECF_R16G16_SSCALED: encodePixels<ECF_R16G16_SSCALED, double>(_pix, _input, _scale); return true;
        case ECF_R16G16B16_USCALED: encodePixels<ECF_R16G16B16_USCALED, double>(_pix, _input, _scale); return true;
        case ECF_R16G16B16_SSCALED: encodePixels<ECF_R16G16B16_SSCALED, double>(_pix, _input, _scale); return true;
        case ECF_R16G16B16A16_USCALED: encodePixels<ECF_R16G16B16A16_USCALED, double>(_pix, _input, _scale); return true;
        case ECF_R16G16B16A16_SSCALED: encodePixels<ECF_R16G16B16A16_SSCALED, double>(_pix, _input, _scale); return true;
        default: return false;
        }
    }


    namespace impl
    {
        template<bool SCALED, ECOLOR_FORMAT cf, typename T>
        struct SCallDecode;

        template<ECOLOR_FORMAT cf, typename T>
        struct SCallDecode<false, cf, T>
        {
            inline void operator()(const void* _pix, T* _output, uint64_t)
            {
                decodePixels<cf, T>(_pix, _output);
            }
        };
        template<ECOLOR_FORMAT cf, typename T>
        struct SCallDecode<true, cf, T>
        {
            inline void operator()(const void* _pix, T* _output, uint64_t _scale)
            {
                decodePixels<cf, T>(_pix, _output, _scale);
            }
        };

        template<bool SCALED, ECOLOR_FORMAT cf, typename T>
        struct SCallEncode;

        template<ECOLOR_FORMAT cf, typename T>
        struct SCallEncode<false, cf, T>
        {
            inline void operator()(void* _pix, const T* _output, uint64_t)
            {
                encodePixels<cf, T>(_pix, _output);
            }
        };
        template<ECOLOR_FORMAT cf, typename T>
        struct SCallEncode<true, cf, T>
        {
            inline void operator()(void* _pix, const T* _output, uint64_t _scale)
            {
                encodePixels<cf, T>(_pix, _output, _scale);
            }
        };
    } //namespace impl


    template<ECOLOR_FORMAT sF, ECOLOR_FORMAT dF>
    inline void convertColor(const void* srcPix, void* dstPix, uint64_t _scale)
    {
        if (isIntegerFormat<sF>() && isIntegerFormat<dF>())
        {
            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;

            decT decbuf[4];
            //decodePixels<sF, decT>(srcPix, decbuf, _scale);
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _scale);
            //encodePixels<dF, encT>(dstPix, reinterpret_cast<encT*>(decbuf), _scale);
            impl::SCallEncode<isScaledFormat<dF>(), dF, encT>{}(dstPix, reinterpret_cast<encT*>(decbuf), _scale);
        }
        else if (
            (isNormalizedFormat<sF>() && isNormalizedFormat<dF>()) ||
            (isFloatingPointFormat<sF>() && isFloatingPointFormat<dF>()) ||
            (isNormalizedFormat<sF>() && isFloatingPointFormat<dF>()) ||
            (isFloatingPointFormat<sF>() && isNormalizedFormat<dF>())
        )
        {
            using decT = double;
            using encT = double;

            decT decbuf[4];
            //decodePixels<sF, decT>(srcPix, decbuf, _scale);
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _scale);
            //encodePixels<dF, encT>(dstPix, decbuf, _scale);
            impl::SCallEncode<isScaledFormat<dF>(), dF, encT>{}(dstPix, decbuf, _scale);
        }
        else if ((isFloatingPointFormat<sF>() || isNormalizedFormat<sF>()) && isIntegerFormat<dF>())
        {
            using decT = double;
            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;

            decT decbuf[4];
            //decodePixels<sF, decT>(srcPix, decbuf, _scale);
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _scale);
            encT encbuf[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                encbuf[i] = decbuf[i];
            //encodePixels<dF, encT>(dstPix, encbuf, _scale);
            impl::SCallEncode<isScaledFormat<dF>(), dF, encT>{}(dstPix, encbuf, _scale);
        }
        else if (isIntegerFormat<sF>() && (isNormalizedFormat<dF>() || isFloatingPointFormat<dF>()))
        {
            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
            using encT = double;

            decT decbuf[4];
            //decodePixels<sF, decT>(srcPix, decbuf, _scale);
            impl::SCallDecode<isScaledFormat<sF>(), sF, decT>{}(srcPix, decbuf, _scale);
            encT encbuf[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                encbuf[i] = decbuf[i];
            //encodePixels<dF, encT>(dstPix, encbuf, _scale);
            impl::SCallEncode<isScaledFormat<dF>(), dF, encT>{}(dstPix, encbuf, _scale);
        }
    }
    template<ECOLOR_FORMAT sF, ECOLOR_FORMAT dF>
    inline void convertColor(const void* srcPix, void* dstPix, uint64_t _scale, size_t _pixCnt)
    {
        const uint32_t srcStride = getTexelOrBlockSize(sF);
        const uint32_t dstStride = getTexelOrBlockSize(dF);

        const uint8_t* src = reinterpret_cast<const uint8_t*>(srcPix);
        uint8_t* dst = reinterpret_cast<uint8_t*>(dstPix);
        for (size_t i = 0u; i < _pixCnt; ++i)
        {
            convertColor<sF, dF>(src, dst, _scale);
            src += srcStride;
            dst += dstStride;
        }
    }

//    inline void convertColor(ECOLOR_FORMAT _sfmt, ECOLOR_FORMAT _dfmt, const void* srcPix, void* dstPix, uint64_t _scale)
//    {
//        if (isIntegerFormat(_sfmt) && isIntegerFormat(_dfmt))
//        {
//#define CONVERSION_CODE(COND1, COND2, STYPE, DTYPE)\
//if (COND1 && COND2)\
//{\
//    STYPE decbuf[4];\
//    decodePixels<STYPE>(_sfmt, srcPix, decbuf, _scale);
//}
//            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
//            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;
//
//            decT decbuf[4];
//            decodePixels<decT>(_sfmt, srcPix, decbuf, _scale);
//            encodePixels<encT>(_dfmt, dstPix, reinterpret_cast<encT*>(decbuf), _scale);
//        }
//        else if (
//            (isNormalizedFormat<sF>() && isNormalizedFormat<dF>()) ||
//            (isFloatingPointFormat<sF>() && isFloatingPointFormat<dF>()) ||
//            (isNormalizedFormat<sF>() && isFloatingPointFormat<dF>()) ||
//            (isFloatingPointFormat<sF>() && isNormalizedFormat<dF>())
//            )
//        {
//            using decT = double;
//            using encT = double;
//
//            decT decbuf[4];
//            decodePixels<sF, decT>(srcPix, decbuf, _scale);
//            encodePixels<dF, encT>(dstPix, decbuf, _scale);
//        }
//        else if ((isFloatingPointFormat<sF>() || isNormalizedFormat<sF>()) && isIntegerFormat<dF>())
//        {
//            using decT = double;
//            using encT = typename std::conditional<isSignedFormat<dF>(), int64_t, uint64_t>::type;
//
//            decT decbuf[4];
//            decodePixels<sF, decT>(srcPix, decbuf, _scale);
//            encT encbuf[4];
//            for (uint32_t i = 0u; i < 4u; ++i)
//                encbuf[i] = decbuf[i];
//            encodePixels<dF, encT>(dstPix, encbuf, _scale);
//        }
//        else if (isIntegerFormat<sF>() && (isNormalizedFormat<dF>() || isFloatingPointFormat<dF>()))
//        {
//            using decT = typename std::conditional<isSignedFormat<sF>(), int64_t, uint64_t>::type;
//            using encT = double;
//
//            decT decbuf[4];
//            decodePixels<sF, decT>(srcPix, decbuf, _scale);
//            encT encbuf[4];
//            for (uint32_t i = 0u; i < 4u; ++i)
//                encbuf[i] = decbuf[i];
//            encodePixels<dF, encT>(dstPix, encbuf, _scale);
//        }
//    }

	//! get the amount of Bits per Pixel of the given color format
	static uint32_t getBitsPerPixelFromFormat(const ECOLOR_FORMAT format)
	{
		switch(format)
		{
		case ECF_A1R5G5B5:
			return 16;
		case ECF_R8G8B8_UINT:
			return 24;
		case ECF_R8G8B8A8_UINT:
			return 32;
        case ECF_B10G11R11_UFLOAT_PACK32:
            return 32;
		case ECF_R16_SFLOAT:
			return 16;
		case ECF_R16G16_SFLOAT:
			return 32;
		case ECF_R16G16B16A16_SFLOAT:
			return 64;
		case ECF_R32_SFLOAT:
			return 32;
		case ECF_R32G32_SFLOAT:
			return 64;
		case ECF_R32G32B32A32_SFLOAT:
			return 128;
		case ECF_R8_UINT:
			return 8;
		case ECF_R8G8_UINT:
			return 16;
        case ECF_BC1_RGB_UNORM_BLOCK:
        case ECF_BC1_RGBA_UNORM_BLOCK:
            return 4;
        case ECF_BC2_UNORM_BLOCK:
        case ECF_BC3_UNORM_BLOCK:
            return 8;
        case ECF_8BIT_PIX:
			return 8;
		case ECF_16BIT_PIX:
			return 16;
		case ECF_24BIT_PIX:
			return 24;
		case ECF_32BIT_PIX:
			return 32;
		case ECF_48BIT_PIX: // rgb @ 16bit
            return 48;
        case ECF_64BIT_PIX:
			return 64;
		case ECF_96BIT_PIX:
			return 96;
		case ECF_128BIT_PIX:
            return 128;
        case ECF_DEPTH16:
            return 16;
        case ECF_DEPTH24:
            return 24;
        case ECF_DEPTH32F:
        case ECF_DEPTH24_STENCIL8:
            return 32;
        case ECF_DEPTH32F_STENCIL8:
            return 40;
        case ECF_STENCIL8:
            return 8;
        case ECF_E5B9G9R9_UFLOAT_PACK32:
            return 32;
		default:
			return 0;
		}
	}   

	//! get
	static uint32_t getCompressedFormatBlockSize(const ECOLOR_FORMAT format)
	{
		switch(format)
		{
            case ECF_BC1_RGB_UNORM_BLOCK:
            case ECF_BC1_RGBA_UNORM_BLOCK:
            case ECF_BC2_UNORM_BLOCK:
            case ECF_BC3_UNORM_BLOCK:
                return 4;
            default:
                return 1;
		}
	}

    //todo remove this
	static bool isFormatCompressed(const ECOLOR_FORMAT format)
	{
	    return getCompressedFormatBlockSize(format)!=1;
	}

	//! Creates a 16 bit A1R5G5B5 color
	inline uint16_t RGBA16(uint32_t r, uint32_t g, uint32_t b, uint32_t a=0xFF)
	{
		return (uint16_t)((a & 0x80) << 8 |
			(r & 0xF8) << 7 |
			(g & 0xF8) << 2 |
			(b & 0xF8) >> 3);
	}


	//! Creates a 16 bit A1R5G5B5 color
	inline uint16_t RGB16(uint32_t r, uint32_t g, uint32_t b)
	{
		return RGBA16(r,g,b);
	}


	//! Creates a 16bit A1R5G5B5 color, based on 16bit input values
	inline uint16_t RGB16from16(uint16_t r, uint16_t g, uint16_t b)
	{
		return (0x8000 |
				(r & 0x1F) << 10 |
				(g & 0x1F) << 5  |
				(b & 0x1F));
	}


	//! Converts a 32bit (X8R8G8B8) color to a 16bit A1R5G5B5 color
	inline uint16_t X8R8G8B8toA1R5G5B5(uint32_t color)
	{
		return (uint16_t)(0x8000 |
			( color & 0x00F80000) >> 9 |
			( color & 0x0000F800) >> 6 |
			( color & 0x000000F8) >> 3);
	}


	//! Converts a 32bit (A8R8G8B8) color to a 16bit A1R5G5B5 color
	inline uint16_t A8R8G8B8toA1R5G5B5(uint32_t color)
	{
		return (uint16_t)(( color & 0x80000000) >> 16|
			( color & 0x00F80000) >> 9 |
			( color & 0x0000F800) >> 6 |
			( color & 0x000000F8) >> 3);
	}


	//! Converts a 32bit (A8R8G8B8) color to a 16bit R5G6B5 color
	inline uint16_t A8R8G8B8toR5G6B5(uint32_t color)
	{
		return (uint16_t)(( color & 0x00F80000) >> 8 |
			( color & 0x0000FC00) >> 5 |
			( color & 0x000000F8) >> 3);
	}


	//! Convert A8R8G8B8 Color from A1R5G5B5 color
	/** build a nicer 32bit Color by extending dest lower bits with source high bits. */
	inline uint32_t A1R5G5B5toA8R8G8B8(uint16_t color)
	{
		return ( (( -( (int32_t) color & 0x00008000 ) >> (int32_t) 31 ) & 0xFF000000 ) |
				(( color & 0x00007C00 ) << 9) | (( color & 0x00007000 ) << 4) |
				(( color & 0x000003E0 ) << 6) | (( color & 0x00000380 ) << 1) |
				(( color & 0x0000001F ) << 3) | (( color & 0x0000001C ) >> 2)
				);
	}


	//! Returns A8R8G8B8 Color from R5G6B5 color
	inline uint32_t R5G6B5toA8R8G8B8(uint16_t color)
	{
		return 0xFF000000 |
			((color & 0xF800) << 8)|
			((color & 0x07E0) << 5)|
			((color & 0x001F) << 3);
	}


	//! Returns A1R5G5B5 Color from R5G6B5 color
	inline uint16_t R5G6B5toA1R5G5B5(uint16_t color)
	{
		return 0x8000 | (((color & 0xFFC0) >> 1) | (color & 0x1F));
	}


	//! Returns R5G6B5 Color from A1R5G5B5 color
	inline uint16_t A1R5G5B5toR5G6B5(uint16_t color)
	{
		return (((color & 0x7FE0) << 1) | (color & 0x1F));
	}



	//! Returns the alpha component from A1R5G5B5 color
	/** In Irrlicht, alpha refers to opacity.
	\return The alpha value of the color. 0 is transparent, 1 is opaque. */
	inline uint32_t getAlpha(uint16_t color)
	{
		return ((color >> 15)&0x1);
	}


	//! Returns the red component from A1R5G5B5 color.
	/** Shift left by 3 to get 8 bit value. */
	inline uint32_t getRed(uint16_t color)
	{
		return ((color >> 10)&0x1F);
	}


	//! Returns the green component from A1R5G5B5 color
	/** Shift left by 3 to get 8 bit value. */
	inline uint32_t getGreen(uint16_t color)
	{
		return ((color >> 5)&0x1F);
	}


	//! Returns the blue component from A1R5G5B5 color
	/** Shift left by 3 to get 8 bit value. */
	inline uint32_t getBlue(uint16_t color)
	{
		return (color & 0x1F);
	}


	//! Returns the average from a 16 bit A1R5G5B5 color
	inline int32_t getAverage(int16_t color)
	{
		return ((getRed(color)<<3) + (getGreen(color)<<3) + (getBlue(color)<<3)) / 3;
	}


	//! Class representing a 32 bit ARGB color.
	/** The color values for alpha, red, green, and blue are
	stored in a single uint32_t. So all four values may be between 0 and 255.
	Alpha in Irrlicht is opacity, so 0 is fully transparent, 255 is fully opaque (solid).
	This class is used by most parts of the Irrlicht Engine
	to specify a color. Another way is using the class SColorf, which
	stores the color values in 4 floats.
	This class must consist of only one uint32_t and must not use virtual functions.
	*/
	class SColor
	{
	public:

		//! Constructor of the Color. Does nothing.
		/** The color value is not initialized to save time. */
		SColor() {}

		//! Constructs the color from 4 values representing the alpha, red, green and blue component.
		/** Must be values between 0 and 255. */
		SColor (uint32_t a, uint32_t r, uint32_t g, uint32_t b)
			: color(((a & 0xff)<<24) | ((r & 0xff)<<16) | ((g & 0xff)<<8) | (b & 0xff)) {}

		//! Constructs the color from a 32 bit value. Could be another color.
		SColor(uint32_t clr)
			: color(clr) {}

		//! Returns the alpha component of the color.
		/** The alpha component defines how opaque a color is.
		\return The alpha value of the color. 0 is fully transparent, 255 is fully opaque. */
		uint32_t getAlpha() const { return color>>24; }

		//! Returns the red component of the color.
		/** \return Value between 0 and 255, specifying how red the color is.
		0 means no red, 255 means full red. */
		uint32_t getRed() const { return (color>>16) & 0xff; }

		//! Returns the green component of the color.
		/** \return Value between 0 and 255, specifying how green the color is.
		0 means no green, 255 means full green. */
		uint32_t getGreen() const { return (color>>8) & 0xff; }

		//! Returns the blue component of the color.
		/** \return Value between 0 and 255, specifying how blue the color is.
		0 means no blue, 255 means full blue. */
		uint32_t getBlue() const { return color & 0xff; }

		//! Get lightness of the color in the range [0,255]
		float getLightness() const
		{
			return 0.5f*(core::max_(core::max_(getRed(),getGreen()),getBlue())+core::min_(core::min_(getRed(),getGreen()),getBlue()));
		}

		//! Get luminance of the color in the range [0,255].
		float getLuminance() const
		{
			return 0.3f*getRed() + 0.59f*getGreen() + 0.11f*getBlue();
		}

		//! Get average intensity of the color in the range [0,255].
		uint32_t getAverage() const
		{
			return ( getRed() + getGreen() + getBlue() ) / 3;
		}

		//! Sets the alpha component of the Color.
		/** The alpha component defines how transparent a color should be.
		\param a The alpha value of the color. 0 is fully transparent, 255 is fully opaque. */
		void setAlpha(uint32_t a) { color = ((a & 0xff)<<24) | (color & 0x00ffffff); }

		//! Sets the red component of the Color.
		/** \param r: Has to be a value between 0 and 255.
		0 means no red, 255 means full red. */
		void setRed(uint32_t r) { color = ((r & 0xff)<<16) | (color & 0xff00ffff); }

		//! Sets the green component of the Color.
		/** \param g: Has to be a value between 0 and 255.
		0 means no green, 255 means full green. */
		void setGreen(uint32_t g) { color = ((g & 0xff)<<8) | (color & 0xffff00ff); }

		//! Sets the blue component of the Color.
		/** \param b: Has to be a value between 0 and 255.
		0 means no blue, 255 means full blue. */
		void setBlue(uint32_t b) { color = (b & 0xff) | (color & 0xffffff00); }

		//! Calculates a 16 bit A1R5G5B5 value of this color.
		/** \return 16 bit A1R5G5B5 value of this color. */
		uint16_t toA1R5G5B5() const { return A8R8G8B8toA1R5G5B5(color); }

		//! Converts color to OpenGL color format
		/** From ARGB to RGBA in 4 byte components for endian aware
		passing to OpenGL
		\param dest: address where the 4x8 bit OpenGL color is stored. */
		void toOpenGLColor(uint8_t* dest) const
		{
			*dest =   (uint8_t)getRed();
			*++dest = (uint8_t)getGreen();
			*++dest = (uint8_t)getBlue();
			*++dest = (uint8_t)getAlpha();
		}

		//! Sets all four components of the color at once.
		/** Constructs the color from 4 values representing the alpha,
		red, green and blue components of the color. Must be values
		between 0 and 255.
		\param a: Alpha component of the color. The alpha component
		defines how transparent a color should be. Has to be a value
		between 0 and 255. 255 means not transparent (opaque), 0 means
		fully transparent.
		\param r: Sets the red component of the Color. Has to be a
		value between 0 and 255. 0 means no red, 255 means full red.
		\param g: Sets the green component of the Color. Has to be a
		value between 0 and 255. 0 means no green, 255 means full
		green.
		\param b: Sets the blue component of the Color. Has to be a
		value between 0 and 255. 0 means no blue, 255 means full blue. */
		void set(uint32_t a, uint32_t r, uint32_t g, uint32_t b)
		{
			color = (((a & 0xff)<<24) | ((r & 0xff)<<16) | ((g & 0xff)<<8) | (b & 0xff));
		}
		void set(uint32_t col) { color = col; }

		//! Compares the color to another color.
		/** \return True if the colors are the same, and false if not. */
		bool operator==(const SColor& other) const { return other.color == color; }

		//! Compares the color to another color.
		/** \return True if the colors are different, and false if they are the same. */
		bool operator!=(const SColor& other) const { return other.color != color; }

		//! comparison operator
		/** \return True if this color is smaller than the other one */
		bool operator<(const SColor& other) const { return (color < other.color); }

		//! Adds two colors, result is clamped to 0..255 values
		/** \param other Color to add to this color
		\return Addition of the two colors, clamped to 0..255 values */
		SColor operator+(const SColor& other) const
		{
			return SColor(core::min_(getAlpha() + other.getAlpha(), 255u),
					core::min_(getRed() + other.getRed(), 255u),
					core::min_(getGreen() + other.getGreen(), 255u),
					core::min_(getBlue() + other.getBlue(), 255u));
		}

		//! Interpolates the color with a float value to another color
		/** \param other: Other color
		\param d: value between 0.0f and 1.0f
		\return Interpolated color. */
		SColor getInterpolated(const SColor &other, float d) const
		{
			d = core::clamp(d, 0.f, 1.f);
			const float inv = 1.0f - d;
			return SColor((uint32_t)core::round32(other.getAlpha()*inv + getAlpha()*d),
				(uint32_t)core::round32(other.getRed()*inv + getRed()*d),
				(uint32_t)core::round32(other.getGreen()*inv + getGreen()*d),
				(uint32_t)core::round32(other.getBlue()*inv + getBlue()*d));
		}

		//! Returns interpolated color. ( quadratic )
		/** \param c1: first color to interpolate with
		\param c2: second color to interpolate with
		\param d: value between 0.0f and 1.0f. */
		SColor getInterpolated_quadratic(const SColor& c1, const SColor& c2, float d) const
		{
			// this*(1-d)*(1-d) + 2 * c1 * (1-d) + c2 * d * d;
			d = core::clamp(d, 0.f, 1.f);
			const float inv = 1.f - d;
			const float mul0 = inv * inv;
			const float mul1 = 2.f * d * inv;
			const float mul2 = d * d;

			return SColor(
					core::clamp( core::floor32(
							getAlpha() * mul0 + c1.getAlpha() * mul1 + c2.getAlpha() * mul2 ), 0, 255 ),
					core::clamp( core::floor32(
							getRed()   * mul0 + c1.getRed()   * mul1 + c2.getRed()   * mul2 ), 0, 255 ),
					core::clamp ( core::floor32(
							getGreen() * mul0 + c1.getGreen() * mul1 + c2.getGreen() * mul2 ), 0, 255 ),
					core::clamp ( core::floor32(
							getBlue()  * mul0 + c1.getBlue()  * mul1 + c2.getBlue()  * mul2 ), 0, 255 ));
		}

		//! set the color by expecting data in the given format
		/** \param data: must point to valid memory containing color information in the given format
			\param format: tells the format in which data is available
		*/
		void setData(const void *data, ECOLOR_FORMAT format)
		{
			switch (format)
			{
				case ECF_A1R5G5B5:
					color = A1R5G5B5toA8R8G8B8(*(uint16_t*)data);
					break;
				case ECF_R5G6B5:
					color = R5G6B5toA8R8G8B8(*(uint16_t*)data);
					break;
				case ECF_R8G8B8A8_UINT:
					color = *(uint32_t*)data;
					break;
				case ECF_R8G8B8_UINT:
					{
						uint8_t* p = (uint8_t*)data;
						set(255, p[0],p[1],p[2]);
					}
					break;
				default:
					color = 0xffffffff;
				break;
			}
		}

		//! Write the color to data in the defined format
		/** \param data: target to write the color. Must contain sufficiently large memory to receive the number of bytes neede for format
			\param format: tells the format used to write the color into data
		*/
		void getData(void *data, ECOLOR_FORMAT format)
		{
			switch(format)
			{
				case ECF_A1R5G5B5:
				{
					uint16_t * dest = (uint16_t*)data;
					*dest = video::A8R8G8B8toA1R5G5B5( color );
				}
				break;

				case ECF_R5G6B5:
				{
					uint16_t * dest = (uint16_t*)data;
					*dest = video::A8R8G8B8toR5G6B5( color );
				}
				break;

				case ECF_R8G8B8_UINT:
				{
					uint8_t* dest = (uint8_t*)data;
					dest[0] = (uint8_t)getRed();
					dest[1] = (uint8_t)getGreen();
					dest[2] = (uint8_t)getBlue();
				}
				break;

				case ECF_R8G8B8A8_UINT:
				{
					uint32_t * dest = (uint32_t*)data;
					*dest = color;
				}
				break;

				default:
				break;
			}
		}

		//! color in A8R8G8B8 Format
		uint32_t color;
	};


	//! Class representing a color with four floats.
	/** The color values for red, green, blue
	and alpha are each stored in a 32 bit floating point variable.
	So all four values may be between 0.0f and 1.0f.
	Another, faster way to define colors is using the class SColor, which
	stores the color values in a single 32 bit integer.
	*/
	class SColorf : private core::vectorSIMDf
	{
	public:
		//! Default constructor for SColorf.
		/** Sets red, green and blue to 0.0f and alpha to 1.0f. */
		SColorf() : vectorSIMDf(0.f,0.f,0.f,1.f) {}

		//! Constructs a color from up to four color values: red, green, blue, and alpha.
		/** \param r: Red color component. Should be a value between
		0.0f meaning no red and 1.0f, meaning full red.
		\param g: Green color component. Should be a value between 0.0f
		meaning no green and 1.0f, meaning full green.
		\param b: Blue color component. Should be a value between 0.0f
		meaning no blue and 1.0f, meaning full blue.
		\param a: Alpha color component of the color. The alpha
		component defines how transparent a color should be. Has to be
		a value between 0.0f and 1.0f, 1.0f means not transparent
		(opaque), 0.0f means fully transparent. */
		SColorf(float r_in, float g_in, float b_in, float a_in = 1.0f) : vectorSIMDf(r_in,g_in,b_in,a_in) {}

		SColorf(const vectorSIMDf& fromVec4) : vectorSIMDf(fromVec4) {}

		//! Constructs a color from 32 bit Color.
		/** \param c: 32 bit color from which this SColorf class is
		constructed from. */
		SColorf(SColor c)
		{
			r = c.getRed();
			g = c.getGreen();
			b = c.getBlue();
			a = c.getAlpha();

			const float inv = 1.0f / 255.0f;
			*this *= inv;
		}

		//! Converts this color to a SColor without floats.
		inline SColor toSColor() const
		{
		    vectorSIMDf tmp = (*this) * 255.f;

			return SColor((uint32_t)core::round32(tmp.a), (uint32_t)core::round32(tmp.r), (uint32_t)core::round32(tmp.g), (uint32_t)core::round32(tmp.b));
		}

		//! Sets three color components to new values at once.
		/** \param rr: Red color component. Should be a value between 0.0f meaning
		no red (=black) and 1.0f, meaning full red.
		\param gg: Green color component. Should be a value between 0.0f meaning
		no green (=black) and 1.0f, meaning full green.
		\param bb: Blue color component. Should be a value between 0.0f meaning
		no blue (=black) and 1.0f, meaning full blue. */
		inline void set(float rr, float gg, float bb) {r = rr; g =gg; b = bb; }

		//! Sets all four color components to new values at once.
		/** \param aa: Alpha component. Should be a value between 0.0f meaning
		fully transparent and 1.0f, meaning opaque.
		\param rr: Red color component. Should be a value between 0.0f meaning
		no red and 1.0f, meaning full red.
		\param gg: Green color component. Should be a value between 0.0f meaning
		no green and 1.0f, meaning full green.
		\param bb: Blue color component. Should be a value between 0.0f meaning
		no blue and 1.0f, meaning full blue. */
		inline void set(float aa, float rr, float gg, float bb) {a = aa; r = rr; g =gg; b = bb; }

		//! Returns the alpha component of the color in the range 0.0 (transparent) to 1.0 (opaque)
		inline float getAlpha() const { return a; }

		//! Returns the red component of the color in the range 0.0 to 1.0
		inline float getRed() const { return r; }

		//! Returns the green component of the color in the range 0.0 to 1.0
		inline float getGreen() const { return g; }

		//! Returns the blue component of the color in the range 0.0 to 1.0
		inline float getBlue() const { return b; }


		//!
		inline vectorSIMDf& getAsVectorSIMDf() {return *this;}
		inline const vectorSIMDf& getAsVectorSIMDf() const {return *this;}
	};


} // end namespace video
} // end namespace irr

#endif