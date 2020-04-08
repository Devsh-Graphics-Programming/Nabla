#ifndef __IRR_E_COLOR_H_INCLUDED__
#define __IRR_E_COLOR_H_INCLUDED__

#include <cstdint>
#include <type_traits>
#include "IrrCompileConfig.h"
#include "vectorSIMD.h"
#include "irr/core/math/rational.h"

namespace irr
{
namespace asset
{
	//! An enum for the color format of textures used by the Irrlicht Engine.
	/** A color format specifies how color information is stored. */
	enum E_FORMAT : uint32_t
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
		EF_UNKNOWN
	};

    enum E_FORMAT_CLASS : uint32_t
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
        EFC_256_BIT
        //and many more for block compression and planar formats... but dont want to waste time on it now
    };

    inline uint32_t getFormatClassBlockBytesize(E_FORMAT_CLASS _fclass)
    {
        switch (_fclass)
        {
        case EFC_8_BIT: _IRR_FALLTHROUGH;
        case EFC_16_BIT: _IRR_FALLTHROUGH;
        case EFC_24_BIT: _IRR_FALLTHROUGH;
        case EFC_32_BIT:
            return _fclass+1u;
        case EFC_48_BIT:
            return 6u;
        case EFC_64_BIT:
            return 8u;
        case EFC_96_BIT:
            return 12u;
        case EFC_128_BIT:
            return 16u;
        case EFC_192_BIT:
            return 24u;
        case EFC_256_BIT:
            return 32u;
        default:
            _IRR_DEBUG_BREAK_IF(1);
            return 0u;
        }
    }

    inline core::vector3du32_SIMD getBlockDimensions(E_FORMAT_CLASS _fclass)
    {
        switch (_fclass)
        {
        case EFC_8_BIT: _IRR_FALLTHROUGH;
        case EFC_16_BIT: _IRR_FALLTHROUGH;
        case EFC_24_BIT: _IRR_FALLTHROUGH;
        case EFC_32_BIT: _IRR_FALLTHROUGH;
        case EFC_48_BIT: _IRR_FALLTHROUGH;
        case EFC_64_BIT: _IRR_FALLTHROUGH;
        case EFC_96_BIT: _IRR_FALLTHROUGH;
        case EFC_128_BIT: _IRR_FALLTHROUGH;
        case EFC_192_BIT: _IRR_FALLTHROUGH;
        case EFC_256_BIT:
            return core::vector3du32_SIMD(1u,1u,1u);
        default:
            _IRR_DEBUG_BREAK_IF(1);
            return core::vector3du32_SIMD(0u);
        }
    }

    inline E_FORMAT_CLASS getFormatClass(E_FORMAT _fmt)
    {
        switch (_fmt)
        {
            case EF_R4G4_UNORM_PACK8: _IRR_FALLTHROUGH;
            case EF_R8_UNORM: _IRR_FALLTHROUGH;
            case EF_R8_SNORM: _IRR_FALLTHROUGH;
            case EF_R8_USCALED: _IRR_FALLTHROUGH;
            case EF_R8_SSCALED: _IRR_FALLTHROUGH;
            case EF_R8_UINT: _IRR_FALLTHROUGH;
            case EF_R8_SINT: _IRR_FALLTHROUGH;
            case EF_R8_SRGB:
                return EFC_8_BIT;
            case EF_R4G4B4A4_UNORM_PACK16: _IRR_FALLTHROUGH;
            case EF_B4G4R4A4_UNORM_PACK16: _IRR_FALLTHROUGH;
            case EF_R5G6B5_UNORM_PACK16: _IRR_FALLTHROUGH;
            case EF_B5G6R5_UNORM_PACK16: _IRR_FALLTHROUGH;
            case EF_R5G5B5A1_UNORM_PACK16: _IRR_FALLTHROUGH;
            case EF_B5G5R5A1_UNORM_PACK16: _IRR_FALLTHROUGH;
            case EF_A1R5G5B5_UNORM_PACK16: _IRR_FALLTHROUGH;
            case EF_R8G8_UNORM: _IRR_FALLTHROUGH;
            case EF_R8G8_SNORM: _IRR_FALLTHROUGH;
            case EF_R8G8_USCALED: _IRR_FALLTHROUGH;
            case EF_R8G8_SSCALED: _IRR_FALLTHROUGH;
            case EF_R8G8_UINT: _IRR_FALLTHROUGH;
            case EF_R8G8_SINT: _IRR_FALLTHROUGH;
            case EF_R8G8_SRGB: _IRR_FALLTHROUGH;
            case EF_R16_UNORM: _IRR_FALLTHROUGH;
            case EF_R16_SNORM: _IRR_FALLTHROUGH;
            case EF_R16_USCALED: _IRR_FALLTHROUGH;
            case EF_R16_SSCALED: _IRR_FALLTHROUGH;
            case EF_R16_UINT: _IRR_FALLTHROUGH;
            case EF_R16_SINT: _IRR_FALLTHROUGH;
            case EF_R16_SFLOAT:
                return EFC_16_BIT;
            case EF_R8G8B8_UNORM: _IRR_FALLTHROUGH;
            case EF_R8G8B8_SNORM: _IRR_FALLTHROUGH;
            case EF_R8G8B8_USCALED: _IRR_FALLTHROUGH;
            case EF_R8G8B8_SSCALED: _IRR_FALLTHROUGH;
            case EF_R8G8B8_UINT: _IRR_FALLTHROUGH;
            case EF_R8G8B8_SINT: _IRR_FALLTHROUGH;
            case EF_R8G8B8_SRGB: _IRR_FALLTHROUGH;
            case EF_B8G8R8_UNORM: _IRR_FALLTHROUGH;
            case EF_B8G8R8_SNORM: _IRR_FALLTHROUGH;
            case EF_B8G8R8_USCALED: _IRR_FALLTHROUGH;
            case EF_B8G8R8_SSCALED: _IRR_FALLTHROUGH;
            case EF_B8G8R8_UINT: _IRR_FALLTHROUGH;
            case EF_B8G8R8_SINT: _IRR_FALLTHROUGH;
            case EF_B8G8R8_SRGB:
                return EFC_24_BIT;
            case EF_R8G8B8A8_UNORM: _IRR_FALLTHROUGH;
            case EF_R8G8B8A8_SNORM: _IRR_FALLTHROUGH;
            case EF_R8G8B8A8_USCALED: _IRR_FALLTHROUGH;
            case EF_R8G8B8A8_SSCALED: _IRR_FALLTHROUGH;
            case EF_R8G8B8A8_UINT: _IRR_FALLTHROUGH;
            case EF_R8G8B8A8_SINT: _IRR_FALLTHROUGH;
            case EF_R8G8B8A8_SRGB: _IRR_FALLTHROUGH;
            case EF_B8G8R8A8_UNORM: _IRR_FALLTHROUGH;
            case EF_B8G8R8A8_SNORM: _IRR_FALLTHROUGH;
            case EF_B8G8R8A8_USCALED: _IRR_FALLTHROUGH;
            case EF_B8G8R8A8_SSCALED: _IRR_FALLTHROUGH;
            case EF_B8G8R8A8_UINT: _IRR_FALLTHROUGH;
            case EF_B8G8R8A8_SINT: _IRR_FALLTHROUGH;
            case EF_B8G8R8A8_SRGB: _IRR_FALLTHROUGH;
            case EF_A8B8G8R8_UNORM_PACK32: _IRR_FALLTHROUGH;
            case EF_A8B8G8R8_SNORM_PACK32: _IRR_FALLTHROUGH;
            case EF_A8B8G8R8_USCALED_PACK32: _IRR_FALLTHROUGH;
            case EF_A8B8G8R8_SSCALED_PACK32: _IRR_FALLTHROUGH;
            case EF_A8B8G8R8_UINT_PACK32: _IRR_FALLTHROUGH;
            case EF_A8B8G8R8_SINT_PACK32: _IRR_FALLTHROUGH;
            case EF_A8B8G8R8_SRGB_PACK32: _IRR_FALLTHROUGH;
            case EF_A2R10G10B10_UNORM_PACK32: _IRR_FALLTHROUGH;
            case EF_A2R10G10B10_SNORM_PACK32: _IRR_FALLTHROUGH;
            case EF_A2R10G10B10_USCALED_PACK32: _IRR_FALLTHROUGH;
            case EF_A2R10G10B10_SSCALED_PACK32: _IRR_FALLTHROUGH;
            case EF_A2R10G10B10_UINT_PACK32: _IRR_FALLTHROUGH;
            case EF_A2R10G10B10_SINT_PACK32: _IRR_FALLTHROUGH;
            case EF_A2B10G10R10_UNORM_PACK32: _IRR_FALLTHROUGH;
            case EF_A2B10G10R10_SNORM_PACK32: _IRR_FALLTHROUGH;
            case EF_A2B10G10R10_USCALED_PACK32: _IRR_FALLTHROUGH;
            case EF_A2B10G10R10_SSCALED_PACK32: _IRR_FALLTHROUGH;
            case EF_A2B10G10R10_UINT_PACK32: _IRR_FALLTHROUGH;
            case EF_A2B10G10R10_SINT_PACK32: _IRR_FALLTHROUGH;
            case EF_R16G16_UNORM: _IRR_FALLTHROUGH;
            case EF_R16G16_SNORM: _IRR_FALLTHROUGH;
            case EF_R16G16_USCALED: _IRR_FALLTHROUGH;
            case EF_R16G16_SSCALED: _IRR_FALLTHROUGH;
            case EF_R16G16_UINT: _IRR_FALLTHROUGH;
            case EF_R16G16_SINT: _IRR_FALLTHROUGH;
            case EF_R16G16_SFLOAT: _IRR_FALLTHROUGH;
            case EF_R32_UINT: _IRR_FALLTHROUGH;
            case EF_R32_SINT: _IRR_FALLTHROUGH;
            case EF_R32_SFLOAT: _IRR_FALLTHROUGH;
            case EF_B10G11R11_UFLOAT_PACK32: _IRR_FALLTHROUGH;
            case EF_E5B9G9R9_UFLOAT_PACK32:
                return EFC_32_BIT;
            case EF_R16G16B16_UNORM: _IRR_FALLTHROUGH;
            case EF_R16G16B16_SNORM: _IRR_FALLTHROUGH;
            case EF_R16G16B16_USCALED: _IRR_FALLTHROUGH;
            case EF_R16G16B16_SSCALED: _IRR_FALLTHROUGH;
            case EF_R16G16B16_UINT: _IRR_FALLTHROUGH;
            case EF_R16G16B16_SINT: _IRR_FALLTHROUGH;
            case EF_R16G16B16_SFLOAT:
                return EFC_48_BIT;
            case EF_R16G16B16A16_UNORM: _IRR_FALLTHROUGH;
            case EF_R16G16B16A16_SNORM: _IRR_FALLTHROUGH;
            case EF_R16G16B16A16_USCALED: _IRR_FALLTHROUGH;
            case EF_R16G16B16A16_SSCALED: _IRR_FALLTHROUGH;
            case EF_R16G16B16A16_UINT: _IRR_FALLTHROUGH;
            case EF_R16G16B16A16_SINT: _IRR_FALLTHROUGH;
            case EF_R16G16B16A16_SFLOAT: _IRR_FALLTHROUGH;
            case EF_R32G32_UINT: _IRR_FALLTHROUGH;
            case EF_R32G32_SINT: _IRR_FALLTHROUGH;
            case EF_R32G32_SFLOAT: _IRR_FALLTHROUGH;
            case EF_R64_UINT: _IRR_FALLTHROUGH;
            case EF_R64_SINT: _IRR_FALLTHROUGH;
            case EF_R64_SFLOAT:
                return EFC_64_BIT;
            case EF_R32G32B32_UINT: _IRR_FALLTHROUGH;
            case EF_R32G32B32_SINT: _IRR_FALLTHROUGH;
            case EF_R32G32B32_SFLOAT:
                return EFC_96_BIT;
            case EF_R32G32B32A32_UINT: _IRR_FALLTHROUGH;
            case EF_R32G32B32A32_SINT: _IRR_FALLTHROUGH;
            case EF_R32G32B32A32_SFLOAT: _IRR_FALLTHROUGH;
            case EF_R64G64_UINT: _IRR_FALLTHROUGH;
            case EF_R64G64_SINT: _IRR_FALLTHROUGH;
            case EF_R64G64_SFLOAT:
                return EFC_128_BIT;
            case EF_R64G64B64_UINT: _IRR_FALLTHROUGH;
            case EF_R64G64B64_SINT: _IRR_FALLTHROUGH;
            case EF_R64G64B64_SFLOAT:
                return EFC_192_BIT;
            case EF_R64G64B64A64_UINT: _IRR_FALLTHROUGH;
            case EF_R64G64B64A64_SINT: _IRR_FALLTHROUGH;
            case EF_R64G64B64A64_SFLOAT:
                return EFC_256_BIT;
            default:
                _IRR_DEBUG_BREAK_IF(1);
                return static_cast<E_FORMAT_CLASS>(EFC_256_BIT+1);
        }
    }

    namespace impl
    {
        template<asset::E_FORMAT cf, asset::E_FORMAT cmp, asset::E_FORMAT... searchtab>
        struct is_any_of : is_any_of<cf, searchtab...> {};

        template<asset::E_FORMAT cf, asset::E_FORMAT cmp>
        struct is_any_of<cf, cmp> : std::false_type {}; //if last comparison is also false, than return false

        template<asset::E_FORMAT cf, asset::E_FORMAT... searchtab>
        struct is_any_of<cf, cf, searchtab...> : std::true_type {};
    }//namespace impl

    //! Utility functions
    inline uint32_t getTexelOrBlockBytesize(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case EF_R8G8B8_UINT: return 3;
        case EF_R8G8B8A8_UINT: return 4;
        case EF_B10G11R11_UFLOAT_PACK32: return 4;
        case EF_R16_SFLOAT: return 2;
        case EF_R16G16_SFLOAT: return 4;
        case EF_R16G16B16A16_SFLOAT: return 8;
        case EF_R32_SFLOAT: return 4;
        case EF_R32G32_SFLOAT: return 8;
        case EF_R32G32B32A32_SFLOAT: return 16;
        case EF_R8_UINT: return 1;
        case EF_R8G8_UINT: return 2;
        case EF_BC1_RGB_UNORM_BLOCK:
        case EF_BC1_RGBA_UNORM_BLOCK:
        case EF_BC1_RGB_SRGB_BLOCK:
        case EF_BC1_RGBA_SRGB_BLOCK:
            return 8;
        case EF_BC2_UNORM_BLOCK:
        case EF_BC3_UNORM_BLOCK:
        case EF_BC2_SRGB_BLOCK:
        case EF_BC3_SRGB_BLOCK:
            return 16;
        case EF_BC4_SNORM_BLOCK:
        case EF_BC4_UNORM_BLOCK:
            return 8u;
        case EF_BC5_SNORM_BLOCK:
        case EF_BC5_UNORM_BLOCK:
        case EF_BC6H_SFLOAT_BLOCK:
        case EF_BC6H_UFLOAT_BLOCK:
        case EF_BC7_SRGB_BLOCK:
        case EF_BC7_UNORM_BLOCK:
            return 16u;
        case EF_D16_UNORM: return 2;
        case EF_X8_D24_UNORM_PACK32: return 3;
        case EF_D32_SFLOAT:
        case EF_D24_UNORM_S8_UINT: return 4;
        case EF_D32_SFLOAT_S8_UINT: return 5;
        case EF_S8_UINT: return 2;
        case EF_E5B9G9R9_UFLOAT_PACK32: return 4;
        case EF_R4G4_UNORM_PACK8: return 1;
        case EF_R4G4B4A4_UNORM_PACK16: return 2;
        case EF_B4G4R4A4_UNORM_PACK16: return 2;
        case EF_R5G6B5_UNORM_PACK16: return 2;
        case EF_B5G6R5_UNORM_PACK16: return 2;
        case EF_R5G5B5A1_UNORM_PACK16: return 2;
        case EF_B5G5R5A1_UNORM_PACK16: return 2;
        case EF_A1R5G5B5_UNORM_PACK16: return 2;
        case EF_R8_UNORM: return 1;
        case EF_R8_SNORM: return 1;
        case EF_R8_USCALED: return 1;
        case EF_R8_SSCALED: return 1;
        case EF_R8_SINT: return 1;
        case EF_R8_SRGB: return 1;
        case EF_R8G8_UNORM: return 2;
        case EF_R8G8_SNORM: return 2;
        case EF_R8G8_USCALED: return 2;
        case EF_R8G8_SSCALED: return 2;
        case EF_R8G8_SINT: return 2;
        case EF_R8G8_SRGB: return 2;
        case EF_R8G8B8_UNORM: return 3;
        case EF_R8G8B8_SNORM: return 3;
        case EF_R8G8B8_USCALED: return 3;
        case EF_R8G8B8_SSCALED: return 3;
        case EF_R8G8B8_SINT: return 3;
        case EF_R8G8B8_SRGB: return 3;
        case EF_B8G8R8_UNORM: return 3;
        case EF_B8G8R8_SNORM: return 3;
        case EF_B8G8R8_USCALED: return 3;
        case EF_B8G8R8_SSCALED: return 3;
        case EF_B8G8R8_UINT: return 3;
        case EF_B8G8R8_SINT: return 3;
        case EF_B8G8R8_SRGB: return 3;
        case EF_R8G8B8A8_UNORM: return 4;
        case EF_R8G8B8A8_SNORM: return 4;
        case EF_R8G8B8A8_USCALED: return 4;
        case EF_R8G8B8A8_SSCALED: return 4;
        case EF_R8G8B8A8_SINT: return 4;
        case EF_R8G8B8A8_SRGB: return 4;
        case EF_B8G8R8A8_UNORM: return 4;
        case EF_B8G8R8A8_SNORM: return 4;
        case EF_B8G8R8A8_USCALED: return 4;
        case EF_B8G8R8A8_SSCALED: return 4;
        case EF_B8G8R8A8_UINT: return 4;
        case EF_B8G8R8A8_SINT: return 4;
        case EF_B8G8R8A8_SRGB: return 4;
        case EF_A8B8G8R8_UNORM_PACK32: return 4;
        case EF_A8B8G8R8_SNORM_PACK32: return 4;
        case EF_A8B8G8R8_USCALED_PACK32: return 4;
        case EF_A8B8G8R8_SSCALED_PACK32: return 4;
        case EF_A8B8G8R8_UINT_PACK32: return 4;
        case EF_A8B8G8R8_SINT_PACK32: return 4;
        case EF_A8B8G8R8_SRGB_PACK32: return 4;
        case EF_A2R10G10B10_UNORM_PACK32: return 4;
        case EF_A2R10G10B10_SNORM_PACK32: return 4;
        case EF_A2R10G10B10_USCALED_PACK32: return 4;
        case EF_A2R10G10B10_SSCALED_PACK32: return 4;
        case EF_A2R10G10B10_UINT_PACK32: return 4;
        case EF_A2R10G10B10_SINT_PACK32: return 4;
        case EF_A2B10G10R10_UNORM_PACK32: return 4;
        case EF_A2B10G10R10_SNORM_PACK32: return 4;
        case EF_A2B10G10R10_USCALED_PACK32: return 4;
        case EF_A2B10G10R10_SSCALED_PACK32: return 4;
        case EF_A2B10G10R10_UINT_PACK32: return 4;
        case EF_A2B10G10R10_SINT_PACK32: return 4;
        case EF_R16_UNORM: return 2;
        case EF_R16_SNORM: return 2;
        case EF_R16_USCALED: return 2;
        case EF_R16_SSCALED: return 2;
        case EF_R16_UINT: return 2;
        case EF_R16_SINT: return 2;
        case EF_R16G16_UNORM: return 4;
        case EF_R16G16_SNORM: return 4;
        case EF_R16G16_USCALED: return 4;
        case EF_R16G16_SSCALED: return 4;
        case EF_R16G16_UINT: return 4;
        case EF_R16G16_SINT: return 4;
        case EF_R16G16B16_UNORM: return 6;
        case EF_R16G16B16_SNORM: return 6;
        case EF_R16G16B16_USCALED: return 6;
        case EF_R16G16B16_SSCALED: return 6;
        case EF_R16G16B16_UINT: return 6;
        case EF_R16G16B16_SINT: return 6;
        case EF_R16G16B16A16_UNORM: return 8;
        case EF_R16G16B16A16_SNORM: return 8;
        case EF_R16G16B16A16_USCALED: return 8;
        case EF_R16G16B16A16_SSCALED: return 8;
        case EF_R16G16B16A16_UINT: return 8;
        case EF_R16G16B16A16_SINT: return 8;
        case EF_R32_UINT: return 4;
        case EF_R32_SINT: return 4;
        case EF_R32G32_UINT: return 8;
        case EF_R32G32_SINT: return 8;
        case EF_R32G32B32_UINT: return 12;
        case EF_R32G32B32_SINT: return 12;
        case EF_R32G32B32A32_UINT: return 16;
        case EF_R32G32B32A32_SINT: return 16;
        case EF_R64_UINT: return 8;
        case EF_R64_SINT: return 8;
        case EF_R64G64_UINT: return 16;
        case EF_R64G64_SINT: return 16;
        case EF_R64G64B64_UINT: return 24;
        case EF_R64G64B64_SINT: return 24;
        case EF_R64G64B64A64_UINT: return 32;
        case EF_R64G64B64A64_SINT: return 32;
        case EF_R16G16B16_SFLOAT: return 6;
        case EF_R32G32B32_SFLOAT: return 12;
        case EF_R64_SFLOAT: return 8;
        case EF_R64G64_SFLOAT: return 16;
        case EF_R64G64B64_SFLOAT: return 24;
        case EF_R64G64B64A64_SFLOAT: return 32;

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
            return 16;

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
            return 8u;

        case EF_PVRTC1_2BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC1_4BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC2_2BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC2_4BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
            return 8u;

        default: return 0;
        }
    }

    inline uint32_t getFormatChannelCount(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
        case EF_R8_UNORM:
        case EF_R8_SNORM:
        case EF_R8_USCALED:
        case EF_R8_SSCALED:
        case EF_R8_UINT:
        case EF_R8_SINT:
        case EF_R8_SRGB:
        case EF_R16_UNORM:
        case EF_R16_SNORM:
        case EF_R16_USCALED:
        case EF_R16_SSCALED:
        case EF_R16_UINT:
        case EF_R16_SINT:
        case EF_R16_SFLOAT:
        case EF_R32_UINT:
        case EF_R32_SINT:
        case EF_R32_SFLOAT:
        case EF_R64_UINT:
        case EF_R64_SINT:
        case EF_R64_SFLOAT:
        case EF_EAC_R11_UNORM_BLOCK:
        case EF_EAC_R11_SNORM_BLOCK:
        case EF_BC4_SNORM_BLOCK:
        case EF_BC4_UNORM_BLOCK:
            return 1u;

        case EF_R8G8_UNORM:
        case EF_R8G8_SNORM:
        case EF_R8G8_USCALED:
        case EF_R8G8_SSCALED:
        case EF_R8G8_UINT:
        case EF_R8G8_SINT:
        case EF_R8G8_SRGB:
        case EF_R4G4_UNORM_PACK8:
        case EF_R16G16_UNORM:
        case EF_R16G16_SNORM:
        case EF_R16G16_USCALED:
        case EF_R16G16_SSCALED:
        case EF_R16G16_UINT:
        case EF_R16G16_SINT:
        case EF_R16G16_SFLOAT:
        case EF_R32G32_UINT:
        case EF_R32G32_SINT:
        case EF_R32G32_SFLOAT:
        case EF_R64G64_UINT:
        case EF_R64G64_SINT:
        case EF_R64G64_SFLOAT:
        case EF_EAC_R11G11_UNORM_BLOCK:
        case EF_EAC_R11G11_SNORM_BLOCK:
        case EF_BC5_SNORM_BLOCK:
        case EF_BC5_UNORM_BLOCK:
            return 2u;

        case EF_R5G6B5_UNORM_PACK16:
        case EF_B5G6R5_UNORM_PACK16:
        case EF_R8G8B8_UNORM:
        case EF_R8G8B8_SNORM:
        case EF_R8G8B8_USCALED:
        case EF_R8G8B8_SSCALED:
        case EF_R8G8B8_UINT:
        case EF_R8G8B8_SINT:
        case EF_R8G8B8_SRGB:
        case EF_B8G8R8_UNORM:
        case EF_B8G8R8_SNORM:
        case EF_B8G8R8_USCALED:
        case EF_B8G8R8_SSCALED:
        case EF_B8G8R8_UINT:
        case EF_B8G8R8_SINT:
        case EF_B8G8R8_SRGB:
        case EF_R16G16B16_UNORM:
        case EF_R16G16B16_SNORM:
        case EF_R16G16B16_USCALED:
        case EF_R16G16B16_SSCALED:
        case EF_R16G16B16_UINT:
        case EF_R16G16B16_SINT:
        case EF_R16G16B16_SFLOAT:
        case EF_R32G32B32_UINT:
        case EF_R32G32B32_SINT:
        case EF_R32G32B32_SFLOAT:
        case EF_R64G64B64_UINT:
        case EF_R64G64B64_SINT:
        case EF_R64G64B64_SFLOAT:
        case EF_B10G11R11_UFLOAT_PACK32:
        case EF_E5B9G9R9_UFLOAT_PACK32:
        case EF_BC1_RGB_UNORM_BLOCK:
        case EF_BC1_RGB_SRGB_BLOCK:
        case EF_G8_B8_R8_3PLANE_420_UNORM:
        case EF_G8_B8R8_2PLANE_420_UNORM:
        case EF_G8_B8_R8_3PLANE_422_UNORM:
        case EF_G8_B8R8_2PLANE_422_UNORM:
        case EF_G8_B8_R8_3PLANE_444_UNORM:
        case EF_ETC2_R8G8B8_UNORM_BLOCK:
        case EF_ETC2_R8G8B8_SRGB_BLOCK:
        case EF_BC6H_SFLOAT_BLOCK:
        case EF_BC6H_UFLOAT_BLOCK:
            return 3u;

        case EF_R4G4B4A4_UNORM_PACK16:
        case EF_B4G4R4A4_UNORM_PACK16:
        case EF_R5G5B5A1_UNORM_PACK16:
        case EF_B5G5R5A1_UNORM_PACK16:
        case EF_A1R5G5B5_UNORM_PACK16:
        case EF_R8G8B8A8_UNORM:
        case EF_R8G8B8A8_SNORM:
        case EF_R8G8B8A8_USCALED:
        case EF_R8G8B8A8_SSCALED:
        case EF_R8G8B8A8_UINT:
        case EF_R8G8B8A8_SINT:
        case EF_R8G8B8A8_SRGB:
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
        case EF_A2R10G10B10_UNORM_PACK32:
        case EF_A2R10G10B10_SNORM_PACK32:
        case EF_A2R10G10B10_USCALED_PACK32:
        case EF_A2R10G10B10_SSCALED_PACK32:
        case EF_A2R10G10B10_UINT_PACK32:
        case EF_A2R10G10B10_SINT_PACK32:
        case EF_A2B10G10R10_UNORM_PACK32:
        case EF_A2B10G10R10_SNORM_PACK32:
        case EF_A2B10G10R10_USCALED_PACK32:
        case EF_A2B10G10R10_SSCALED_PACK32:
        case EF_A2B10G10R10_UINT_PACK32:
        case EF_A2B10G10R10_SINT_PACK32:
        case EF_R16G16B16A16_UNORM:
        case EF_R16G16B16A16_SNORM:
        case EF_R16G16B16A16_USCALED:
        case EF_R16G16B16A16_SSCALED:
        case EF_R16G16B16A16_UINT:
        case EF_R16G16B16A16_SINT:
        case EF_R16G16B16A16_SFLOAT:
        case EF_R32G32B32A32_UINT:
        case EF_R32G32B32A32_SINT:
        case EF_R32G32B32A32_SFLOAT:
        case EF_R64G64B64A64_UINT:
        case EF_R64G64B64A64_SINT:
        case EF_R64G64B64A64_SFLOAT:
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
        case EF_ETC2_R8G8B8A1_UNORM_BLOCK:
        case EF_ETC2_R8G8B8A1_SRGB_BLOCK:
        case EF_ETC2_R8G8B8A8_UNORM_BLOCK:
        case EF_ETC2_R8G8B8A8_SRGB_BLOCK:
        case EF_BC7_UNORM_BLOCK:
        case EF_BC7_SRGB_BLOCK:
        case EF_PVRTC1_2BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC1_4BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC2_2BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC2_4BPP_UNORM_BLOCK_IMG:
        case EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
        case EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
            return 4u;

        default:
            return 0u;
        }
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

    inline bool isBGRALayoutFormat(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
        //case EF_B8G8R8_UNORM:
        //case EF_B8G8R8_SNORM:
        //case EF_B8G8R8_USCALED:
        //case EF_B8G8R8_SSCALED:
        //case EF_B8G8R8_UINT:
        //case EF_B8G8R8_SINT:
        //case EF_B8G8R8_SRGB:
        case EF_A1R5G5B5_UNORM_PACK16:
        case EF_B8G8R8A8_UNORM:
        case EF_B8G8R8A8_SNORM:
        case EF_B8G8R8A8_USCALED:
        case EF_B8G8R8A8_SSCALED:
        case EF_B8G8R8A8_UINT:
        case EF_B8G8R8A8_SINT:
        case EF_B8G8R8A8_SRGB:
        case EF_A2R10G10B10_UNORM_PACK32:
        case EF_A2R10G10B10_SNORM_PACK32:
        case EF_A2R10G10B10_USCALED_PACK32:
        case EF_A2R10G10B10_SSCALED_PACK32:
        case EF_A2R10G10B10_UINT_PACK32:
        case EF_A2R10G10B10_SINT_PACK32:
            return true;
        default:
            return false;
        }
    }


    static inline const core::vector4du32_SIMD MaxTexelBlockDimensions(12u, 12u, 1u, 1u);

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

	inline core::rational<uint32_t> getBytesPerPixel(asset::E_FORMAT _fmt)
	{
		auto dims = getBlockDimensions(_fmt);
		return { getTexelOrBlockBytesize(_fmt), dims[0]*dims[1]*dims[2] };
	}

    template<asset::E_FORMAT cf>
    constexpr bool isSignedFormat()
    {
        return impl::is_any_of <
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
        return impl::is_any_of <
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
        return impl::is_any_of<
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
            EF_E5B9G9R9_UFLOAT_PACK32,
            EF_BC6H_SFLOAT_BLOCK,
            EF_BC6H_UFLOAT_BLOCK,
            EF_BC6H_UFLOAT_BLOCK
        >::value;
    }
    template<asset::E_FORMAT cf>
    constexpr bool isNormalizedFormat()
    {
        return impl::is_any_of <
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
        return impl::is_any_of <
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
        return impl::is_any_of<
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
        return impl::is_any_of<
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
    template<asset::E_FORMAT cf>
    constexpr bool isPlanarFormat()
    {
        return impl::is_any_of<
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
        case EF_R16G16_SNORM:
        case EF_R16G16_SSCALED:
        case EF_R16G16_SINT:
        case EF_R16G16B16_SNORM:
        case EF_R16G16B16_SSCALED:
        case EF_R16G16B16_SINT:
        case EF_R16G16B16A16_SNORM:
        case EF_R16G16B16A16_SSCALED:
        case EF_R16G16B16A16_SINT:
        case EF_R32_SINT:
        case EF_R32G32_SINT:
        case EF_R32G32B32_SINT:
        case EF_R32G32B32A32_SINT:
        case EF_R64_SINT:
        case EF_R64G64_SINT:
        case EF_R64G64B64_SINT:
        case EF_R64G64B64A64_SINT:
        case EF_R16G16B16_SFLOAT:
        case EF_R32G32B32_SFLOAT:
        case EF_R64_SFLOAT:
        case EF_R64G64_SFLOAT:
        case EF_R64G64B64_SFLOAT:
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
        case EF_R8_SINT:
        case EF_R8_SRGB:
        case EF_R8G8_SINT:
        case EF_R8G8_SRGB:
        case EF_R8G8B8_SINT:
        case EF_R8G8B8_SRGB:
        case EF_B8G8R8_UINT:
        case EF_B8G8R8_SINT:
        case EF_B8G8R8_SRGB:
        case EF_R8G8B8A8_SINT:
        case EF_R8G8B8A8_SRGB:
        case EF_B8G8R8A8_UINT:
        case EF_B8G8R8A8_SINT:
        case EF_B8G8R8A8_SRGB:
        case EF_A8B8G8R8_UINT_PACK32:
        case EF_A8B8G8R8_SINT_PACK32:
        case EF_A8B8G8R8_SRGB_PACK32:
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
        default: return false;
        }
    }
    inline bool isFloatingPointFormat(asset::E_FORMAT _fmt)
    {
        switch (_fmt)
        {
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

    template<E_FORMAT format>
    struct format_interm_storage_type
    {
        using type = typename std::conditional<isIntegerFormat<format>(),typename std::conditional<isSignedFormat<format>(),int64_t,uint64_t>::type,double>::type;
    };
}
}

namespace std
{
    template <>
    struct hash<irr::asset::E_FORMAT>
    {
        std::size_t operator()(const irr::asset::E_FORMAT& k) const noexcept { return k; }
    };
}

#endif //__IRR_E_COLOR_H_INCLUDED__