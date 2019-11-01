#ifndef __IRR_C_OPENGL_COMMON_H_INCLUDED__
#define __IRR_C_OPENGL_COMMON_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "COpenGLExtensionHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
namespace irr
{
namespace video
{

inline GLenum	getSizedOpenGLFormatFromOurFormat(asset::E_FORMAT format)
{
	using namespace asset;
	switch (format)
	{
		case asset::EF_A1R5G5B5_UNORM_PACK16:
			return GL_RGB5_A1;
			break;
		case asset::EF_R5G6B5_UNORM_PACK16:
			return GL_RGB565;
			break;
			// Floating Point texture formats. Thanks to Patryk "Nadro" Nadrowski.
		case asset::EF_B10G11R11_UFLOAT_PACK32:
			return GL_R11F_G11F_B10F;
			break;
		case asset::EF_R16_SFLOAT:
			return GL_R16F;
			break;
		case asset::EF_R16G16_SFLOAT:
			return GL_RG16F;
			break;
		case asset::EF_R16G16B16_SFLOAT:
			return GL_RGB16F;
		case asset::EF_R16G16B16A16_SFLOAT:
			return GL_RGBA16F;
			break;
		case asset::EF_R32_SFLOAT:
			return GL_R32F;
			break;
		case asset::EF_R32G32_SFLOAT:
			return GL_RG32F;
			break;
		case asset::EF_R32G32B32_SFLOAT:
			return GL_RGB32F;
			break;
		case asset::EF_R32G32B32A32_SFLOAT:
			return GL_RGBA32F;
			break;
		case asset::EF_R8_UNORM:
			return GL_R8;
			break;
		case asset::EF_R8_SRGB:
			if (!COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_EXT_texture_sRGB_R8])
				return GL_SR8_EXT;
			break;
		case asset::EF_R8G8_UNORM:
			return GL_RG8;
			break;
		case asset::EF_R8G8_SRGB:
			if (!COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_EXT_texture_sRGB_RG8])
				return GL_SRG8_EXT;
			break;
		case asset::EF_R8G8B8_UNORM:
			return GL_RGB8;
			break;
		case asset::EF_B8G8R8A8_UNORM:
			return GL_RGBA8;
			break;
		case asset::EF_B8G8R8A8_SRGB:
			return GL_SRGB8_ALPHA8;
			break;
		case asset::EF_R8G8B8A8_UNORM:
			return GL_RGBA8;
			break;
		case asset::EF_R8_UINT:
			return GL_R8UI;
			break;
		case asset::EF_R8G8_UINT:
			return GL_RG8UI;
			break;
		case asset::EF_R8G8B8_UINT:
			return GL_RGB8UI;
			break;
		case asset::EF_B8G8R8A8_UINT:
			return GL_RGBA8UI;
			break;
		case asset::EF_R8G8B8A8_UINT:
			return GL_RGBA8UI;
			break;
		case asset::EF_R8_SINT:
			return GL_R8I;
			break;
		case asset::EF_R8G8_SINT:
			return GL_RG8I;
			break;
		case asset::EF_R8G8B8_SINT:
			return GL_RGB8I;
			break;
		case asset::EF_B8G8R8A8_SINT:
			return GL_RGBA8I;
			break;
		case asset::EF_R8G8B8A8_SINT:
			return GL_RGBA8I;
			break;
		case asset::EF_R8_SNORM:
			return GL_R8_SNORM;
			break;
		case asset::EF_R8G8_SNORM:
			return GL_RG8_SNORM;
			break;
		case asset::EF_R8G8B8_SNORM:
			return GL_RGB8_SNORM;
			break;
		case asset::EF_B8G8R8A8_SNORM:
			return GL_RGBA8_SNORM;
			break;
		case asset::EF_R8G8B8A8_SNORM:
			return GL_RGBA8_SNORM;
			break;
		case asset::EF_R16_UNORM:
			return GL_R16;
			break;
		case asset::EF_R16G16_UNORM:
			return GL_RG16;
			break;
		case asset::EF_R16G16B16_UNORM:
			return GL_RGB16;
			break;
		case asset::EF_R16G16B16A16_UNORM:
			return GL_RGBA16;
			break;
		case asset::EF_R16_UINT:
			return GL_R16UI;
			break;
		case asset::EF_R16G16_UINT:
			return GL_RG16UI;
			break;
		case asset::EF_R16G16B16_UINT:
			return GL_RGB16UI;
			break;
		case asset::EF_R16G16B16A16_UINT:
			return GL_RGBA16UI;
			break;
		case asset::EF_R16_SINT:
			return GL_R16I;
			break;
		case asset::EF_R16G16_SINT:
			return GL_RG16I;
			break;
		case asset::EF_R16G16B16_SINT:
			return GL_RGB16I;
			break;
		case asset::EF_R16G16B16A16_SINT:
			return GL_RGBA16I;
			break;
		case asset::EF_R16_SNORM:
			return GL_R16_SNORM;
			break;
		case asset::EF_R16G16_SNORM:
			return GL_RG16_SNORM;
			break;
		case asset::EF_R16G16B16_SNORM:
			return GL_RGB16_SNORM;
			break;
		case asset::EF_R16G16B16A16_SNORM:
			return GL_RGBA16_SNORM;
			break;
		case asset::EF_R32_UINT:
			return GL_R32UI;
			break;
		case asset::EF_R32G32_UINT:
			return GL_RG32UI;
			break;
		case asset::EF_R32G32B32_UINT:
			return GL_RGB32UI;
			break;
		case asset::EF_R32G32B32A32_UINT:
			return GL_RGBA32UI;
			break;
		case asset::EF_R32_SINT:
			return GL_R32I;
			break;
		case asset::EF_R32G32_SINT:
			return GL_RG32I;
			break;
		case asset::EF_R32G32B32_SINT:
			return GL_RGB32I;
			break;
		case asset::EF_R32G32B32A32_SINT:
			return GL_RGBA32I;
			break;
		case asset::EF_A2B10G10R10_UNORM_PACK32:
			return GL_RGB10_A2;
			break;
		case asset::EF_A2B10G10R10_UINT_PACK32:
			return GL_RGB10_A2UI;
			break;
		case asset::EF_R8G8B8_SRGB:
			return GL_SRGB8;
			break;
		case asset::EF_R8G8B8A8_SRGB:
			return GL_SRGB8_ALPHA8;
			break;
		case asset::EF_BC1_RGB_UNORM_BLOCK:
			return GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
			break;
		case asset::EF_BC1_RGBA_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
			break;
		case asset::EF_BC2_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
			break;
		case asset::EF_BC3_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
			break;
		case asset::EF_BC1_RGB_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB_S3TC_DXT1_EXT;
			break;
		case asset::EF_BC1_RGBA_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
			break;
		case asset::EF_BC2_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
			break;
		case asset::EF_BC3_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
			break;
		case asset::EF_BC7_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_BPTC_UNORM;
		case asset::EF_BC7_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM;
		case asset::EF_BC6H_SFLOAT_BLOCK:
			return GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT;
		case asset::EF_BC6H_UFLOAT_BLOCK:
			return GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT;
		case asset::EF_ETC2_R8G8B8_UNORM_BLOCK:
			return GL_COMPRESSED_RGB8_ETC2;
		case asset::EF_ETC2_R8G8B8_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ETC2;
		case asset::EF_ETC2_R8G8B8A1_UNORM_BLOCK:
			return GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2;
		case asset::EF_ETC2_R8G8B8A1_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2;
		case asset::EF_ETC2_R8G8B8A8_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA8_ETC2_EAC;
		case asset::EF_ETC2_R8G8B8A8_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC;
		case asset::EF_EAC_R11G11_UNORM_BLOCK:
			return GL_COMPRESSED_RG11_EAC;
		case asset::EF_EAC_R11G11_SNORM_BLOCK:
			return GL_COMPRESSED_SIGNED_RG11_EAC;
		case asset::EF_EAC_R11_UNORM_BLOCK:
			return GL_COMPRESSED_R11_EAC;
		case asset::EF_EAC_R11_SNORM_BLOCK:
			return GL_COMPRESSED_SIGNED_R11_EAC;
		case EF_ASTC_4x4_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_4x4_KHR;
		case EF_ASTC_5x4_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_4x4_KHR;
		case EF_ASTC_5x5_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_5x5_KHR;
		case EF_ASTC_6x5_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_6x5_KHR;
		case EF_ASTC_6x6_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_6x6_KHR;
		case EF_ASTC_8x5_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_8x5_KHR;
		case EF_ASTC_8x6_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_8x6_KHR;
		case EF_ASTC_8x8_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_8x8_KHR;
		case EF_ASTC_10x5_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x5_KHR;
		case EF_ASTC_10x6_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x6_KHR;
		case EF_ASTC_10x8_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x6_KHR;
		case EF_ASTC_10x10_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x6_KHR;
		case EF_ASTC_12x10_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x6_KHR;
		case EF_ASTC_12x12_UNORM_BLOCK:
			return GL_COMPRESSED_RGBA_ASTC_10x6_KHR;
		case EF_ASTC_4x4_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR;
		case EF_ASTC_5x4_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR;
		case EF_ASTC_5x5_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR;
		case EF_ASTC_6x5_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR;
		case EF_ASTC_6x6_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR;
		case EF_ASTC_8x5_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR;
		case EF_ASTC_8x6_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR;
		case EF_ASTC_8x8_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR;
		case EF_ASTC_10x5_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR;
		case EF_ASTC_10x6_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR;
		case EF_ASTC_10x8_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR;
		case EF_ASTC_10x10_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR;
		case EF_ASTC_12x10_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR;
		case EF_ASTC_12x12_SRGB_BLOCK:
			return GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR;
			/* // todo bc
		case asset::EF_R_BC4:
			return GL_COMPRESSED_RED_RGTC1_EXT;
			break;
		case asset::EF_RG_BC5:
			return GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
			break;
			*/
			/// this is totally wrong but safe - most probs have to reupload
		case asset::EF_D16_UNORM:
			return GL_DEPTH_COMPONENT16;
			break;
		case asset::EF_X8_D24_UNORM_PACK32:
			return GL_DEPTH_COMPONENT24;
			break;
		case asset::EF_D24_UNORM_S8_UINT:
			return GL_DEPTH24_STENCIL8;
			break;
		case asset::EF_D32_SFLOAT:
			return GL_DEPTH_COMPONENT32F;
			break;
		case asset::EF_D32_SFLOAT_S8_UINT:
			return GL_DEPTH32F_STENCIL8;
			break;
		case asset::EF_S8_UINT:
			return GL_STENCIL_INDEX8;
			break;
		case asset::EF_E5B9G9R9_UFLOAT_PACK32:
			return GL_RGB9_E5;
			break;
		default:
			break;
	}
#ifdef _IRR_DEBUG
	os::Printer::log("Unsupported texture format", ELL_ERROR);
#endif // _IRR_DEBUG
	return GL_INVALID_ENUM;
}

inline asset::E_FORMAT	getOurFormatFromSizedOpenGLFormat(GLenum sizedFormat)
{
}


static GLenum formatEnumToGLenum(asset::E_FORMAT fmt)
{
    using namespace asset;
    switch (fmt)
    {
		case EF_R16_SFLOAT:
		case EF_R16G16_SFLOAT:
		case EF_R16G16B16_SFLOAT:
		case EF_R16G16B16A16_SFLOAT:
			return GL_HALF_FLOAT;
		case EF_R32_SFLOAT:
		case EF_R32G32_SFLOAT:
		case EF_R32G32B32_SFLOAT:
		case EF_R32G32B32A32_SFLOAT:
			return GL_FLOAT;
		case EF_B10G11R11_UFLOAT_PACK32:
			return GL_UNSIGNED_INT_10F_11F_11F_REV;
		case EF_R8_UNORM:
		case EF_R8_UINT:
		case EF_R8G8_UNORM:
		case EF_R8G8_UINT:
		case EF_R8G8B8_UNORM:
		case EF_R8G8B8_UINT:
		case EF_R8G8B8A8_UNORM:
		case EF_R8G8B8A8_UINT:
		case EF_R8_USCALED:
		case EF_R8G8_USCALED:
		case EF_R8G8B8_USCALED:
		case EF_R8G8B8A8_USCALED:
		case EF_B8G8R8A8_UNORM:
			return GL_UNSIGNED_BYTE;
		case EF_R8_SNORM:
		case EF_R8_SINT:
		case EF_R8G8_SNORM:
		case EF_R8G8_SINT:
		case EF_R8G8B8_SNORM:
		case EF_R8G8B8_SINT:
		case EF_R8G8B8A8_SNORM:
		case EF_R8G8B8A8_SINT:
		case EF_R8_SSCALED:
		case EF_R8G8_SSCALED:
		case EF_R8G8B8_SSCALED:
		case EF_R8G8B8A8_SSCALED:
			return GL_BYTE;
		case EF_R16_UNORM:
		case EF_R16_UINT:
		case EF_R16G16_UNORM:
		case EF_R16G16_UINT:
		case EF_R16G16B16_UNORM:
		case EF_R16G16B16_UINT:
		case EF_R16G16B16A16_UNORM:
		case EF_R16G16B16A16_UINT:
		case EF_R16_USCALED:
		case EF_R16G16_USCALED:
		case EF_R16G16B16_USCALED:
		case EF_R16G16B16A16_USCALED:
			return GL_UNSIGNED_SHORT;
		case EF_R16_SNORM:
		case EF_R16_SINT:
		case EF_R16G16_SNORM:
		case EF_R16G16_SINT:
		case EF_R16G16B16_SNORM:
		case EF_R16G16B16_SINT:
		case EF_R16G16B16A16_SNORM:
		case EF_R16G16B16A16_SINT:
		case EF_R16_SSCALED:
		case EF_R16G16_SSCALED:
		case EF_R16G16B16_SSCALED:
		case EF_R16G16B16A16_SSCALED:
			return GL_SHORT;
		case EF_R32_UINT:
		case EF_R32G32_UINT:
		case EF_R32G32B32_UINT:
		case EF_R32G32B32A32_UINT:
			return GL_UNSIGNED_INT;
		case EF_R32_SINT:
		case EF_R32G32_SINT:
		case EF_R32G32B32_SINT:
		case EF_R32G32B32A32_SINT:
			return GL_INT;
		case EF_A2R10G10B10_UNORM_PACK32:
		case EF_A2B10G10R10_UNORM_PACK32:
		case EF_A2B10G10R10_USCALED_PACK32:
		case EF_A2B10G10R10_UINT_PACK32:
			return GL_UNSIGNED_INT_2_10_10_10_REV;
		case EF_A2R10G10B10_SNORM_PACK32:
		case EF_A2B10G10R10_SNORM_PACK32:
		case EF_A2B10G10R10_SSCALED_PACK32:
		case EF_A2B10G10R10_SINT_PACK32:
			return GL_INT_2_10_10_10_REV;
		case EF_R64_SFLOAT:
		case EF_R64G64_SFLOAT:
		case EF_R64G64B64_SFLOAT:
		case EF_R64G64B64A64_SFLOAT:
			return GL_DOUBLE;

		default:
			return (GLenum)0;
    }
}

}
}
#endif


#endif