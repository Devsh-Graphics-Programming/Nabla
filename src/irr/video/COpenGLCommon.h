#ifndef __IRR_C_OPENGL_COMMON_H_INCLUDED__
#define __IRR_C_OPENGL_COMMON_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "COpenGLExtensionHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
namespace irr
{
namespace video
{

inline GLenum			getSizedOpenGLFormatFromOurFormat(asset::E_FORMAT format)
{
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