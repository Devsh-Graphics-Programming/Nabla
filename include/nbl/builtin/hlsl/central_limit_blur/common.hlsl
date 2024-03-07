#ifndef _NBL_BUILTIN_HLSL_CENTRAL_LIMIT_BLUR_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_CENTRAL_LIMIT_BLUR_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl 
{
namespace hlsl
{
namespace central_limit_blur
{

enum WrapMode : uint32_t
{
	WRAP_MODE_CLAMP_TO_EDGE,
	WRAP_MODE_CLAMP_TO_BORDER,
	WRAP_MODE_REPEAT,
	WRAP_MODE_MIRROR,
};

enum BorderColor : uint32_t
{
	BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
	BORDER_COLOR_INT_TRANSPARENT_BLACK,
	BORDER_COLOR_FLOAT_OPAQUE_BLACK,
	BORDER_COLOR_INT_OPAQUE_BLACK,
	BORDER_COLOR_FLOAT_OPAQUE_WHITE,
	BORDER_COLOR_INT_OPAQUE_WHITE
};

struct BoxBlurParams
{
	float32_t radius;
	uint32_t direction : 2;
	uint32_t channelCount : 3;
	uint32_t wrapMode : 2;
	uint32_t borderColorType : 3;
	
	nbl::hlsl::float32_t4 getBorderColor()
	{
		nbl::hlsl::float32_t4 borderColor = nbl::hlsl::float32_t4( 1.f, 0.f, 1.f, 1.f );
		switch( borderColorType )
		{
		case BORDER_COLOR_FLOAT_TRANSPARENT_BLACK:
		case BORDER_COLOR_INT_TRANSPARENT_BLACK:
			borderColor = nbl::hlsl::float32_t4( 0.f, 0.f, 0.f, 0.f );
			break;

		case BORDER_COLOR_FLOAT_OPAQUE_BLACK:
		case BORDER_COLOR_INT_OPAQUE_BLACK:
			borderColor = nbl::hlsl::float32_t4( 0.f, 0.f, 0.f, 1.f );
			break;

		case BORDER_COLOR_FLOAT_OPAQUE_WHITE:
		case BORDER_COLOR_INT_OPAQUE_WHITE:
			borderColor = nbl::hlsl::float32_t4( 1.f, 1.f, 1.f, 1.f );
			break;
		}
		return borderColor;
	}
};

}
}
}
#endif
