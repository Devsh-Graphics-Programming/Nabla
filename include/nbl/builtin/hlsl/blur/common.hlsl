#pragma once

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

_NBL_STATIC_INLINE_CONSTEXPR uint32_t WORKGROUP_SIZE = 256u;
_NBL_STATIC_INLINE_CONSTEXPR uint32_t PASSES_PER_AXIS = 3u;
_NBL_STATIC_INLINE_CONSTEXPR uint32_t AXIS_DIM = 3u; // HUH ?

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
	nbl::hlsl::uint32_t4 inputDimensions;
	nbl::hlsl::uint32_t4 inputStrides;
	nbl::hlsl::uint32_t4 outputStrides;
	float radius;

    uint32_t getDirection()
    {
        return nbl::hlsl::bitfield_extract_unsigned( inputDimensions.w, 30, 2 );
    }
    uint32_t getChannelCount()
    {
        return nbl::hlsl::bitfield_extract_unsigned( inputDimensions.w, 27, 3 );
    }

    uint32_t getWrapMode()
    {
        return nbl::hlsl::bitfield_extract_unsigned( inputDimensions.w, 25, 2 );
    }

    uint32_t getBorderColorType()
    {
        return nbl::hlsl::bitfield_extract_unsigned( inputDimensions.w, 22, 3 );
    }
	nbl::hlsl::float32_t4 nbl_glsl_ext_Blur_getBorderColor()
	{
		using namespace nbl::hlsl;

		float32_t4 borderColor = float32_t4( 1.f, 0.f, 1.f, 1.f );
		switch( getBorderColorType() )
		{
		case BORDER_COLOR_FLOAT_TRANSPARENT_BLACK:
		case BORDER_COLOR_INT_TRANSPARENT_BLACK:
			borderColor = float32_t4( 0.f, 0.f, 0.f, 0.f );
			break;

		case BORDER_COLOR_FLOAT_OPAQUE_BLACK:
		case BORDER_COLOR_INT_OPAQUE_BLACK:
			borderColor = float32_t4( 0.f, 0.f, 0.f, 1.f );
			break;

		case BORDER_COLOR_FLOAT_OPAQUE_WHITE:
		case BORDER_COLOR_INT_OPAQUE_WHITE:
			borderColor = float32_t4( 1.f, 1.f, 1.f, 1.f );
			break;
		}
		return borderColor;
	}
};

struct BufferAccessor
{
	nbl::hlsl::float32_t getPaddedData( const nbl::hlsl::uint32_t3 coordinate, const uint32_t channel );
	void setData( const nbl::hlsl::uint32_t3 coordinate, const uint32_t channel, const nbl::hlsl::float32_t val );
};