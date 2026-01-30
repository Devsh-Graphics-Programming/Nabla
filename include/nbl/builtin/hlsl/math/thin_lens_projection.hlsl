#ifndef _NBL_BUILTIN_HLSL_MATH_THIN_LENS_PROJECTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_THIN_LENS_PROJECTION_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>

namespace nbl
{
namespace hlsl
{
namespace math
{
namespace thin_lens
{

template<typename FloatingPoint NBL_FUNC_REQUIRES(concepts::FloatingPoint<FloatingPoint>)
inline matrix<FloatingPoint, 4, 4> rhPerspectiveFovMatrix(FloatingPoint fieldOfViewRadians, FloatingPoint aspectRatio, FloatingPoint zNear, FloatingPoint zFar)
{
	const FloatingPoint h = core::reciprocal<FloatingPoint>(tan(fieldOfViewRadians * 0.5f));
	_NBL_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
	const float w = h / aspectRatio;

	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<FloatingPoint, 4, 4> m;
	m[0] = vector<FloatingPoint, 4>(w, 0.f, 0.f, 0.f);
	m[1] = vector<FloatingPoint, 4>(0.f, -h, 0.f, 0.f);
	m[2] = vector<FloatingPoint, 4>(0.f, 0.f, -zFar / (zFar - zNear), -zNear * zFar / (zFar - zNear));
	m[3] = vector<FloatingPoint, 4>(0.f, 0.f, -1.f, 0.f);

	return m;
}
template<typename FloatingPoint NBL_FUNC_REQUIRES(concepts::FloatingPoint<FloatingPoint>)
inline matrix<FloatingPoint, 4, 4> lhPerspectiveFovMatrix(FloatingPoint fieldOfViewRadians, FloatingPoint aspectRatio, FloatingPoint zNear, FloatingPoint zFar)
{
	const FloatingPoint h = core::reciprocal<FloatingPoint>(tan(fieldOfViewRadians * 0.5f));
	_NBL_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
	const float w = h / aspectRatio;

	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<FloatingPoint, 4, 4> m;
	m[0] = vector<FloatingPoint, 4>(w, 0.f, 0.f, 0.f);
	m[1] = vector<FloatingPoint, 4>(0.f, -h, 0.f, 0.f);
	m[2] = vector<FloatingPoint, 4>(0.f, 0.f, zFar / (zFar - zNear), -zNear * zFar / (zFar - zNear));
	m[3] = vector<FloatingPoint, 4>(0.f, 0.f, 1.f, 0.f);

	return m;
}

template<typename FloatingPoint  NBL_FUNC_REQUIRES(concepts::FloatingPoint<FloatingPoint>)
inline matrix<FloatingPoint, 4, 4> rhProjectionOrthoMatrix(FloatingPoint widthOfViewVolume, FloatingPoint heightOfViewVolume, FloatingPoint zNear, FloatingPoint zFar)
{
	_NBL_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<FloatingPoint, 4, 4> m;
	m[0] = vector<FloatingPoint, 4>(2.f / widthOfViewVolume, 0.f, 0.f, 0.f);
	m[1] = vector<FloatingPoint, 4>(0.f, -2.f / heightOfViewVolume, 0.f, 0.f);
	m[2] = vector<FloatingPoint, 4>(0.f, 0.f, -1.f / (zFar - zNear), -zNear / (zFar - zNear));
	m[3] = vector<FloatingPoint, 4>(0.f, 0.f, 0.f, 1.f);

	return m;
}

template<typename FloatingPoint NBL_FUNC_REQUIRES(concepts::FloatingPoint<FloatingPoint>)
inline matrix<FloatingPoint, 4, 4> lhProjectionOrthoMatrix(FloatingPoint widthOfViewVolume, FloatingPoint heightOfViewVolume, FloatingPoint zNear, FloatingPoint zFar)
{
	_NBL_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<FloatingPoint, 4, 4> m;
	m[0] = vector<FloatingPoint, 4>(2.f / widthOfViewVolume, 0.f, 0.f, 0.f);
	m[1] = vector<FloatingPoint, 4>(0.f, -2.f / heightOfViewVolume, 0.f, 0.f);
	m[2] = vector<FloatingPoint, 4>(0.f, 0.f, 1.f / (zFar - zNear), -zNear / (zFar - zNear));
	m[3] = vector<FloatingPoint, 4>(0.f, 0.f, 0.f, 1.f);

	return m;
}

}
}
}
}

#endif