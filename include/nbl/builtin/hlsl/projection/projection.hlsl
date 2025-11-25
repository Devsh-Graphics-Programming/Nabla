#ifndef _NBL_BUILTIN_HLSL_PROJECTION_PROJECTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_PROJECTION_PROJECTION_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
// TODO: use glm instead for c++
template<typename FloatingPoint>
inline matrix<FloatingPoint, 4, 4> buildProjectionMatrixPerspectiveFovRH(FloatingPoint fieldOfViewRadians, FloatingPoint aspectRatio, FloatingPoint zNear, FloatingPoint zFar)
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
template<typename FloatingPoint>
inline matrix<FloatingPoint, 4, 4> buildProjectionMatrixPerspectiveFovLH(FloatingPoint fieldOfViewRadians, FloatingPoint aspectRatio, FloatingPoint zNear, FloatingPoint zFar)
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

template<typename FloatingPoint>
inline matrix<FloatingPoint, 4, 4> buildProjectionMatrixOrthoRH(FloatingPoint widthOfViewVolume, FloatingPoint heightOfViewVolume, FloatingPoint zNear, FloatingPoint zFar)
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

template<typename FloatingPoint>
inline matrix<FloatingPoint, 4, 4> buildProjectionMatrixOrthoLH(FloatingPoint widthOfViewVolume, FloatingPoint heightOfViewVolume, FloatingPoint zNear, FloatingPoint zFar)
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

#endif