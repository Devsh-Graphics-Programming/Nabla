#ifndef _NBL_BUILTIN_HLSL_PROJECTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_PROJECTION_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
// TODO: use glm instead for c++
template<typename T>
inline matrix<T, 4, 4> buildProjectionMatrixPerspectiveFovRH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
{
	const float h = core::reciprocal<float>(tanf(fieldOfViewRadians * 0.5f));
	_NBL_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
	const float w = h / aspectRatio;

	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<T, 4, 4> m;
	m[0] = vector<T, 4>(w, 0.f, 0.f, 0.f);
	m[1] = vector<T, 4>(0.f, -h, 0.f, 0.f);
	m[2] = vector<T, 4>(0.f, 0.f, -zFar / (zFar - zNear), -zNear * zFar / (zFar - zNear));
	m[3] = vector<T, 4>(0.f, 0.f, -1.f, 0.f);

	return m;
}
template<typename T>
inline matrix<T, 4, 4> buildProjectionMatrixPerspectiveFovLH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
{
	const float h = core::reciprocal<float>(tanf(fieldOfViewRadians * 0.5f));
	_NBL_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
	const float w = h / aspectRatio;

	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<T, 4, 4> m;
	m[0] = vector<T, 4>(w, 0.f, 0.f, 0.f);
	m[1] = vector<T, 4>(0.f, -h, 0.f, 0.f);
	m[2] = vector<T, 4>(0.f, 0.f, zFar / (zFar - zNear), -zNear * zFar / (zFar - zNear));
	m[3] = vector<T, 4>(0.f, 0.f, 1.f, 0.f);

	return m;
}

template<typename T>
inline matrix<T, 4, 4> buildProjectionMatrixOrthoRH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
{
	_NBL_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<T, 4, 4> m;
	m[0] = vector<T, 4>(2.f / widthOfViewVolume, 0.f, 0.f, 0.f);
	m[1] = vector<T, 4>(0.f, -2.f / heightOfViewVolume, 0.f, 0.f);
	m[2] = vector<T, 4>(0.f, 0.f, -1.f / (zFar - zNear), -zNear / (zFar - zNear));
	m[3] = vector<T, 4>(0.f, 0.f, 0.f, 1.f);

	return m;
}

template<typename T>
inline matrix<T, 4, 4> buildProjectionMatrixOrthoLH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
{
	_NBL_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<T, 4, 4> m;
	m[0] = vector<T, 4>(2.f / widthOfViewVolume, 0.f, 0.f, 0.f);
	m[1] = vector<T, 4>(0.f, -2.f / heightOfViewVolume, 0.f, 0.f);
	m[2] = vector<T, 4>(0.f, 0.f, 1.f / (zFar - zNear), -zNear / (zFar - zNear));
	m[3] = vector<T, 4>(0.f, 0.f, 0.f, 1.f);

	return m;
}

}
}

#endif