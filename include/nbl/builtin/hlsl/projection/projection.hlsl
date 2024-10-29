#ifndef _NBL_BUILTIN_HLSL_PROJECTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_PROJECTION_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
// TODO: use glm instead for c++
inline float32_t4x4 buildProjectionMatrixPerspectiveFovLH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
{
#ifndef __HLSL_VERSION
	_NBL_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero
#endif

	const float h = core::reciprocal<float>(tanf(fieldOfViewRadians * 0.5f));
	const float w = h / aspectRatio;
	
	float32_t4x4 m;
	m[0] = float32_t4(w, 0.f, 0.f, 0.f);
	m[1] = float32_t4(0.f, -h, 0.f, 0.f);
	m[2] = float32_t4(0.f, 0.f, zFar / (zFar - zNear), -zNear * zFar / (zFar - zNear));
	m[3] = float32_t4(0.f, 0.f, 1.f, 0.f);
	
	return m;
}

inline float32_t4x4 buildProjectionMatrixPerspectiveFovRH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
{
	const float h = core::reciprocal<float>(tanf(fieldOfViewRadians * 0.5f));
#ifndef __HLSL_VERSION
	_NBL_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero
#endif
	const float w = h / aspectRatio;

	float32_t4x4 m;
	m[0] = hlsl::float32_t4(w, 0.f, 0.f, 0.f);
	m[1] = hlsl::float32_t4(0.f, -h, 0.f, 0.f);
	m[2] = hlsl::float32_t4(0.f, 0.f, -zFar / (zFar - zNear), -zNear * zFar / (zFar - zNear));
	m[3] = hlsl::float32_t4(0.f, 0.f, -1.f, 0.f);

	return m;
}

inline float32_t4x4 buildProjectionMatrixOrthoRH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
{
#ifndef __HLSL_VERSION
	_NBL_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero
#endif

	float32_t4x4 m;
	m[0] = hlsl::float32_t4(2.f / widthOfViewVolume, 0.f, 0.f, 0.f);
	m[1] = hlsl::float32_t4(0.f, -2.f / heightOfViewVolume, 0.f, 0.f);
	m[2] = hlsl::float32_t4(0.f, 0.f, -1.f / (zFar - zNear), -zNear / (zFar - zNear));
	m[3] = hlsl::float32_t4(0.f, 0.f, 0.f, 1.f);

	return m;
}
inline float32_t4x4 buildProjectionMatrixOrthoLH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
{
#ifndef __HLSL_VERSION
	_NBL_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero
#endif

	float32_t4x4 m;
	m[0] = hlsl::float32_t4(2.f / widthOfViewVolume, 0.f, 0.f, 0.f);
	m[1] = hlsl::float32_t4(0.f, -2.f / heightOfViewVolume, 0.f, 0.f);
	m[2] = hlsl::float32_t4(0.f, 0.f, 1.f / (zFar - zNear), -zNear / (zFar - zNear));
	m[3] = hlsl::float32_t4(0.f, 0.f, 0.f, 1.f);

	return m;
}

}
}

#endif