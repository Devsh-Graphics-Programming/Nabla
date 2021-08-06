// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _CAMERA_IMPL_
#define _CAMERA_IMPL_

#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

using namespace nbl;
using namespace core;
using namespace ui;

class Camera { 
public:
	Camera( const core::vector3df& position,
			const core::vectorSIMDf& lookat,
			float aspect,
			float fovy = core::radians(60),
			float znear = 1.0f,
			float zfar = 3000.0f,
			bool leftHanded = true
	) 
		: position(position)
		, target(lookat)
		, upVector(0.0f, 1.0f, 0.0f)
		, fovy(core::radians(60))
		, aspectRatio(16.f/9.f)
		, zNear(znear)
		, zFar(zfar)
		, leftHanded(leftHanded) 
	{
		recomputeViewMatrix();
		recomputeProjectionMatrix();
	}

	~Camera() = default;

public:

	inline const core::matrix4SIMD& getProjectionMatrix() const { return projMatrix; }
	inline const core::matrix3x4SIMD & getViewMatrix() const {	return viewMatrix; }
	inline const core::matrix4SIMD & getConcatenatedMatrix() const { return concatMatrix; }

	inline void setProjectionMatrix(const core::matrix4SIMD& projection) {
		projMatrix = projection;
		leftHanded = core::determinant(projMatrix) < 0.f;
		concatMatrix = core::matrix4SIMD::concatenateBFollowedByAPrecisely(projMatrix, core::matrix4SIMD(viewMatrix));
	}
	
	inline void setPosition(const core::vector3df& pos) {
		position.set(pos);
		recomputeViewMatrix();
	}
	
	inline const core::vector3df& getPosition() const { return position; }

	inline void setTarget(const core::vector3df& pos) {
		target.set(pos);
		recomputeViewMatrix();
	}

	inline const core::vectorSIMDf& getTarget() const { return target; }

	inline void setUpVector(const core::vectorSIMDf& up) {
		upVector = up;
	}

	inline const core::vectorSIMDf& getUpVector() const { return upVector; }

	inline void recomputeProjectionMatrix() {
		if (leftHanded)
			projMatrix = core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(fovy, aspectRatio, zNear, zFar);
		else
			projMatrix = core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(fovy, aspectRatio, zNear, zFar);
		concatMatrix = core::matrix4SIMD::concatenateBFollowedByAPrecisely(projMatrix, core::matrix4SIMD(viewMatrix));
	}
	
	inline void recomputeViewMatrix() {
		core::vectorSIMDf pos;
		pos.set(position);

		core::vectorSIMDf tgtv = core::normalize(target - pos);

		// if upvector and vector to the target are the same, we have a
		// problem. so solve this problem:
		core::vectorSIMDf up = core::normalize(upVector);

		core::vectorSIMDf dp = core::dot(tgtv,up);

		if (core::iszero(core::abs(dp)[0]-1.f))
		{
			up.X += 0.5f;
		}

		if (leftHanded)
			viewMatrix = core::matrix3x4SIMD::buildCameraLookAtMatrixLH(pos, target, up);
		else
			viewMatrix = core::matrix3x4SIMD::buildCameraLookAtMatrixRH(pos, target, up);
		concatMatrix = core::matrix4SIMD::concatenateBFollowedByAPrecisely(projMatrix, core::matrix4SIMD(viewMatrix));
	}

	inline bool getLeftHanded() const { return leftHanded; }
	
	inline float getNearValue() const { return zNear; }

	inline float getFarValue() const { return zFar; }

	inline float getAspectRatio() const { return aspectRatio; }

	inline float getFOV() const { return fovy; }

	inline void setLeftHanded(bool _leftHanded = true)
	{
		leftHanded = _leftHanded;
		recomputeViewMatrix();
		recomputeProjectionMatrix();
	}

	inline void setNearValue(float zn)
	{
		zNear = zn;
		recomputeProjectionMatrix();
	}

	inline void setFarValue(float zf)
	{
		zFar = zf;
		recomputeProjectionMatrix();
	}

	inline void setAspectRatio(float aspect)
	{
		aspectRatio = aspect;
		recomputeProjectionMatrix();
	}

	inline void setFOV(float fovy)
	{
		fovy = fovy;
		recomputeProjectionMatrix();
	}
public:
	
	void mouseProcess(const IMouseEventChannel::range_t& events)
	{
		for (auto eventIt=events.begin(); eventIt!=events.end(); eventIt++)
		{
			//logger->log("Mouse event at %d us",system::ILogger::ELL_INFO,(*eventIt).timeStamp);
		}
	}

	void keyboardProcess(const IKeyboardEventChannel::range_t& events)
	{
		for (auto eventIt=events.begin(); eventIt!=events.end(); eventIt++)
		{
			//logger->log("Mouse event at %d us",system::ILogger::ELL_INFO,(*eventIt).timeStamp);
		}
	}

private:
	core::vector3df position;
	core::vectorSIMDf target;
	core::vectorSIMDf upVector;
	
	core::matrix3x4SIMD viewMatrix;
	core::matrix4SIMD concatMatrix;

	float fovy;	// Field of view, in radians.
	float aspectRatio;	// aspectRatio ratio.
	float zNear;	// value of the near view-plane.
	float zFar;	// Z-value of the far view-plane.

	// actual projection matrix used
	core::matrix4SIMD projMatrix;

	bool leftHanded;
};

#endif // _CAMERA_IMPL_