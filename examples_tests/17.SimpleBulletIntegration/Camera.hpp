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
		, firstUpdate(true)
	{
		allKeysUp();
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
			auto ev = *eventIt;

			if(ev.type == ui::SMouseEvent::EET_CLICK && ev.clickEvent.mouseButton == ui::EMB_LEFT_BUTTON) {
				if(ev.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_PRESSED) {
					mouseDown = true;
				} else if (ev.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_RELEASED) {
					mouseDown = false;
				}
			}

		}
	}

	void keyboardProcess(const IKeyboardEventChannel::range_t& events)
	{
		for (auto eventIt=events.begin(); eventIt!=events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if(ev.keyCode == ui::EKC_UP_ARROW || ev.keyCode == ui::EKC_W) {
				if(ev.action == ui::SKeyboardEvent::ECA_PRESSED) {
					keysDown[Keys::EKA_MOVE_FORWARD] = true;
				} else if(ev.action == ui::SKeyboardEvent::ECA_RELEASED) {
					keysDown[Keys::EKA_MOVE_FORWARD] = false;
				}
			}

			if(ev.keyCode == ui::EKC_DOWN_ARROW || ev.keyCode == ui::EKC_S) {
				if(ev.action == ui::SKeyboardEvent::ECA_PRESSED) {
					keysDown[Keys::EKA_MOVE_BACKWARD] = true;
				} else if(ev.action == ui::SKeyboardEvent::ECA_RELEASED) {
					keysDown[Keys::EKA_MOVE_BACKWARD] = false;
				}
			}

			if(ev.keyCode == ui::EKC_LEFT_ARROW || ev.keyCode == ui::EKC_A) {
				if(ev.action == ui::SKeyboardEvent::ECA_PRESSED) {
					keysDown[Keys::EKA_MOVE_LEFT] = true;
				} else if(ev.action == ui::SKeyboardEvent::ECA_RELEASED) {
					keysDown[Keys::EKA_MOVE_LEFT] = false;
				}
			}

			if(ev.keyCode == ui::EKC_RIGHT_ARROW || ev.keyCode == ui::EKC_D) {
				if(ev.action == ui::SKeyboardEvent::ECA_PRESSED) {
					keysDown[Keys::EKA_MOVE_RIGHT] = true;
				} else if(ev.action == ui::SKeyboardEvent::ECA_RELEASED) {
					keysDown[Keys::EKA_MOVE_RIGHT] = false;
				}
			}
		}
	}

	void update(double dt) {
		float moveSpeed = 0.1f;
		float rotateSpeed = 1.0f;

		if(!mouseDown) {
			return;
		}

		if(firstUpdate) {
			// Set Cursor to middle of the screen
			firstUpdate = false;
		}

		// update position
		core::vectorSIMDf pos; pos.set(getPosition());

		// Update rotation
		core::vectorSIMDf target = getTarget() - pos;
		core::vector3df relativeRotation = target.getAsVector3df().getHorizontalAngle();
	
		// set target
		
		target.set(0,0, core::max(1.f, core::length(pos)[0]), 1.f);
		core::vectorSIMDf movedir = target;

		core::matrix3x4SIMD mat;
		{
			core::matrix4x3 tmp;
			tmp.setRotationDegrees(core::vector3df(relativeRotation.X, relativeRotation.Y, 0));
			mat.set(tmp);
		}
		mat.transformVect(target);
		target.makeSafe3D();

		movedir = target;

		movedir.makeSafe3D();
		movedir = core::normalize(movedir);

		if (keysDown[Keys::EKA_MOVE_FORWARD])
			pos += movedir * dt * moveSpeed;

		if (keysDown[Keys::EKA_MOVE_BACKWARD])
			pos -= movedir * dt * moveSpeed;

		// strafing

		core::vectorSIMDf strafevect; strafevect.set(target);
		if (leftHanded)
			strafevect = core::cross(strafevect, upVector);
		else
			strafevect = core::cross(upVector, strafevect);

		strafevect = core::normalize(strafevect);

		if (keysDown[Keys::EKA_MOVE_LEFT])
			pos += strafevect * dt * moveSpeed;

		if (keysDown[Keys::EKA_MOVE_RIGHT])
			pos -= strafevect * dt * moveSpeed;

		// write translation
		setPosition(pos.getAsVector3df());

		// write right target
		target += pos;
		setTarget(target.getAsVector3df());
	}

private:
	
	void allKeysUp() {
		for (uint32_t i=0; i< Keys::EKA_COUNT; ++i) {
			keysDown[i] = false;
		}
		mouseDown = false;
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
	
	// Animation
	
	enum Keys {
		EKA_MOVE_FORWARD = 0,
		EKA_MOVE_BACKWARD,
		EKA_MOVE_LEFT,
		EKA_MOVE_RIGHT,
		EKA_COUNT,
	};

	bool keysDown[(size_t)Keys::EKA_COUNT] = {};
	bool firstUpdate = true;
	bool mouseDown = false;
};

#endif // _CAMERA_IMPL_