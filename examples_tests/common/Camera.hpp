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
	Camera() = default;

	Camera( const core::vectorSIMDf& position,
			const core::vectorSIMDf& lookat,
			const core::matrix4SIMD& projection,
			float moveSpeed = 1.0f,
			float rotateSpeed = 1.0f,
			const core::vectorSIMDf& upVec = core::vectorSIMDf(0.0f, 1.0f, 0.0f),
			const core::vectorSIMDf& backupUpVec = core::vectorSIMDf(0.5f, 1.0f, 0.0f)
	) 
		: position(position)
		, target(lookat)
		, firstUpdate(true)
		, moveSpeed(moveSpeed)
		, rotateSpeed(rotateSpeed)
		, upVector(upVec)
		, backupUpVector(backupUpVec)
	{
		allKeysUp();
		setProjectionMatrix(projection);
		recomputeViewMatrix();
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
	
	inline void setPosition(const core::vectorSIMDf& pos) {
		position.set(pos);
		recomputeViewMatrix();
	}
	
	inline const core::vectorSIMDf& getPosition() const { return position; }

	inline void setTarget(const core::vectorSIMDf& pos) {
		target.set(pos);
		recomputeViewMatrix();
	}

	inline const core::vectorSIMDf& getTarget() const { return target; }

	inline void setUpVector(const core::vectorSIMDf& up) {
		upVector = up;
	}
	
	inline void setBackupUpVector(const core::vectorSIMDf& up) {
		backupUpVector = up;
	}

	inline const core::vectorSIMDf& getUpVector() const { return upVector; }
	
	inline const core::vectorSIMDf& getBackupUpVector() const { return backupUpVector; }

	inline void recomputeViewMatrix() {
		core::vectorSIMDf pos = position;
		core::vectorSIMDf localTarget = core::normalize(target - pos);

		// if upvector and vector to the target are the same, we have a
		// problem. so solve this problem:
		core::vectorSIMDf up = core::normalize(upVector);
		core::vectorSIMDf cross = core::cross(localTarget, up);
		bool upVectorNeedsChange = core::lengthsquared(cross)[0] == 0;
		if (upVectorNeedsChange)
		{
			up = core::normalize(backupUpVector);
		}

		if (leftHanded)
			viewMatrix = core::matrix3x4SIMD::buildCameraLookAtMatrixLH(pos, target, up);
		else
			viewMatrix = core::matrix3x4SIMD::buildCameraLookAtMatrixRH(pos, target, up);
		concatMatrix = core::matrix4SIMD::concatenateBFollowedByAPrecisely(projMatrix, core::matrix4SIMD(viewMatrix));
	}

	inline bool getLeftHanded() const { return leftHanded; }

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

			if(ev.type == ui::SMouseEvent::EET_MOVEMENT && mouseDown) {
				core::vectorSIMDf pos = getPosition();
				core::vectorSIMDf localTarget = getTarget() - pos;

				// Get Relative Rotation for localTarget in Radians
				float relativeRotationX, relativeRotationY;
				relativeRotationY = atan2(localTarget.X, localTarget.Z);
				const double z1 = core::sqrt(localTarget.X*localTarget.X + localTarget.Z*localTarget.Z);
				relativeRotationX = atan2(z1, localTarget.Y) - core::PI<float>()/2;
				
				constexpr float RotateSpeedScale = 0.003f; 
				relativeRotationX -= ev.movementEvent.relativeMovementY * rotateSpeed * RotateSpeedScale * -1.0f;
				float tmpYRot = ev.movementEvent.relativeMovementX * rotateSpeed * RotateSpeedScale * -1.0f;
				if (leftHanded)
					relativeRotationY -= tmpYRot;
				else
					relativeRotationY += tmpYRot;

				const double MaxVerticalAngle = core::radians<float>(88.0f);
				if (relativeRotationX > MaxVerticalAngle*2 &&
					relativeRotationX < 2*core::PI<float>()-MaxVerticalAngle)
				{
					relativeRotationX = 2*core::PI<float>()-MaxVerticalAngle;
				}
				else
				if (relativeRotationX > MaxVerticalAngle &&
					relativeRotationX < 2*core::PI<float>()-MaxVerticalAngle)
				{
					relativeRotationX = MaxVerticalAngle;
				}

				localTarget.set(0,0, core::max(1.f, core::length(pos)[0]), 1.f);

				core::matrix3x4SIMD mat;
				{
					mat.setRotation(core::quaternion(relativeRotationX, relativeRotationY, 0));
				}
				mat.transformVect(localTarget);
				
				setTarget(localTarget + pos);
			}
		}
	}

	void keyboardProcess(const IKeyboardEventChannel::range_t& events)
	{
		for(uint32_t k = 0; k < Keys::EKA_COUNT; ++k) {
			perActionDt[k] = 0.0;
		}

		/*
		* If a Key was already being held down from previous frames
		* Compute with this assumption that the key will be held down for this whole frame as well,
		* And If an UP event was sent It will get subtracted it from this value. (Currently Disabled Because we Need better Oracle)
		*/
		for(uint32_t k = 0; k < Keys::EKA_COUNT; ++k) {
			if(keysDown[k] == true) {
				auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - lastVirtualUpTimeStamp).count();
				assert(timeDiff >= 0);
				perActionDt[k] += timeDiff;
			}
		}

		for (auto eventIt=events.begin(); eventIt!=events.end(); eventIt++)
		{
			auto ev = *eventIt;
			
			// Accumulate the periods for which a key was down
			auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - ev.timeStamp).count();
			assert(timeDiff >= 0);

			if(ev.keyCode == ui::EKC_UP_ARROW || ev.keyCode == ui::EKC_W) {
				if(ev.action == ui::SKeyboardEvent::ECA_PRESSED && keysDown[Keys::EKA_MOVE_FORWARD] == false) {
					perActionDt[Keys::EKA_MOVE_FORWARD] += timeDiff; 
					keysDown[Keys::EKA_MOVE_FORWARD] = true;
				} else if(ev.action == ui::SKeyboardEvent::ECA_RELEASED) {
					// perActionDt[Keys::EKA_MOVE_FORWARD] -= timeDiff; 
					keysDown[Keys::EKA_MOVE_FORWARD] = false;
				}
			}

			if(ev.keyCode == ui::EKC_DOWN_ARROW || ev.keyCode == ui::EKC_S) {
				if(ev.action == ui::SKeyboardEvent::ECA_PRESSED && keysDown[Keys::EKA_MOVE_BACKWARD] == false) {
					perActionDt[Keys::EKA_MOVE_BACKWARD] += timeDiff; 
					keysDown[Keys::EKA_MOVE_BACKWARD] = true;
				} else if(ev.action == ui::SKeyboardEvent::ECA_RELEASED) {
					// perActionDt[Keys::EKA_MOVE_BACKWARD] -= timeDiff; 
					keysDown[Keys::EKA_MOVE_BACKWARD] = false;
				}
			}

			if(ev.keyCode == ui::EKC_LEFT_ARROW || ev.keyCode == ui::EKC_A) {
				if(ev.action == ui::SKeyboardEvent::ECA_PRESSED && keysDown[Keys::EKA_MOVE_LEFT] == false) {
					perActionDt[Keys::EKA_MOVE_LEFT] += timeDiff; 
					keysDown[Keys::EKA_MOVE_LEFT] = true;
				} else if(ev.action == ui::SKeyboardEvent::ECA_RELEASED) {
					// perActionDt[Keys::EKA_MOVE_LEFT] -= timeDiff; 
					keysDown[Keys::EKA_MOVE_LEFT] = false;
				}
			}

			if(ev.keyCode == ui::EKC_RIGHT_ARROW || ev.keyCode == ui::EKC_D) {
				if(ev.action == ui::SKeyboardEvent::ECA_PRESSED && keysDown[Keys::EKA_MOVE_RIGHT] == false) {
					perActionDt[Keys::EKA_MOVE_RIGHT] += timeDiff; 
					keysDown[Keys::EKA_MOVE_RIGHT] = true;
				} else if(ev.action == ui::SKeyboardEvent::ECA_RELEASED) {
					// perActionDt[Keys::EKA_MOVE_RIGHT] -= timeDiff; 
					keysDown[Keys::EKA_MOVE_RIGHT] = false;
				}
			}
		}
	}

	void beginInputProcessing(std::chrono::microseconds _nextPresentationTimeStamp) {
		nextPresentationTimeStamp = _nextPresentationTimeStamp;

		if(firstUpdate) {
			lastVirtualUpTimeStamp = nextPresentationTimeStamp;
			// Set Cursor to middle of the screen
			firstUpdate = false;
		}

		return;
	}
	
	void endInputProcessing(std::chrono::microseconds _nextPresentationTimeStamp) {
		core::vectorSIMDf pos = getPosition();
		core::vectorSIMDf localTarget = getTarget() - pos;

		core::vectorSIMDf movedir = localTarget;
		movedir.makeSafe3D();
		movedir = core::normalize(movedir);

		constexpr float MoveSpeedScale = 0.02f; 

		pos += movedir * perActionDt[Keys::EKA_MOVE_FORWARD] * moveSpeed * MoveSpeedScale;

		pos -= movedir * perActionDt[Keys::EKA_MOVE_BACKWARD] * moveSpeed * MoveSpeedScale;

		// strafing
		
		// if upvector and vector to the target are the same, we have a
		// problem. so solve this problem:
		core::vectorSIMDf up = core::normalize(upVector);
		core::vectorSIMDf cross = core::cross(localTarget, up);
		bool upVectorNeedsChange = core::lengthsquared(cross)[0] == 0;
		if (upVectorNeedsChange)
		{
			up = core::normalize(backupUpVector);
		}

		core::vectorSIMDf strafevect = localTarget;
		if (leftHanded)
			strafevect = core::cross(strafevect, up);
		else
			strafevect = core::cross(up, strafevect);

		strafevect = core::normalize(strafevect);

		pos += strafevect * perActionDt[Keys::EKA_MOVE_LEFT] * moveSpeed * MoveSpeedScale;

		pos -= strafevect * perActionDt[Keys::EKA_MOVE_RIGHT] * moveSpeed * MoveSpeedScale;

		setPosition(pos);

		setTarget(localTarget + pos);

		lastVirtualUpTimeStamp = nextPresentationTimeStamp;
	}

private:
	
	void allKeysUp() {
		for (uint32_t i=0; i< Keys::EKA_COUNT; ++i) {
			keysDown[i] = false;
		}
		mouseDown = false;
	}

private:
	core::vectorSIMDf position;
	core::vectorSIMDf target;
	core::vectorSIMDf upVector;
	core::vectorSIMDf backupUpVector;
	
	core::matrix3x4SIMD viewMatrix;
	core::matrix4SIMD concatMatrix;

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

	bool keysDown[Keys::EKA_COUNT] = {};
	// perActionDt is the duration which the key was being held down from lastVirtualUpTimeStamp(=last "guessed" presentation time) to nextPresentationTimeStamp
	double perActionDt[Keys::EKA_COUNT] = {};

	float moveSpeed;
	float rotateSpeed;

	bool firstUpdate = true;
	bool mouseDown = false;

	std::chrono::microseconds nextPresentationTimeStamp;
	std::chrono::microseconds lastVirtualUpTimeStamp;
};

#endif // _CAMERA_IMPL_