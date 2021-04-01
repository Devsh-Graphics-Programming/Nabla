// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CSceneNodeAnimatorCameraFPS.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"
#include "Keycodes.h"
#include "ICursorControl.h"
#include "ICameraSceneNode.h"

namespace nbl
{
namespace scene
{

//! constructor
CSceneNodeAnimatorCameraFPS::CSceneNodeAnimatorCameraFPS(gui::ICursorControl* cursorControl,
		float rotateSpeed, float moveSpeed, float jumpSpeed,
		SKeyMap* keyMapArray, uint32_t keyMapSize, bool noVerticalMovement, bool invertY)
: CursorControl(cursorControl), MaxVerticalAngle(88.0f),
	MoveSpeed(moveSpeed), RotateSpeed(rotateSpeed), JumpSpeed(jumpSpeed),
	MouseYDirection(invertY ? -1.0f : 1.0f),
	LastAnimationTime(0), firstUpdate(true), firstInput(true), NoVerticalMovement(noVerticalMovement)
{
	#ifdef _NBL_DEBUG
	setDebugName("CCameraSceneNodeAnimatorFPS");
	#endif

	if (CursorControl)
		CursorControl->grab();

	allKeysUp();

	// create key map
	if (!keyMapArray || !keyMapSize)
	{
		// create default key map
		KeyMap.push_back(SKeyMap(EKA_MOVE_FORWARD, nbl::KEY_UP));
		KeyMap.push_back(SKeyMap(EKA_MOVE_BACKWARD, nbl::KEY_DOWN));
		KeyMap.push_back(SKeyMap(EKA_STRAFE_LEFT, nbl::KEY_LEFT));
		KeyMap.push_back(SKeyMap(EKA_STRAFE_RIGHT, nbl::KEY_RIGHT));
		KeyMap.push_back(SKeyMap(EKA_JUMP_UP, nbl::KEY_KEY_J));
	}
	else
	{
		// create custom key map
		setKeyMap(keyMapArray, keyMapSize);
	}
}


//! destructor
CSceneNodeAnimatorCameraFPS::~CSceneNodeAnimatorCameraFPS()
{
	if (CursorControl)
		CursorControl->drop();
}


//! It is possible to send mouse and key events to the camera. Most cameras
//! may ignore this input, but camera scene nodes which are created for
//! example with scene::ISceneManager::addMayaCameraSceneNode or
//! scene::ISceneManager::addFPSCameraSceneNode, may want to get this input
//! for changing their position, look at target or whatever.
bool CSceneNodeAnimatorCameraFPS::OnEvent(const SEvent& evt)
{
	switch(evt.EventType)
	{
	case EET_KEY_INPUT_EVENT:
		for (uint32_t i=0; i<KeyMap.size(); ++i)
		{
			if (KeyMap[i].KeyCode == evt.KeyInput.Key)
			{
				CursorKeys[KeyMap[i].Action] = evt.KeyInput.PressedDown;
				return true;
			}
		}
		break;

	case EET_MOUSE_INPUT_EVENT:
		if (evt.MouseInput.Event == EMIE_MOUSE_MOVED)
		{
			CursorPos = CursorControl->getRelativePosition();
			return true;
		}
		break;

	default:
		break;
	}

	return false;
}


void CSceneNodeAnimatorCameraFPS::animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs)
{
	if (!node/* || node->getType() != ESNT_CAMERA*/)
		return;

	ICameraSceneNode* camera = static_cast<ICameraSceneNode*>(node);

	if (firstUpdate)
	{
		camera->updateAbsolutePosition();
		if (CursorControl )
		{
			CursorControl->setPosition(0.5f, 0.5f);
			CursorPos = CenterCursor = CursorControl->getRelativePosition();
		}

		LastAnimationTime = timeMs;

		firstUpdate = false;
	}

	// If the camera isn't the active camera, and receiving input, then don't process it.
	if(!camera->isInputReceiverEnabled())
	{
		firstInput = true;
		return;
	}

	if ( firstInput )
	{
		allKeysUp();
		firstInput = false;
	}
/*
	scene::ISceneManager * smgr = camera->getSceneManager();
	if(smgr && smgr->getActiveCamera() != camera)
		return;
*/
	// get time
	float timeDiff = (float) ( timeMs - LastAnimationTime );
	LastAnimationTime = timeMs;

	// update position
	core::vectorSIMDf pos; pos.set(camera->getPosition());

	// Update rotation
	core::vectorSIMDf abspos; abspos.set(camera->getAbsolutePosition());
	core::vectorSIMDf target = camera->getTarget() - abspos;
	core::vector3df relativeRotation = target.getAsVector3df().getHorizontalAngle();

	if (CursorControl)
	{
		if (CursorPos != CenterCursor)
		{
			relativeRotation.X -= (0.5f - CursorPos.Y) * RotateSpeed * MouseYDirection;
			float tmpYRot = (0.5f - CursorPos.X) * RotateSpeed;
			if (camera->getLeftHanded())
				relativeRotation.Y -= tmpYRot;
			else
				relativeRotation.Y += tmpYRot;

			// X < MaxVerticalAngle or X > 360-MaxVerticalAngle

			if (relativeRotation.X > MaxVerticalAngle*2 &&
				relativeRotation.X < 360.0f-MaxVerticalAngle)
			{
				relativeRotation.X = 360.0f-MaxVerticalAngle;
			}
			else
			if (relativeRotation.X > MaxVerticalAngle &&
				relativeRotation.X < 360.0f-MaxVerticalAngle)
			{
				relativeRotation.X = MaxVerticalAngle;
			}

			// Do the fix as normal, special case below
			// reset cursor position to the centre of the window.
			CursorControl->setPosition(0.5f, 0.5f);
			CenterCursor = CursorControl->getRelativePosition();

			// needed to avoid problems when the event receiver is disabled
			CursorPos = CenterCursor;
		}

		// Special case, mouse is whipped outside of window before it can update.
		auto mousepos = CursorControl->getRelativePosition();

		// Only if we are moving outside quickly.
		bool reset = mousepos.X<0.f||mousepos.X>1.f||mousepos.Y<0.f||mousepos.Y>1.f;
		if(reset)
		{
			// Force a reset.
			CursorControl->setPosition(0.5f, 0.5f);
			CenterCursor = CursorControl->getRelativePosition();
			CursorPos = CenterCursor;
 		}
	}

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

	if (NoVerticalMovement)
	{
		core::matrix4x3 tmp;
		tmp.setRotationDegrees(core::vector3df(0, relativeRotation.Y, 0));
		mat.set(tmp);
		mat.transformVect(movedir);
	}
	else
	{
		movedir = target;
	}

	movedir.makeSafe3D();
	movedir = core::normalize(movedir);

	if (CursorKeys[EKA_MOVE_FORWARD])
		pos += movedir * timeDiff * MoveSpeed;

	if (CursorKeys[EKA_MOVE_BACKWARD])
		pos -= movedir * timeDiff * MoveSpeed;

	// strafing

	core::vectorSIMDf strafevect; strafevect.set(target);
	if (camera->getLeftHanded())
		strafevect = core::cross(strafevect,camera->getUpVector());
	else
		strafevect = core::cross(camera->getUpVector(),strafevect);

	if (NoVerticalMovement)
		strafevect.Y = 0.0f;

	strafevect = core::normalize(strafevect);

	if (CursorKeys[EKA_STRAFE_LEFT])
		pos += strafevect * timeDiff * MoveSpeed;

	if (CursorKeys[EKA_STRAFE_RIGHT])
		pos -= strafevect * timeDiff * MoveSpeed;

	// write translation
	camera->setPosition(pos.getAsVector3df());

	// write right target
	target += pos;
	camera->setTarget(target.getAsVector3df());
}


void CSceneNodeAnimatorCameraFPS::allKeysUp()
{
	for (uint32_t i=0; i<EKA_COUNT; ++i)
		CursorKeys[i] = false;
}


//! Sets the rotation speed
void CSceneNodeAnimatorCameraFPS::setRotateSpeed(float speed)
{
	RotateSpeed = speed;
}


//! Sets the movement speed
void CSceneNodeAnimatorCameraFPS::setMoveSpeed(float speed)
{
	MoveSpeed = speed;
}


//! Gets the rotation speed
float CSceneNodeAnimatorCameraFPS::getRotateSpeed() const
{
	return RotateSpeed;
}


// Gets the movement speed
float CSceneNodeAnimatorCameraFPS::getMoveSpeed() const
{
	return MoveSpeed;
}


//! Sets the keyboard mapping for this animator
void CSceneNodeAnimatorCameraFPS::setKeyMap(SKeyMap *map, uint32_t count)
{
	// clear the keymap
	KeyMap.clear();

	// add actions
	for (uint32_t i=0; i<count; ++i)
	{
		KeyMap.push_back(map[i]);
	}
}

void CSceneNodeAnimatorCameraFPS::setKeyMap(const core::vector<SKeyMap>& keymap)
{
	KeyMap=keymap;
}

const core::vector<SKeyMap>& CSceneNodeAnimatorCameraFPS::getKeyMap() const
{
	return KeyMap;
}


//! Sets whether vertical movement should be allowed.
void CSceneNodeAnimatorCameraFPS::setVerticalMovement(bool allow)
{
	NoVerticalMovement = !allow;
}


//! Sets whether the Y axis of the mouse should be inverted.
void CSceneNodeAnimatorCameraFPS::setInvertMouse(bool invert)
{
	if (invert)
		MouseYDirection = -1.0f;
	else
		MouseYDirection = 1.0f;
}


} // namespace scene
} // namespace nbl

