// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CSceneNodeAnimatorCameraModifiedMaya.h"
#include "ICursorControl.h"
#include "ICameraSceneNode.h"
#include "SViewFrustum.h"
#include "ISceneManager.h"

#include <iostream>

namespace nbl
{
namespace scene
{

//! constructor
CSceneNodeAnimatorCameraModifiedMaya::CSceneNodeAnimatorCameraModifiedMaya(gui::ICursorControl* cursor,
	float rotateSpeed, float zoomSpeed, float translateSpeed, float distance,
	float scrollZoomSpeed, bool zoomWithRMB)
	: CursorControl(cursor), OldCamera(0), MousePos(0.5f, 0.5f),
	ZoomSpeed(zoomSpeed), RotateSpeed(rotateSpeed), TranslateSpeed(translateSpeed),
	CurrentZoom(distance), RotX(0.0f), RotY(0.0f),
	ZoomDelta(0.0f), ZoomWithRMB(zoomWithRMB), StepZooming(false), ScrllZoomSpeed(-scrollZoomSpeed),
	Zooming(false), Rotating(false), Moving(false), Translating(false), ShiftTranslating(false), MouseShift(false)
{
#ifdef _NBL_DEBUG
	setDebugName("CSceneNodeAnimatorCameraModifiedMaya");
#endif

	if (CursorControl)
	{
		CursorControl->grab();
		MousePos = CursorControl->getRelativePosition();
	}

	allKeysUp();
}

//! destructor
CSceneNodeAnimatorCameraModifiedMaya::~CSceneNodeAnimatorCameraModifiedMaya()
{
	if (CursorControl)
		CursorControl->drop();
}


void CSceneNodeAnimatorCameraModifiedMaya::setZoomAndRotationBasedOnTargetAndPosition(const core::vectorSIMDf& position, const core::vectorSIMDf& target)
{
	core::vectorSIMDf relativeTarget = position - target;
	// core::vector3df relativeRotation = relativeTarget.getAsVector3df().getHorizontalAngle();
	float angleX = 0.0f;
	float angleY = 0.0f;

	const double tmp = core::degrees(atan2((double)relativeTarget.z, (double)relativeTarget.x));
	angleX = tmp;
	if (angleX < 0)
		angleX += 360;
	if (angleX >= 360)
		angleX -= 360;

	const double z1 = core::sqrt(relativeTarget.x*relativeTarget.x + relativeTarget.z*relativeTarget.z);
	angleY = (core::degrees(atan2((double)relativeTarget.y, (double)z1)) - 0.0);
	if (angleY < 0)
		angleY += 360;
	if (angleY >= 360)
		angleY -= 360;

	CurrentZoom = core::sqrt(core::dot(relativeTarget, relativeTarget)[0]); 
	OldTarget = target;
	RotX = 360-angleX;
	RotY = angleY;
}

//! It is possible to send mouse and key events to the camera. Most cameras
//! may ignore this input, but camera scene nodes which are created for
//! example with scene::ISceneManager::addMayaCameraSceneNode or
//! scene::ISceneManager::addMeshViewerCameraSceneNode, may want to get this input
//! for changing their position, look at target or whatever.
bool CSceneNodeAnimatorCameraModifiedMaya::OnEvent(const SEvent& event)
{
	if (event.EventType != EET_MOUSE_INPUT_EVENT)
		return false;

	MouseShift = event.MouseInput.Shift;

	switch (event.MouseInput.Event)
	{
	case EMIE_LMOUSE_PRESSED_DOWN:
		MouseKeys[0] = true;
		break;
	case EMIE_RMOUSE_PRESSED_DOWN:
		MouseKeys[2] = true;
		break;
	case EMIE_MMOUSE_PRESSED_DOWN:
		MouseKeys[1] = true;
		break;
	case EMIE_LMOUSE_LEFT_UP:
		MouseKeys[0] = false;
		break;
	case EMIE_RMOUSE_LEFT_UP:
		MouseKeys[2] = false;
		break;
	case EMIE_MMOUSE_LEFT_UP:
		MouseKeys[1] = false;
		break;
	case EMIE_MOUSE_MOVED:
	{
		// Reset mouse-keys when they are no longer pressed (might be missed when it happened outside Irrlicht)
		if (!event.MouseInput.isLeftPressed())
			MouseKeys[0] = false;
		if (!event.MouseInput.isMiddlePressed())
			MouseKeys[1] = false;
		if (!event.MouseInput.isRightPressed())
			MouseKeys[2] = false;

		MousePos = CursorControl->getRelativePosition();
        break;
	}
	case EMIE_MOUSE_WHEEL:
	{
		if (!StepZooming && !Zooming)
		{
			StepZooming = true;
			ZoomDelta = event.MouseInput.Wheel * ScrllZoomSpeed;
		}
        break;
	}
				
	case EMIE_LMOUSE_DOUBLE_CLICK:
	case EMIE_RMOUSE_DOUBLE_CLICK:
	case EMIE_MMOUSE_DOUBLE_CLICK:
	case EMIE_LMOUSE_TRIPLE_CLICK:
	case EMIE_RMOUSE_TRIPLE_CLICK:
	case EMIE_MMOUSE_TRIPLE_CLICK:
	case EMIE_COUNT:
		return false;
	}
	return true;
}


//! OnAnimate() is called just before rendering the whole scene.
void CSceneNodeAnimatorCameraModifiedMaya::animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs)
{
	if (!node/* || node->getType() != ESNT_CAMERA*/)
		return;

	ICameraSceneNode * camera = static_cast<ICameraSceneNode*>(node);

	// If the camera isn't the active camera, and receiving input, then don't process it.
	if (!camera->isInputReceiverEnabled())
		return;

	const SViewFrustum* va = camera->getViewFrustum();

	float nRotX = RotX;
	float nRotY = RotY;

	// Check for zooming with right-button
	if (ZoomWithRMB && isMouseKeyDown(2))
	{
		if (!Zooming)
		{
			ZoomDelta = 0.f;
			Zooming = true;
		}
		else
		{
			ZoomDelta = ((ZoomStart.X - MousePos.X) + (ZoomStart.Y - MousePos.Y)) * ZoomSpeed;
					
		}
		ZoomStart = MousePos;
				
	}
	else if (Zooming)
	{
		Zooming = false;
	}

	// Zoom the cam		
	core::vectorSIMDf zoomTarget(0.0f, 0.0f, 0.0f);	// move target to allow further zooming
	if (StepZooming || Zooming)
	{
		if (StepZooming)
			CurrentZoom -= ZoomDelta * ScrllZoomSpeed;
		else
			CurrentZoom += ZoomDelta * ZoomSpeed;

		const float minDistance = 1.0f;
		if (CurrentZoom < minDistance)
		{
			core::vectorSIMDf pos; pos.set(camera->getPosition());
			zoomTarget = core::normalize(camera->getTarget() - pos);
			zoomTarget *= (-CurrentZoom + minDistance);
			CurrentZoom = 1.0f;
		}
		StepZooming = false;
	}

	// Translation ---------------------------------
	core::vectorSIMDf translateTarget(OldTarget);

	core::vectorSIMDf tvectX; tvectX.set(camera->getPosition());
	tvectX -= camera->getTarget();

	core::vectorSIMDf other = tvectX;
	if(camera->getLeftHanded())
		tvectX = core::normalize(core::cross(tvectX,camera->getUpVector()));
	else
		tvectX = core::normalize(core::cross(camera->getUpVector(), tvectX));

	core::vectorSIMDf tvectY = va->getFarLeftDown() - va->getFarRightDown();
	if (camera->getUpVector().Y<0.f)
		other *= -1.f;
	
	if(camera->getLeftHanded())
		tvectY = core::normalize(core::cross(other, tvectY));
	else
		tvectY = core::normalize(core::cross(tvectY, other));

			
	if ((isMouseKeyDown(1) || isMouseKeyDown(2)) && !(StepZooming || Zooming))
	{
		if (!Translating && !ShiftTranslating)
		{
			TranslateStart = MousePos;
					
			if (MouseShift)
				ShiftTranslating = true;
			else
				Translating = true;

		}
		else
		{
			translateTarget += tvectX * (MousePos.X - TranslateStart.X) * TranslateSpeed +
				tvectY * (TranslateStart.Y - MousePos.Y) * TranslateSpeed;
			if (MouseShift != ShiftTranslating)
			{
				OldTarget = translateTarget;
				TranslateStart = MousePos;
						
				if (MouseShift)
				{
					ShiftTranslating = true;
					Translating = false;
				}
				else
				{
					Translating = true;
					ShiftTranslating = false;
				}
			}
		}
	}
	else if (Translating || ShiftTranslating)	// first event after releasing mouse-buttons
	{
		translateTarget += tvectX * (MousePos.X - TranslateStart.X) * TranslateSpeed+
			tvectY * (TranslateStart.Y - MousePos.Y) * TranslateSpeed;
		OldTarget = translateTarget;
				
		Translating = false;
		ShiftTranslating = false;
	}

	// Rotation ------------------------------------
	auto getValueDependentOnHandOrientation = [&](const float& expression)
	{
		return core::mix<float, bool>(-expression, expression, camera->getLeftHanded());
	};

	if (isMouseKeyDown(0) && !(StepZooming || Zooming))
	{
		if (!Rotating)
		{
			RotateStart = MousePos;
					
			Rotating = true;
			nRotX = RotX;
			nRotY = RotY;	
		}
		else
		{
			nRotX += getValueDependentOnHandOrientation((RotateStart.X - MousePos.X) * RotateSpeed);
			nRotY += getValueDependentOnHandOrientation((RotateStart.Y - MousePos.Y) * RotateSpeed) * (camera->getLeftHanded() ? -1.0f : +1.0f);
		}
	}
	else
	{
		if (Rotating)
		{
			RotX += getValueDependentOnHandOrientation((RotateStart.X - MousePos.X) * RotateSpeed);
			RotY += getValueDependentOnHandOrientation((RotateStart.Y - MousePos.Y) * RotateSpeed) * (camera->getLeftHanded() ? -1.0f : +1.0f);
			nRotX = RotX;
			nRotY = RotY;
					
			Rotating = false;
		}
	}

	// Set Pos ------------------------------------

	camera->setTarget((translateTarget + zoomTarget).getAsVector3df());

	if (core::length(zoomTarget)[0]>0.00001f)
		OldTarget = camera->getTarget();


	core::vectorSIMDf position; position.set(camera->getPosition());
	core::vectorSIMDf target = camera->getTarget();

	position.X = CurrentZoom + target.X;
	position.Y = target.Y;
	position.Z = target.Z;
	
	position.rotateXYByRAD(core::radians(nRotY), target);
	position.rotateXZByRAD(-core::radians(nRotX), target);

	camera->setPosition(position.getAsVector3df());

	// Rotation Error ----------------------------

	// jox: fixed bug: jitter when rotating to the top and bottom of y

	core::vectorSIMDf UpVector(0.0f, 1.0f, 0.0f);
    UpVector.rotateXYByRAD(-core::radians(nRotY));
    UpVector.rotateXZByRAD(core::PI<float>()-core::radians(nRotX));

	camera->setUpVector(UpVector);
}


bool CSceneNodeAnimatorCameraModifiedMaya::isMouseKeyDown(int32_t key) const
{
	return MouseKeys[key];
}


void CSceneNodeAnimatorCameraModifiedMaya::allKeysUp()
{
	for (int32_t i = 0; i < 3; ++i)
		MouseKeys[i] = false;
}


//! Sets the rotation speed
void CSceneNodeAnimatorCameraModifiedMaya::setRotateSpeed(float speed)
{
	RotateSpeed = speed;
}


//! Sets the movement speed
void CSceneNodeAnimatorCameraModifiedMaya::setMoveSpeed(float speed)
{
	TranslateSpeed = speed;
}


//! Sets the zoom speed
void CSceneNodeAnimatorCameraModifiedMaya::setZoomSpeed(float speed)
{
	ZoomSpeed = speed;
}

//! Sets the zoom speed
void CSceneNodeAnimatorCameraModifiedMaya::setStepZoomSpeed(float speed)
{
	ScrllZoomSpeed = speed;
}

//! Set the distance
void CSceneNodeAnimatorCameraModifiedMaya::setDistance(float distance)
{
	CurrentZoom = distance;
}


//! Gets the rotation speed
float CSceneNodeAnimatorCameraModifiedMaya::getRotateSpeed() const
{
	return RotateSpeed;
}


// Gets the movement speed
float CSceneNodeAnimatorCameraModifiedMaya::getMoveSpeed() const
{
	return TranslateSpeed;
}


//! Gets the zoom speed
float CSceneNodeAnimatorCameraModifiedMaya::getZoomSpeed() const
{
	return ZoomSpeed;
}

//! Gets the step zoom speed
float CSceneNodeAnimatorCameraModifiedMaya::getStepZoomSpeed() const
{
	return ScrllZoomSpeed;
}

//! Returns the current distance, i.e. orbit radius
float CSceneNodeAnimatorCameraModifiedMaya::getDistance() const
{
	return CurrentZoom;
}

void CSceneNodeAnimatorCameraModifiedMaya::toggleZoomWithRightButton()
{
	ZoomWithRMB = !ZoomWithRMB;
}

} // end namespace
} // end namespace

