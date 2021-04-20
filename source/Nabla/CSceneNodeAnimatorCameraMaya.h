// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_SCENE_NODE_ANIMATOR_CAMERA_MAYA_H_INCLUDED__
#define __NBL_C_SCENE_NODE_ANIMATOR_CAMERA_MAYA_H_INCLUDED__

#include "ISceneNodeAnimatorCameraMaya.h"
#include "ICameraSceneNode.h"
#include "vector2d.h"

namespace nbl
{

namespace gui
{
	class ICursorControl;
}

namespace scene
{

	//! Special scene node animator for FPS cameras
	/** This scene node animator can be attached to a camera to make it act
	like a 3d modelling tool camera
	*/
	class CSceneNodeAnimatorCameraMaya : public ISceneNodeAnimatorCameraMaya
	{
    protected:
		//! Destructor
		virtual ~CSceneNodeAnimatorCameraMaya();

	public:
		//! Constructor
		CSceneNodeAnimatorCameraMaya(gui::ICursorControl* cursor, float rotateSpeed = -1500.f,
			float zoomSpeed = 200.f, float translationSpeed = 1500.f, float distance=70.f);

		//! Animates the scene node, currently only works on cameras
		virtual void animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs);

		//! Event receiver
		virtual bool OnEvent(const SEvent& event);

		//! Returns the speed of movement in units per millisecond
		virtual float getMoveSpeed() const;

		//! Sets the speed of movement in units per millisecond
		virtual void setMoveSpeed(float moveSpeed);

		//! Returns the rotation speed
		virtual float getRotateSpeed() const;

		//! Set the rotation speed
		virtual void setRotateSpeed(float rotateSpeed);

		//! Returns the zoom speed
		virtual float getZoomSpeed() const;

		//! Set the zoom speed
		virtual void setZoomSpeed(float zoomSpeed);

		//! Returns the current distance, i.e. orbit radius
		virtual float getDistance() const;

		//! Set the distance
		virtual void setDistance(float distance);

		//! This animator will receive events when attached to the active camera
		virtual bool isEventReceiverEnabled() const
		{
			return true;
		}

	private:

		void allKeysUp();
		void animate();
		bool isMouseKeyDown(int32_t key) const;

		bool MouseKeys[3];

		gui::ICursorControl *CursorControl;
		scene::ICameraSceneNode* OldCamera;
		core::vectorSIMDf OldTarget;
		core::vectorSIMDf LastCameraTarget;	// to find out if the camera target was moved outside this animator
		core::position2df RotateStart;
		core::position2df ZoomStart;
		core::position2df TranslateStart;
		core::position2df MousePos;
		float ZoomSpeed;
		float RotateSpeed;
		float TranslateSpeed;
		float CurrentZoom;
		float RotX, RotY;
		bool Zooming;
		bool Rotating;
		bool Moving;
		bool Translating;
	};

} // end namespace scene
} // end namespace nbl

#endif

