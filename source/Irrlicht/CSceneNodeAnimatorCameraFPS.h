// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __C_SCENE_NODE_ANIMATOR_CAMERA_FPS_H_INCLUDED__
#define __C_SCENE_NODE_ANIMATOR_CAMERA_FPS_H_INCLUDED__

#include "ISceneNodeAnimatorCameraFPS.h"
#include "vector2d.h"
#include "position2d.h"
#include "SKeyMap.h"


namespace irr
{
namespace gui
{
	class ICursorControl;
}

namespace scene
{

	//! Special scene node animator for FPS cameras
	class CSceneNodeAnimatorCameraFPS : public ISceneNodeAnimatorCameraFPS
	{
    protected:
		//! Destructor
		virtual ~CSceneNodeAnimatorCameraFPS();

	public:
		//! Constructor
		CSceneNodeAnimatorCameraFPS(gui::ICursorControl* cursorControl,
			float rotateSpeed = 100.0f, float moveSpeed = .5f, float jumpSpeed=0.f,
			SKeyMap* keyMapArray=0, uint32_t keyMapSize=0, bool noVerticalMovement=false,
			bool invertY=false);

		//! Animates the scene node, currently only works on cameras
		virtual void animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs);

		//! Event receiver
		virtual bool OnEvent(const SEvent& event);

		//! Returns the speed of movement in units per second
		virtual float getMoveSpeed() const;

		//! Sets the speed of movement in units per second
		virtual void setMoveSpeed(float moveSpeed);

		//! Returns the rotation speed
		virtual float getRotateSpeed() const;

		//! Set the rotation speed
		virtual void setRotateSpeed(float rotateSpeed);

		//! Sets the keyboard mapping for this animator (old style)
		//! \param keymap: an vector of keyboard mappings, see SKeyMap
		//! \param count: the size of the keyboard map vector
		virtual void setKeyMap(SKeyMap *map, uint32_t count);

		//! Sets the keyboard mapping for this animator
		//!	\param keymap The new keymap vector
		virtual void setKeyMap(const core::vector<SKeyMap>& keymap);

		//! Gets the keyboard mapping for this animator
		virtual const core::vector<SKeyMap>& getKeyMap() const;

		//! Sets whether vertical movement should be allowed.
		virtual void setVerticalMovement(bool allow);

		//! Sets whether the Y axis of the mouse should be inverted.
		/** If enabled then moving the mouse down will cause
		the camera to look up. It is disabled by default. */
		virtual void setInvertMouse(bool invert);

		//! This animator will receive events when attached to the active camera
		virtual bool isEventReceiverEnabled() const
		{
			return true;
		}

		//! Returns the type of this animator
		virtual ESCENE_NODE_ANIMATOR_TYPE getType() const
		{
			return ESNAT_CAMERA_FPS;
		}

		//! Creates a clone of this animator.
		/** Please note that you will have to drop
		(IReferenceCounted::drop()) the returned pointer once you're
		done with it. */
		virtual ISceneNodeAnimator* createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager=0);

	private:
		void allKeysUp();

		gui::ICursorControl *CursorControl;

		float MaxVerticalAngle;

		float MoveSpeed;
		float RotateSpeed;
		float JumpSpeed;
		// -1.0f for inverted mouse, defaults to 1.0f
		float MouseYDirection;

		int32_t LastAnimationTime;

		core::vector<SKeyMap> KeyMap;
		core::position2d<float> CenterCursor, CursorPos;

		bool CursorKeys[EKA_COUNT];

		bool firstUpdate;
		bool firstInput;
		bool NoVerticalMovement;
	};

} // end namespace scene
} // end namespace irr

#endif // __C_SCENE_NODE_ANIMATOR_CAMERA_FPS_H_INCLUDED__

