// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_SCENE_NODE_ANIMATOR_CAMERA_MAYA_H_INCLUDED__
#define __NBL_I_SCENE_NODE_ANIMATOR_CAMERA_MAYA_H_INCLUDED__

#include "ISceneNodeAnimator.h"

namespace nbl
{
namespace scene
{
//! Special scene node animator for Maya-style cameras
/** This scene node animator can be attached to a camera to make it act like a 3d
	modelling tool.
	The camera is moving relative to the target with the mouse, by pressing either
	of the three buttons.
	In order to move the camera, set a new target for the camera. The distance defines
	the current orbit radius the camera moves on. Distance can be changed via the setter
	or by mouse events.
	*/
class ISceneNodeAnimatorCameraMaya : public ISceneNodeAnimator
{
public:
    //! Returns the speed of movement
    virtual float getMoveSpeed() const = 0;

    //! Sets the speed of movement
    virtual void setMoveSpeed(float moveSpeed) = 0;

    //! Returns the rotation speed
    virtual float getRotateSpeed() const = 0;

    //! Set the rotation speed
    virtual void setRotateSpeed(float rotateSpeed) = 0;

    //! Returns the zoom speed
    virtual float getZoomSpeed() const = 0;

    //! Set the zoom speed
    virtual void setZoomSpeed(float zoomSpeed) = 0;

    //! Returns the current distance, i.e. orbit radius
    virtual float getDistance() const = 0;

    //! Set the distance
    virtual void setDistance(float distance) = 0;
};

}  // end namespace scene
}  // end namespace nbl

#endif
