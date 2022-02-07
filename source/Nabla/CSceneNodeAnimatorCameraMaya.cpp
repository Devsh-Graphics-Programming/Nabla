// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CSceneNodeAnimatorCameraMaya.h"
#include "ICursorControl.h"
#include "ICameraSceneNode.h"
#include "SViewFrustum.h"
#include "ISceneManager.h"

namespace nbl
{
namespace scene
{
//! constructor
CSceneNodeAnimatorCameraMaya::CSceneNodeAnimatorCameraMaya(gui::ICursorControl* cursor,
    float rotateSpeed, float zoomSpeed, float translateSpeed, float distance)
    : CursorControl(cursor), OldCamera(0), MousePos(0.5f, 0.5f),
      ZoomSpeed(zoomSpeed), RotateSpeed(rotateSpeed), TranslateSpeed(translateSpeed),
      CurrentZoom(distance), RotX(0.0f), RotY(0.0f),
      Zooming(false), Rotating(false), Moving(false), Translating(false)
{
#ifdef _NBL_DEBUG
    setDebugName("CSceneNodeAnimatorCameraMaya");
#endif

    if(CursorControl)
    {
        CursorControl->grab();
        MousePos = CursorControl->getRelativePosition();
    }

    allKeysUp();
}

//! destructor
CSceneNodeAnimatorCameraMaya::~CSceneNodeAnimatorCameraMaya()
{
    if(CursorControl)
        CursorControl->drop();
}

//! It is possible to send mouse and key events to the camera. Most cameras
//! may ignore this input, but camera scene nodes which are created for
//! example with scene::ISceneManager::addMayaCameraSceneNode or
//! scene::ISceneManager::addMeshViewerCameraSceneNode, may want to get this input
//! for changing their position, look at target or whatever.
bool CSceneNodeAnimatorCameraMaya::OnEvent(const SEvent& event)
{
    if(event.EventType != EET_MOUSE_INPUT_EVENT)
        return false;

    switch(event.MouseInput.Event)
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
            MousePos = CursorControl->getRelativePosition();
            break;
        case EMIE_MOUSE_WHEEL:
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
void CSceneNodeAnimatorCameraMaya::animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs)
{
    //Alt + LM = Rotate around camera pivot
    //Alt + LM + MM = Dolly forth/back in view direction (speed % distance camera pivot - max distance to pivot)
    //Alt + MM = Move on camera plane (Screen center is about the mouse pointer, depending on move speed)

    if(!node /* || node->getType() != ESNT_CAMERA*/)
        return;

    ICameraSceneNode* camera = static_cast<ICameraSceneNode*>(node);

    // If the camera isn't the active camera, and receiving input, then don't process it.
    if(!camera->isInputReceiverEnabled())
        return;
    /*
	scene::ISceneManager * smgr = camera->getSceneManager();
	if (smgr && smgr->getActiveCamera() != camera)
		return;
*/
    if(OldCamera != camera)
    {
        LastCameraTarget = OldTarget = camera->getTarget();
        OldCamera = camera;
    }
    else
    {
        OldTarget += camera->getTarget() - LastCameraTarget;
    }

    float nRotX = RotX;
    float nRotY = RotY;
    float nZoom = CurrentZoom;

    if((isMouseKeyDown(0) && isMouseKeyDown(2)) || isMouseKeyDown(1))
    {
        if(!Zooming)
        {
            ZoomStart = MousePos;
            Zooming = true;
        }
        else
        {
            const float targetMinDistance = 0.1f;
            nZoom += (ZoomStart.X - MousePos.X) * ZoomSpeed;

            if(nZoom < targetMinDistance)  // jox: fixed bug: bounce back when zooming to close
                nZoom = targetMinDistance;
        }
    }
    else if(Zooming)
    {
        const float old = CurrentZoom;
        CurrentZoom = CurrentZoom + (ZoomStart.X - MousePos.X) * ZoomSpeed;
        nZoom = CurrentZoom;

        if(nZoom < 0)
            nZoom = CurrentZoom = old;
        Zooming = false;
    }

    // Translation ---------------------------------

    core::vector3df_SIMD translate(OldTarget);

    core::vector3df_SIMD target, upVector;
    upVector = camera->getUpVector();
    target = camera->getTarget();

    core::vector3df_SIMD pos, tvectX;
    pos.getAsVector3df() = camera->getPosition();
    tvectX = pos - target;
    if(camera->getLeftHanded())
        tvectX = normalize(cross(tvectX, upVector));
    else
        tvectX = normalize(cross(upVector, tvectX));

    const SViewFrustum* const va = camera->getViewFrustum();
    core::vector3df_SIMD tvectY = (va->getFarLeftDown() - va->getFarRightDown());
    tvectY = normalize(cross(tvectY, upVector.Y > 0 ? pos - target : target - pos));

    if(isMouseKeyDown(2) && !Zooming)
    {
        if(!Translating)
        {
            TranslateStart = MousePos;
            Translating = true;
        }
        else
        {
            translate += tvectX * (TranslateStart.X - MousePos.X) * TranslateSpeed +
                tvectY * (TranslateStart.Y - MousePos.Y) * TranslateSpeed;
        }
    }
    else if(Translating)
    {
        translate += tvectX * (TranslateStart.X - MousePos.X) * TranslateSpeed +
            tvectY * (TranslateStart.Y - MousePos.Y) * TranslateSpeed;
        OldTarget = translate;
        Translating = false;
    }

    // Rotation ------------------------------------

    auto getValueDependentOnHandOrientation = [&](const float& expression) {
        return core::mix<float, bool>(-expression, expression, camera->getLeftHanded());
    };

    if(isMouseKeyDown(0) && !Zooming)
    {
        if(!Rotating)
        {
            RotateStart = MousePos;
            Rotating = true;
            nRotX = RotX;
            nRotY = RotY;
        }
        else
        {
            nRotX += getValueDependentOnHandOrientation((RotateStart.X - MousePos.X) * RotateSpeed);
            nRotY += getValueDependentOnHandOrientation((RotateStart.Y - MousePos.Y) * RotateSpeed);
        }
    }
    else if(Rotating)
    {
        RotX += getValueDependentOnHandOrientation((RotateStart.X - MousePos.X) * RotateSpeed);
        RotY += getValueDependentOnHandOrientation((RotateStart.Y - MousePos.Y) * RotateSpeed);
        nRotX = RotX;
        nRotY = RotY;
        Rotating = false;
    }

    // Set pos ------------------------------------

    pos = translate;
    pos.X += nZoom;

    pos.rotateXYByRAD(core::radians(nRotY), translate);
    pos.rotateXZByRAD(-core::radians(nRotX), translate);

    camera->setPosition(pos.getAsVector3df());
    camera->setTarget(translate.getAsVector3df());

    // Rotation Error ----------------------------

    // jox: fixed bug: jitter when rotating to the top and bottom of y
    pos.set(0, 1, 0);
    pos.rotateXYByRAD(-core::radians(nRotY));
    pos.rotateXZByRAD(core::PI<float>() - core::radians(nRotX));
    camera->setUpVector(pos);
    LastCameraTarget = camera->getTarget();
}

bool CSceneNodeAnimatorCameraMaya::isMouseKeyDown(int32_t key) const
{
    return MouseKeys[key];
}

void CSceneNodeAnimatorCameraMaya::allKeysUp()
{
    for(int32_t i = 0; i < 3; ++i)
        MouseKeys[i] = false;
}

//! Sets the rotation speed
void CSceneNodeAnimatorCameraMaya::setRotateSpeed(float speed)
{
    RotateSpeed = speed;
}

//! Sets the movement speed
void CSceneNodeAnimatorCameraMaya::setMoveSpeed(float speed)
{
    TranslateSpeed = speed;
}

//! Sets the zoom speed
void CSceneNodeAnimatorCameraMaya::setZoomSpeed(float speed)
{
    ZoomSpeed = speed;
}

//! Set the distance
void CSceneNodeAnimatorCameraMaya::setDistance(float distance)
{
    CurrentZoom = distance;
}

//! Gets the rotation speed
float CSceneNodeAnimatorCameraMaya::getRotateSpeed() const
{
    return RotateSpeed;
}

// Gets the movement speed
float CSceneNodeAnimatorCameraMaya::getMoveSpeed() const
{
    return TranslateSpeed;
}

//! Gets the zoom speed
float CSceneNodeAnimatorCameraMaya::getZoomSpeed() const
{
    return ZoomSpeed;
}

//! Returns the current distance, i.e. orbit radius
float CSceneNodeAnimatorCameraMaya::getDistance() const
{
    return CurrentZoom;
}

}  // end namespace
}  // end namespace
