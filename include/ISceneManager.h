// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_SCENE_MANAGER_H_INCLUDED__
#define __NBL_I_SCENE_MANAGER_H_INCLUDED__

#include "nbl/core/core.h"

#include "path.h"
#include "vector3d.h"
#include "dimension2d.h"
#include "SColor.h"
#include "nbl/asset/ICPUMesh.h"

namespace nbl
{
struct SKeyMap;
struct SEvent;
class IrrlichtDevice;

namespace io
{
class IReadFile;
class IWriteFile;
class IFileSystem;
}  // end namespace io

namespace scene
{
class ICameraSceneNode;
class IDummyTransformationSceneNode;
class IMeshSceneNode;
class ISceneNode;
class ISceneNodeAnimator;

//! The Scene Manager manages scene nodes, mesh recources, cameras and all the other stuff.
/** All Scene nodes can be created only here. There is a always growing
	list of scene nodes for lots of purposes: Indoor rendering scene nodes,
	different Camera scene nodes (addCameraSceneNode(), addCameraSceneNodeMaya()), and so on.
	A scene node is a node in the hierachical scene tree. Every scene node
	may have children, which are other scene nodes. Children move relative
	the their parents position. If the parent of a node is not visible, its
	children won't be visible, too. In this way, it is for example easily
	possible to attach a light to a moving car or to place a walking
	character on a moving platform on a moving ship.
	The SceneManager is also able to load 3d mesh files of different
	formats. Take a look at getMesh() to find out what formats are
	supported. If these formats are not enough, use
	addExternalMeshLoader() to add new formats to the engine.
	*/
class ISceneManager : public virtual core::IReferenceCounted
{
public:
    //! Get the video driver.
    /** \return Pointer to the video Driver.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
    virtual video::IVideoDriver* getVideoDriver() = 0;

    //! Adds a camera scene node to the scene tree and sets it as active camera.
    /** This camera does not react on user input like for example the one created with
		addCameraSceneNodeFPS(). If you want to move or animate it, use animators or the
		ISceneNode::setPosition(), ICameraSceneNode::setTarget() etc methods.
		By default, a camera's look at position (set with setTarget()) and its scene node
		rotation (set with setRotation()) are independent. If you want to be able to
		control the direction that the camera looks by using setRotation() then call
		ICameraSceneNode::bindTargetAndRotation(true) on it.
		\param position: Position of the space relative to its parent where the camera will be placed.
		\param lookat: Position where the camera will look at. Also known as target.
		\param parent: Parent scene node of the camera. Can be null. If the parent moves,
		the camera will move too.
		\param id: id of the camera. This id can be used to identify the camera.
		\param makeActive Flag whether this camera should become the active one.
		Make sure you always have one active camera.
		\return Pointer to interface to camera if successful, otherwise 0.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
    virtual ICameraSceneNode* addCameraSceneNode(IDummyTransformationSceneNode* parent = 0,
        const core::vector3df& position = core::vector3df(0, 0, 0),
        const core::vectorSIMDf& lookat = core::vectorSIMDf(0, 0, 100),
        int32_t id = -1, bool makeActive = true) = 0;

    //! Adds a maya style user controlled camera scene node to the scene tree.
    /** This is a standard camera with an animator that provides mouse control similar
		to camera in the 3D Software Maya by Alias Wavefront.
		The camera does not react on setPosition anymore after applying this animator. Instead
		use setTarget, to fix the target the camera the camera hovers around. And setDistance
		to set the current distance from that target, i.e. the radius of the orbit the camera
		hovers on.
		\param parent: Parent scene node of the camera. Can be null.
		\param rotateSpeed: Rotation speed of the camera.
		\param zoomSpeed: Zoom speed of the camera.
		\param translationSpeed: TranslationSpeed of the camera.
		\param id: id of the camera. This id can be used to identify the camera.
		\param distance Initial distance of the camera from the object
		\param makeActive Flag whether this camera should become the active one.
		Make sure you always have one active camera.
		\return Returns a pointer to the interface of the camera if successful, otherwise 0.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
    virtual ICameraSceneNode* addCameraSceneNodeMaya(IDummyTransformationSceneNode* parent = 0,
        float rotateSpeed = -1500.f, float zoomSpeed = 200.f,
        float translationSpeed = 1500.f, int32_t id = -1, float distance = 70.f,
        bool makeActive = true) = 0;

    virtual ICameraSceneNode* addCameraSceneNodeModifiedMaya(IDummyTransformationSceneNode* parent = 0,
        float rotateSpeed = -1500.f, float zoomSpeed = 200.f,
        float translationSpeed = 1500.f, int32_t id = -1, float distance = 70.f,
        float scrlZoomSpeed = 10.0f, bool scroolWithRMB = false,
        bool makeActive = true) = 0;

    //! Adds a camera scene node with an animator which provides mouse and keyboard control appropriate for first person shooters (FPS).
    /** This FPS camera is intended to provide a demonstration of a
		camera that behaves like a typical First Person Shooter. It is
		useful for simple demos and prototyping but is not intended to
		provide a full solution for a production quality game. It binds
		the camera scene node rotation to the look-at target; @see
		ICameraSceneNode::bindTargetAndRotation(). With this camera,
		you look with the mouse, and move with cursor keys. If you want
		to change the key layout, you can specify your own keymap. For
		example to make the camera be controlled by the cursor keys AND
		the keys W,A,S, and D, do something like this:
		\code
		 SKeyMap keyMap[8];
		 keyMap[0].Action = EKA_MOVE_FORWARD;
		 keyMap[0].KeyCode = KEY_UP;
		 keyMap[1].Action = EKA_MOVE_FORWARD;
		 keyMap[1].KeyCode = KEY_KEY_W;

		 keyMap[2].Action = EKA_MOVE_BACKWARD;
		 keyMap[2].KeyCode = KEY_DOWN;
		 keyMap[3].Action = EKA_MOVE_BACKWARD;
		 keyMap[3].KeyCode = KEY_KEY_S;

		 keyMap[4].Action = EKA_STRAFE_LEFT;
		 keyMap[4].KeyCode = KEY_LEFT;
		 keyMap[5].Action = EKA_STRAFE_LEFT;
		 keyMap[5].KeyCode = KEY_KEY_A;

		 keyMap[6].Action = EKA_STRAFE_RIGHT;
		 keyMap[6].KeyCode = KEY_RIGHT;
		 keyMap[7].Action = EKA_STRAFE_RIGHT;
		 keyMap[7].KeyCode = KEY_KEY_D;

		camera = sceneManager->addCameraSceneNodeFPS(0, 100, 500, -1, keyMap, 8);
		\endcode
		\param parent: Parent scene node of the camera. Can be null.
		\param rotateSpeed: Speed in degress with which the camera is
		rotated. This can be done only with the mouse.
		\param moveSpeed: Speed in units per millisecond with which
		the camera is moved. Movement is done with the cursor keys.
		\param id: id of the camera. This id can be used to identify
		the camera.
		\param keyMapArray: Optional pointer to an array of a keymap,
		specifying what keys should be used to move the camera. If this
		is null, the default keymap is used. You can define actions
		more then one time in the array, to bind multiple keys to the
		same action.
		\param keyMapSize: Amount of items in the keymap array.
		\param noVerticalMovement: Setting this to true makes the
		camera only move within a horizontal plane, and disables
		vertical movement as known from most ego shooters. Default is
		'false', with which it is possible to fly around in space, if
		no gravity is there.
		\param jumpSpeed: Speed with which the camera is moved when
		jumping.
		\param invertMouse: Setting this to true makes the camera look
		up when the mouse is moved down and down when the mouse is
		moved up, the default is 'false' which means it will follow the
		movement of the mouse cursor.
		\param makeActive Flag whether this camera should become the active one.
		Make sure you always have one active camera.
		\return Pointer to the interface of the camera if successful,
		otherwise 0. This pointer should not be dropped. See
		IReferenceCounted::drop() for more information. */
    virtual ICameraSceneNode* addCameraSceneNodeFPS(IDummyTransformationSceneNode* parent = 0,
        float rotateSpeed = 100.0f, float moveSpeed = 0.5f, int32_t id = -1,
        SKeyMap* keyMapArray = 0, int32_t keyMapSize = 0, bool noVerticalMovement = false,
        float jumpSpeed = 0.f, bool invertMouse = false,
        bool makeActive = true) = 0;

    //! Get the current active camera.
    /** \return The active camera is returned. Note that this can
		be NULL, if there was no camera created yet.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
    virtual ICameraSceneNode* getActiveCamera() const = 0;

    //! Sets the currently active camera.
    /** The previous active camera will be deactivated.
		\param camera: The new camera which should be active. */
    virtual void setActiveCamera(ICameraSceneNode* camera) = 0;

    //! Posts an input event to the environment.
    /** Usually you do not have to
		use this method, it is used by the internal engine. */
    virtual bool receiveIfEventReceiverDidNotAbsorb(const SEvent& event) = 0;

    //! Clears the whole scene.
    /** All scene nodes are removed. */
    virtual void clear() = 0;
};

}  // end namespace scene
}  // end namespace nbl

#endif
