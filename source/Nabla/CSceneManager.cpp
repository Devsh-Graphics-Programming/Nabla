// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "BuildConfigOptions.h"
#include "CSceneManager.h"
#include "IFileSystem.h"
#include "IReadFile.h"
#include "IWriteFile.h"
#include "IrrlichtDevice.h"

#include "nbl_os.h"

#include "CCameraSceneNode.h"

#include "CSceneNodeAnimatorCameraFPS.h"
#include "CSceneNodeAnimatorCameraMaya.h"
#include "CSceneNodeAnimatorCameraModifiedMaya.h"

namespace nbl
{
namespace scene
{

//! constructor
CSceneManager::CSceneManager(IrrlichtDevice* device, video::IVideoDriver* driver, nbl::ITimer* timer, io::IFileSystem* fs,
		gui::ICursorControl* cursorControl)
: ISceneNode(0, 0), Driver(driver), Timer(timer), Device(device),
	CursorControl(cursorControl), ActiveCamera(0)
{
	#ifdef _NBL_DEBUG
	ISceneManager::setDebugName("CSceneManager ISceneManager");
	#endif

	if (Driver)
		Driver->grab();
	if (CursorControl)
		CursorControl->grab();
}


//! destructor
CSceneManager::~CSceneManager()
{
	if (CursorControl)
		CursorControl->drop();

	if (ActiveCamera)
		ActiveCamera->drop();
	ActiveCamera = 0;

	// remove all nodes and animators before dropping the driver
	// as render targets may be destroyed twice

	removeAll();
	removeAnimators();

	if (Driver)
		Driver->drop();
}


//! Adds a camera scene node to the tree and sets it as active camera.
//! \param position: Position of the space relative to its parent where the camera will be placed.
//! \param lookat: Position where the camera will look at. Also known as target.
//! \param parent: Parent scene node of the camera. Can be null. If the parent moves,
//! the camera will move too.
//! \return Returns pointer to interface to camera
ICameraSceneNode* CSceneManager::addCameraSceneNode(IDummyTransformationSceneNode* parent,
	const core::vector3df& position, const core::vectorSIMDf& lookat, int32_t id,
	bool makeActive)
{
	if (!parent)
		parent = this;

	ICameraSceneNode* node = new CCameraSceneNode(parent, this, id, position, lookat);

	if (makeActive)
		setActiveCamera(node);
	node->drop();

	return node;
}


//! Adds a camera scene node which is able to be controlled with the mouse similar
//! to in the 3D Software Maya by Alias Wavefront.
//! The returned pointer must not be dropped.
ICameraSceneNode* CSceneManager::addCameraSceneNodeMaya(IDummyTransformationSceneNode* parent,
	float rotateSpeed, float zoomSpeed, float translationSpeed, int32_t id, float distance,
	bool makeActive)
{
	ICameraSceneNode* node = addCameraSceneNode(parent, core::vector3df(),
			core::vectorSIMDf(0,0,100), id, makeActive);
	if (node)
	{
		ISceneNodeAnimator* anm = new CSceneNodeAnimatorCameraMaya(CursorControl,
			rotateSpeed, zoomSpeed, translationSpeed, distance);

		node->addAnimator(anm);
		anm->drop();
	}

	return node;
}

ICameraSceneNode* CSceneManager::addCameraSceneNodeModifiedMaya(IDummyTransformationSceneNode* parent,
	float rotateSpeed, float zoomSpeed,
	float translationSpeed, int32_t id, float distance,
	float scrlZoomSpeedMultiplier, bool zoomWithRMB,
	bool makeActive)
{
	ICameraSceneNode* node = addCameraSceneNode(parent, core::vector3df(),
		core::vectorSIMDf(0, 0, 100), id, makeActive);
	if (node)
	{
		ISceneNodeAnimator* anm = new CSceneNodeAnimatorCameraModifiedMaya(CursorControl,
			rotateSpeed, zoomSpeed, translationSpeed, distance, scrlZoomSpeedMultiplier, zoomWithRMB);

		node->addAnimator(anm);
		anm->drop();
	}

	return node;
}


//! Adds a camera scene node which is able to be controlled with the mouse and keys
//! like in most first person shooters (FPS):
ICameraSceneNode* CSceneManager::addCameraSceneNodeFPS(IDummyTransformationSceneNode* parent,
	float rotateSpeed, float moveSpeed, int32_t id, SKeyMap* keyMapArray,
	int32_t keyMapSize, bool noVerticalMovement, float jumpSpeed,
	bool invertMouseY, bool makeActive)
{
	ICameraSceneNode* node = addCameraSceneNode(parent, core::vector3df(),
			core::vectorSIMDf(0,0,100), id, makeActive);
	if (node)
	{
		ISceneNodeAnimator* anm = new CSceneNodeAnimatorCameraFPS(CursorControl,
				rotateSpeed, moveSpeed, jumpSpeed,
				keyMapArray, keyMapSize, noVerticalMovement, invertMouseY);

		// Bind the node's rotation to its target. This is consistent with 1.4.2 and below.
		node->bindTargetAndRotation(true);
		node->addAnimator(anm);
		anm->drop();
	}

	return node;
}


//! Returns the current active camera.
//! \return The active camera is returned. Note that this can be NULL, if there
//! was no camera created yet.
ICameraSceneNode* CSceneManager::getActiveCamera() const
{
	return ActiveCamera;
}


//! Sets the active camera. The previous active camera will be deactivated.
//! \param camera: The new camera which should be active.
void CSceneManager::setActiveCamera(ICameraSceneNode* camera)
{
	if (camera)
		camera->grab();
	if (ActiveCamera)
		ActiveCamera->drop();

	ActiveCamera = camera;
}


//! renders the node.
void CSceneManager::render()
{
}

//!
void CSceneManager::OnAnimate(uint32_t timeMs)
{
    size_t prevSize = Children.size();
    for (size_t i=0; i<prevSize;)
    {
        IDummyTransformationSceneNode* tmpChild = Children[i];
        if (tmpChild->isISceneNode())
            static_cast<ISceneNode*>(tmpChild)->OnAnimate(timeMs);
        else
            OnAnimate_static(tmpChild,timeMs);

        if (Children[i]>tmpChild)
            prevSize = Children.size();
        else
            i++;
    }
}

//! Posts an input event to the environment. Usually you do not have to
//! use this method, it is used by the internal engine.
bool CSceneManager::receiveIfEventReceiverDidNotAbsorb(const SEvent& event)
{
	bool ret = false;
	ICameraSceneNode* cam = getActiveCamera();
	if (cam)
		ret = cam->OnEvent(event);

	return ret;
}


//! Removes all children of this scene node
void CSceneManager::removeAll()
{
	ISceneNode::removeAll();
	setActiveCamera(0);
}


//! Clears the whole scene. All scene nodes are removed.
void CSceneManager::clear()
{
	removeAll();
}


} // end namespace scene
} // end namespace nbl

