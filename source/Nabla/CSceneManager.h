// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_SCENE_MANAGER_H_INCLUDED__
#define __NBL_C_SCENE_MANAGER_H_INCLUDED__

#include "ISceneManager.h"
#include "ISceneNode.h"

namespace nbl
{
namespace scene
{

/*!
	The Scene Manager manages scene nodes, mesh recources, cameras and all the other stuff.
*/
class CSceneManager : public ISceneManager, public ISceneNode
{
	public:
        //!
		virtual void OnAnimate(uint32_t timeMs);

		//! renders the node.
		virtual void render();

		//! Adds a camera scene node to the tree and sets it as active camera.
		//! \param position: Position of the space relative to its parent where the camera will be placed.
		//! \param lookat: Position where the camera will look at. Also known as target.
		//! \param parent: Parent scene node of the camera. Can be null. If the parent moves,
		//! the camera will move too.
		//! \return Pointer to interface to camera
		virtual ICameraSceneNode* addCameraSceneNode(IDummyTransformationSceneNode* parent = 0,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vectorSIMDf & lookat = core::vectorSIMDf(0,0,100),
			int32_t id=-1, bool makeActive=true) override;

		//! Adds a camera scene node which is able to be controlle with the mouse similar
		//! like in the 3D Software Maya by Alias Wavefront.
		//! The returned pointer must not be dropped.
		virtual ICameraSceneNode* addCameraSceneNodeMaya(IDummyTransformationSceneNode* parent=0,
			float rotateSpeed=-1500.f, float zoomSpeed=200.f,
			float translationSpeed=1500.f, int32_t id=-1, float distance=70.f,
			bool makeActive=true);

		virtual ICameraSceneNode* addCameraSceneNodeModifiedMaya(IDummyTransformationSceneNode* parent = 0,
			float rotateSpeed = -1500.f, float zoomSpeed = 200.f,
			float translationSpeed = 1500.f, int32_t id = -1, float distance = 70.f,
			float scrlZoomSpeed = 10.0f, bool zoomlWithRMB = false,
			bool makeActive = true) override;

		//! Adds a camera scene node which is able to be controled with the mouse and keys
		//! like in most first person shooters (FPS):
		virtual ICameraSceneNode* addCameraSceneNodeFPS(IDummyTransformationSceneNode* parent = 0,
			float rotateSpeed = 100.0f, float moveSpeed = .5f, int32_t id=-1,
			SKeyMap* keyMapArray=0, int32_t keyMapSize=0,
			bool noVerticalMovement=false, float jumpSpeed = 0.f,
			bool invertMouseY=false, bool makeActive=true);
	};

} // end namespace video
} // end namespace scene

#endif

