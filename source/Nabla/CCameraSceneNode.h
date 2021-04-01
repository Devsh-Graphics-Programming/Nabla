// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_CAMERA_SCENE_NODE_H_INCLUDED__
#define __NBL_C_CAMERA_SCENE_NODE_H_INCLUDED__

#include "ICameraSceneNode.h"
#include "SViewFrustum.h"
#include "matrix4x3.h"

namespace nbl
{
namespace scene
{

class CCameraSceneNode : public ICameraSceneNode
{
	public:
		//! constructor
		CCameraSceneNode(	IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
							const core::vector3df& position = core::vector3df(0,0,0),
							const core::vectorSIMDf& lookat = core::vectorSIMDf(0,0,100));

		//! Sets the projection matrix of the camera.
		virtual void setProjectionMatrix(const core::matrix4SIMD& projection);

		//! Gets the current view matrix of the camera
		//! \return Returns the current view matrix of the camera.
		virtual const core::matrix3x4SIMD& getViewMatrix() const override;

		virtual const core::matrix4SIMD& getConcatenatedMatrix() const override {return concatMatrix;}

		//! It is possible to send mouse and key events to the camera. Most cameras
		//! may ignore this input, but camera scene nodes which are created for
		//! example with scene::ISceneManager::addMayaCameraSceneNode or
		//! scene::ISceneManager::addMeshViewerCameraSceneNode, may want to get this input
		//! for changing their position, look at target or whatever.
		virtual bool OnEvent(const SEvent& event);

		//! Sets the look at target of the camera
		/** If the camera's target and rotation are bound ( @see bindTargetAndRotation() )
		then calling this will also change the camera's scene node rotation to match the target.
		\param pos: Look at target of the camera. */
		virtual void setTarget(const core::vector3df& pos) override;

		//! Sets the rotation of the node.
		/** This only modifies the relative rotation of the node.
		If the camera's target and rotation are bound ( @see bindTargetAndRotation() )
		then calling this will also change the camera's target to match the rotation.
		\param rotation New rotation of the node in degrees. */
		virtual void setRotation(const core::vector3df& rotation);

		//! Gets the current look at target of the camera
		/** \return The current look at target of the camera */
		virtual const core::vectorSIMDf& getTarget() const override;

		//! Sets the up vector of the camera.
		//! \param pos: New upvector of the camera.
		virtual void setUpVector(const core::vectorSIMDf& pos);

		//! Gets the up vector of the camera.
		//! \return Returns the up vector of the camera.
		virtual const core::vectorSIMDf& getUpVector() const override;

		//! Render
		virtual void render();

		//!
		virtual void recomputeProjectionMatrix();

		//! Returns the view area. Sometimes needed by bsp or lod render nodes.
		virtual const SViewFrustum* getViewFrustum() const;

		//! Disables or enables the camera to get key or mouse inputs.
		//! If this is set to true, the camera will respond to key inputs
		//! otherwise not.
		virtual void setInputReceiverEnabled(bool enabled);

		//! Returns if the input receiver of the camera is currently enabled.
		virtual bool isInputReceiverEnabled() const;

		//! Binds the camera scene node's rotation to its target position and vice vera, or unbinds them.
		virtual void bindTargetAndRotation(bool bound);

		//! Queries if the camera scene node's rotation and its target position are bound together.
		virtual bool getTargetAndRotationBinding(void) const;

		//! Creates a clone of this scene node and its children.
		virtual ISceneNode* clone(IDummyTransformationSceneNode* newParent=0, ISceneManager* newManager=0) { assert(false); return nullptr; }

	protected:
		void recalculateViewArea();

		core::vectorSIMDf Target;
		core::vectorSIMDf UpVector;

        core::matrix3x4SIMD viewMatrix;
        core::matrix4SIMD concatMatrix;
		SViewFrustum ViewArea;

		bool InputReceiverEnabled;
		bool TargetAndRotationAreBound;
	};

} // end namespace
} // end namespace

#endif

