// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_CAMERA_SCENE_NODE_H_INCLUDED__
#define __I_CAMERA_SCENE_NODE_H_INCLUDED__

#include "ISceneNode.h"
#include "IEventReceiver.h"
#include "matrixutil.h"

namespace irr
{
namespace scene
{
	struct SViewFrustum;

	//! Scene Node which is a (controlable) camera.
	/** The whole scene will be rendered from the cameras point of view.
	Because the ICameraScenNode is a SceneNode, it can be attached to any
	other scene node, and will follow its parents movement, rotation and so
	on.
	*/
	class ICameraSceneNode : public ISceneNode, public IEventReceiver
	{
	public:

		//! Constructor
		ICameraSceneNode(IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f,1.0f,1.0f))
			: ISceneNode(parent, mgr, id, position, rotation, scale) {}

		//! Sets the projection matrix of the camera.
		/** The matrix class has some methods to build a
		projection matrix. e.g:
		core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH.
		Note that the matrix will only stay as set by this method until
		one of the following Methods are called: setNearValue,
		setFarValue, setAspectRatio, setFOV.
		The function will figure it out if you've set an orthogonal matrix.
		\param projection The new projection matrix of the camera.
		*/
		virtual void setProjectionMatrix(const core::matrix4SIMD& projection) =0;

		//! Gets the current projection matrix of the camera.
		/** \return The current projection matrix of the camera. */
		virtual const core::matrix4SIMD& getProjectionMatrix() const =0;

		//! Gets the current view matrix of the camera.
		/** \return The current view matrix of the camera. */
		virtual const core::matrix4x3& getViewMatrix() const =0;

		virtual const core::matrix4SIMD& getConcatenatedMatrix() const =0;

		//! It is possible to send mouse and key events to the camera.
		/** Most cameras may ignore this input, but camera scene nodes
		which are created for example with
		ISceneManager::addCameraSceneNodeMaya or
		ISceneManager::addCameraSceneNodeFPS, may want to get
		this input for changing their position, look at target or
		whatever. */
		virtual bool OnEvent(const SEvent& event) =0;

		//! Sets the look at target of the camera
		/** If the camera's target and rotation are bound ( @see
		bindTargetAndRotation() ) then calling this will also change
		the camera's scene node rotation to match the target.
		Note that setTarget uses the current absolute position
		internally, so if you changed setPosition since last rendering you must
		call updateAbsolutePosition before using this function.
		\param pos Look at target of the camera, in world co-ordinates. */
		virtual void setTarget(const core::vector3df& pos) =0;

		//! Sets the rotation of the node.
		/** This only modifies the relative rotation of the node.
		If the camera's target and rotation are bound ( @see
		bindTargetAndRotation() ) then calling this will also change
		the camera's target to match the rotation.
		\param rotation New rotation of the node in degrees. */
		virtual void setRotation(const core::vector3df& rotation) =0;

		//! Gets the current look at target of the camera
		/** \return The current look at target of the camera, in world co-ordinates */
		virtual const core::vectorSIMDf& getTarget() const =0;

		//! Sets the up vector of the camera.
		/** \param pos: New upvector of the camera. */
		virtual void setUpVector(const core::vector3df& pos) =0;

		//! Gets the up vector of the camera.
		/** \return The up vector of the camera, in world space. */
		virtual const core::vectorSIMDf& getUpVector() const =0;

		//! Gets the value of the near plane of the camera.
		/** \return The value of the near plane of the camera. */
		virtual float getNearValue() const =0;

		//! Gets the value of the far plane of the camera.
		/** \return The value of the far plane of the camera. */
		virtual float getFarValue() const =0;

		//! Gets the aspect ratio of the camera.
		/** \return The aspect ratio of the camera. */
		virtual float getAspectRatio() const =0;

		//! Gets the field of view of the camera.
		/** \return The field of view of the camera in radians. */
		virtual float getFOV() const =0;

		//! Sets the value of the near clipping plane. (default: 1.0f)
		/** \param zn: New z near value. */
		virtual void setNearValue(float zn) =0;

		//! Sets the value of the far clipping plane (default: 2000.0f)
		/** \param zf: New z far value. */
		virtual void setFarValue(float zf) =0;

		//! Sets the aspect ratio (default: 4.0f / 3.0f)
		/** \param aspect: New aspect ratio. */
		virtual void setAspectRatio(float aspect) =0;

		//! Sets the field of view (Default: PI / 2.5f)
		/** \param fovy: New field of view in radians. */
		virtual void setFOV(float fovy) =0;

		//! Get the view frustum.
		/** Needed sometimes by bspTree or LOD render nodes.
		\return The current view frustum. */
		virtual const SViewFrustum* getViewFrustum() const =0;

		//! Disables or enables the camera to get key or mouse inputs.
		/** If this is set to true, the camera will respond to key
		inputs otherwise not. */
		virtual void setInputReceiverEnabled(bool enabled) =0;

		//! Checks if the input receiver of the camera is currently enabled.
		virtual bool isInputReceiverEnabled() const =0;

		//! Binds the camera scene node's rotation to its target position and vice vera, or unbinds them.
		/** When bound, calling setRotation() will update the camera's
		target position to be along its +Z axis, and likewise calling
		setTarget() will update its rotation so that its +Z axis will
		point at the target point. FPS camera use this binding by
		default; other cameras do not.
		\param bound True to bind the camera's scene node rotation
		and targetting, false to unbind them.
		@see getTargetAndRotationBinding() */
		virtual void bindTargetAndRotation(bool bound) =0;

		//! Queries if the camera scene node's rotation and its target position are bound together.
		/** @see bindTargetAndRotation() */
		virtual bool getTargetAndRotationBinding(void) const =0;
	};

} // end namespace scene
} // end namespace irr

#endif

