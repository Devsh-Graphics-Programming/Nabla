// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SKINNED_MESH_SCENE_NODE_H_INCLUDED__
#define __I_SKINNED_MESH_SCENE_NODE_H_INCLUDED__

#include "ISceneNode.h"
#include "irr/video/IGPUSkinnedMesh.h"
#include "ISkinningStateManager.h"
#include "IAnimatedMeshSceneNode.h"

namespace irr
{
namespace scene
{

	//! Scene node capable of displaying an Skinned mesh and its shadow.
	/** The shadow is optional: If a shadow should be displayed too, just
	invoke the ISkinnedMeshSceneNode::createShadowVolumeSceneNode().*/
	class ISkinnedMeshSceneNode : public ISceneNode
	{
	public:

		//! Constructor
		ISkinnedMeshSceneNode(IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f))
			: ISceneNode(parent, mgr, id, position, rotation, scale) {}

		//! Returns type of the scene node
		virtual ESCENE_NODE_TYPE getType() const { return ESNT_SKINNED_MESH; }

		virtual video::ITextureBufferObject* getBonePoseTBO() const =0;

		//! Sets the current frame number.
		/** From now on the animation is played from this frame.
		\param frame: Number of the frame to let the animation be started from.
		The frame number must be a valid frame number of the IMesh used by this
		scene node. Set ISkinnedMesh::getMesh() for details. */
		virtual void setCurrentFrame(const float& frame) = 0;

		//! Sets the frame numbers between the animation is looped.
		/** The default is 0 - MaximalFrameCount of the mesh.
		\param begin: Start frame number of the loop.
		\param end: End frame number of the loop.
		\return True if successful, false if not. */
		virtual bool setFrameLoop(const float& begin, const float& end) = 0;

		//! Sets the speed with which the animation is played.
		/** \param framesPerSecond: Frames per second played. */
		virtual void setAnimationSpeed(const float&  framesPerSecond) = 0;

		//! Gets the speed with which the animation is played.
		/** \return Frames per second played. */
		virtual float getAnimationSpeed() const =0;

		//! only for EBUM_NONE and EBUM_READ, it dictates what is the actual frequency we want to bother updating the mesh
		//! because we don't want to waste CPU time if we can tolerate the bones updating at 120Hz or similar
		virtual void setDesiredUpdateFrequency(const float& hertz) = 0;

		virtual float getDesiredUpdateFrequency() const =0;

		//! returns the material based on the zero based index i. To get the amount
		//! of materials used by this scene node, use getMaterialCount().
		//! This function is needed for inserting the node into the scene hirachy on a
		//! optimal position for minimizing renderstate changes, but can also be used
		//! to directly modify the material of a scene node.
		virtual video::SGPUMaterial& getMaterial(uint32_t i) = 0;

		//! returns amount of materials used by this scene node.
		virtual uint32_t getMaterialCount() const = 0;

        virtual size_t getBoneCount() const = 0;

		//! frame
		virtual void OnRegisterSceneNode() = 0;

		//! OnAnimate() is called just before rendering the whole scene.
		virtual void OnAnimate(uint32_t timeMs) = 0;

		//! renders the node.
		virtual void render() = 0;

		virtual void setBoundingBox(const core::aabbox3d<float>& bbox) = 0;
		//! returns the axis aligned bounding box of this node
		virtual const core::aabbox3d<float>& getBoundingBox() = 0;

		//! Get a pointer to a joint in the mesh.
		/** With this method it is possible to attach scene nodes to
		joints for example possible to attach a weapon to the left hand
		of an Skinned model. This example shows how:
		\code
		ISceneNode* hand =
			yourSkinnedMeshSceneNode->getJointNode("LeftHand");
		hand->addChild(weaponSceneNode);
		\endcode
		Please note that the joint returned by this method may not exist
		before this call and the joints in the node were created by it.
		\param jointID: ID of the joint.
		\return Pointer to the scene node which represents the joint
		with the specified name. Returns 0 if the contained mesh is not
		an skinned mesh or the name of the joint could not be found. */
		virtual ISkinningStateManager::IBoneSceneNode* getJointNode(const size_t& jointID) = 0;

		//! Returns the currently displayed frame number.
		virtual float getFrameNr() const = 0;
		//! Returns the current start frame number.
		virtual float getStartFrame() const = 0;
		//! Returns the current end frame number.
		virtual float getEndFrame() const = 0;

		//! Sets looping mode which is on by default.
		/** If set to false, animations will not be played looped. */
		virtual void setLoopMode(bool playAnimationLooped) = 0;

		//! returns the current loop mode
		/** When true the animations are played looped */
		virtual bool getLoopMode() const = 0;

		//! Sets a callback interface which will be called if an animation playback has ended.
		/** Set this to 0 to disable the callback again.
		Please note that this will only be called when in non looped
		mode, see ISkinnedMeshSceneNode::setLoopMode(). */
		virtual void setAnimationEndCallback(IAnimationEndCallBack<ISkinnedMeshSceneNode>* callback=0) = 0;

		//! Sets a new mesh
		virtual void setMesh(video::IGPUSkinnedMesh* mesh, const ISkinningStateManager::E_BONE_UPDATE_MODE& boneControl=ISkinningStateManager::EBUM_NONE) = 0;

		//! Returns the current mesh
		virtual video::IGPUSkinnedMesh* getMesh(void) = 0;

		//! animates the joints in the mesh based on the current frame.
		/** Also takes in to account transitions. */
		virtual void animateJoints(/*bool CalculateAbsolutePositions=true*/) = 0;

		//! Creates a clone of this scene node and its children.
		/** \param newParent An optional new parent.
		\param newManager An optional new scene manager.
		\return The newly created clone of this node. */
		inline virtual ISceneNode* clone(IDummyTransformationSceneNode* newParent=0, ISceneManager* newManager=0) {return NULL;}

	};

} // end namespace scene
} // end namespace irr

#endif


