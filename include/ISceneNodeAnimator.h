// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_SCENE_NODE_ANIMATOR_H_INCLUDED__
#define __NBL_I_SCENE_NODE_ANIMATOR_H_INCLUDED__

#include "nbl/core/core.h"

#include "ESceneNodeAnimatorTypes.h"
#include "IEventReceiver.h"

namespace nbl
{
namespace scene
{
	class IDummyTransformationSceneNode;
	class ISceneManager;

	//! Animates a scene node. Can animate position, rotation, material, and so on.
	/** A scene node animator is able to animate a scene node in a very simple way. It may
	change its position, rotation, scale and/or material. There are lots of animators
	to choose from. You can create scene node animators with the ISceneManager interface.
	*/
	class ISceneNodeAnimator : public virtual core::IReferenceCounted, public IEventReceiver
	{
	public:
		//! Animates a scene node.
		/** \param node Node to animate.
		\param timeMs Current time in milli seconds. */
		virtual void animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs) =0;

		//! Creates a clone of this animator.
		/** Please note that you will have to drop
		(IReferenceCounted::drop()) the returned pointer after calling this. */
		virtual ISceneNodeAnimator* createClone(IDummyTransformationSceneNode* node,
				ISceneManager* newManager=0) =0;

		//! Returns true if this animator receives events.
		/** When attached to an active camera, this animator will be
		able to respond to events such as mouse and keyboard events. */
		virtual bool isEventReceiverEnabled() const
		{
			return false;
		}

		//! Event receiver, override this function for camera controlling animators
		virtual bool OnEvent(const SEvent& event)
		{
			return false;
		}

		//! Returns type of the scene node animator
		virtual ESCENE_NODE_ANIMATOR_TYPE getType() const
		{
			return ESNAT_UNKNOWN;
		}

		//! Returns if the animator has finished.
		/** This is only valid for non-looping animators with a discrete end state.
		\return true if the animator has finished, false if it is still running. */
		virtual bool hasFinished(void) const
		{
			return false;
		}
	};


} // end namespace scene
} // end namespace nbl

#endif

