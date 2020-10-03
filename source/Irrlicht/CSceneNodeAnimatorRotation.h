// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __C_SCENE_NODE_ANIMATOR_ROTATION_H_INCLUDED__
#define __C_SCENE_NODE_ANIMATOR_ROTATION_H_INCLUDED__

#include "ISceneNode.h"

namespace irr
{
namespace scene
{
	class CSceneNodeAnimatorRotation : public ISceneNodeAnimator
	{
	public:

		//! constructor
		CSceneNodeAnimatorRotation(uint32_t time, const core::vector3df& rotation);

		//! animates a scene node
		virtual void animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs);

		//! Returns type of the scene node animator
		virtual ESCENE_NODE_ANIMATOR_TYPE getType() const { return ESNAT_ROTATION; }

		//! Creates a clone of this animator.
		/** Please note that you will have to drop
		(IReferenceCounted::drop()) the returned pointer after calling this. */
		virtual ISceneNodeAnimator* createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager=0);

	private:

		core::vector3df Rotation;
		uint32_t StartTime;
	};


} // end namespace scene
} // end namespace irr

#endif

