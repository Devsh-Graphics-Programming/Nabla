// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

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
		CSceneNodeAnimatorRotation(u32 time, const core::vector3df& rotation);

		//! animates a scene node
		virtual void animateNode(IDummyTransformationSceneNode* node, u32 timeMs);

		//! Returns type of the scene node animator
		virtual ESCENE_NODE_ANIMATOR_TYPE getType() const { return ESNAT_ROTATION; }

		//! Creates a clone of this animator.
		/** Please note that you will have to drop
		(IReferenceCounted::drop()) the returned pointer after calling this. */
		virtual ISceneNodeAnimator* createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager=0);

	private:

		core::vector3df Rotation;
		u32 StartTime;
	};


} // end namespace scene
} // end namespace irr

#endif

