// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_SCENE_NODE_ANIMATOR_DELETE_H_INCLUDED__
#define __NBL_C_SCENE_NODE_ANIMATOR_DELETE_H_INCLUDED__

#include "ISceneNodeAnimatorFinishing.h"

namespace nbl
{
namespace scene
{
	class CSceneNodeAnimatorDelete : public ISceneNodeAnimatorFinishing
	{
	public:

		//! constructor
		CSceneNodeAnimatorDelete(ISceneManager* manager, uint32_t when);

		//! animates a scene node
		virtual void animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs);

		//! Returns type of the scene node animator
		virtual ESCENE_NODE_ANIMATOR_TYPE getType() const
		{
			return ESNAT_DELETION;
		}

		//! Creates a clone of this animator.
		/** Please note that you will have to drop
		(IReferenceCounted::drop()) the returned pointer after calling
		this. */
		virtual ISceneNodeAnimator* createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager=0);

	private:

		ISceneManager* SceneManager;
	};


} // end namespace scene
} // end namespace nbl

#endif

