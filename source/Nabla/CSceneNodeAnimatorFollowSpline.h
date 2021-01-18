// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_SCENE_NODE_ANIMATOR_FOLLOW_SPLINE_H_INCLUDED__
#define __NBL_C_SCENE_NODE_ANIMATOR_FOLLOW_SPLINE_H_INCLUDED__

#include "ISceneNode.h"

#include "ISceneNodeAnimatorFinishing.h"

namespace nbl
{
namespace scene
{
	//! Scene node animator based free code Matthias Gall wrote and sent in. (Most of
	//! this code is written by him, I only modified bits.)
	class CSceneNodeAnimatorFollowSpline : public ISceneNodeAnimatorFinishing
	{
	public:

		//! constructor
		CSceneNodeAnimatorFollowSpline(uint32_t startTime,
			const core::vector< core::vector3df >& points,
			float speed = 1.0f, float tightness = 0.5f, bool loop=true, bool pingpong=false);

		//! animates a scene node
		virtual void animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs);

		//! Returns type of the scene node animator
		virtual ESCENE_NODE_ANIMATOR_TYPE getType() const { return ESNAT_FOLLOW_SPLINE; }

		//! Creates a clone of this animator.
		/** Please note that you will have to drop
		(IReferenceCounted::drop()) the returned pointer after calling
		this. */
		virtual ISceneNodeAnimator* createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager=0);

	protected:

		//! clamps a the value idx to fit into range 0..size-1
		int32_t clamp(int32_t idx, int32_t size);

		core::vector< core::vector3df > Points;
		float Speed;
		float Tightness;
		uint32_t StartTime;
		bool Loop;
		bool PingPong;
	};


} // end namespace scene
} // end namespace nbl

#endif

