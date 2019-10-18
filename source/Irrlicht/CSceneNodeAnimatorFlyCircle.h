// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_SCENE_NODE_ANIMATOR_FLY_CIRCLE_H_INCLUDED__
#define __C_SCENE_NODE_ANIMATOR_FLY_CIRCLE_H_INCLUDED__

#include "ISceneNode.h"

namespace irr
{
namespace scene
{
	class CSceneNodeAnimatorFlyCircle : public ISceneNodeAnimator
	{
	public:

		//! constructor
		CSceneNodeAnimatorFlyCircle(uint32_t time,
				const core::vector3df& center, float radius,
				float speed, const core::vectorSIMDf& direction,
				float radiusEllipsoid);

		//! animates a scene node
		virtual void animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs);

		//! Returns type of the scene node animator
		virtual ESCENE_NODE_ANIMATOR_TYPE getType() const { return ESNAT_FLY_CIRCLE; }

		//! Creates a clone of this animator.
		/** Please note that you will have to drop
		(IReferenceCounted::drop()) the returned pointer after calling
		this. */
		virtual ISceneNodeAnimator* createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager=0);

	private:
		// do some initial calculations
		void init();

		// circle center
		core::vector3df Center;
		// up-vector, normal to the circle's plane
		core::vectorSIMDf Direction;
		// Two helper vectors
		core::vectorSIMDf VecU;
		core::vectorSIMDf VecV;
		float Radius;
		float RadiusEllipsoid;
		float Speed;
		uint32_t StartTime;
	};


} // end namespace scene
} // end namespace irr

#endif

