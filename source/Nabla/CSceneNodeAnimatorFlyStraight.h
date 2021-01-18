// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_SCENE_NODE_ANIMATOR_FLY_STRAIGHT_H_INCLUDED__
#define __NBL_C_SCENE_NODE_ANIMATOR_FLY_STRAIGHT_H_INCLUDED__

#include "ISceneNodeAnimatorFinishing.h"

namespace nbl
{
namespace scene
{
	class CSceneNodeAnimatorFlyStraight : public ISceneNodeAnimatorFinishing
	{
	public:

		//! constructor
		CSceneNodeAnimatorFlyStraight(const core::vectorSIMDf& startPoint,
						const core::vectorSIMDf& endPoint,
						uint32_t timeForWay,
						bool loop, uint32_t now, bool pingpong);

		//! animates a scene node
		virtual void animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs);

		//! Returns type of the scene node animator
		virtual ESCENE_NODE_ANIMATOR_TYPE getType() const { return ESNAT_FLY_STRAIGHT; }

		//! Creates a clone of this animator.
		/** Please note that you will have to drop
		(IReferenceCounted::drop()) the returned pointer after calling this. */
		virtual ISceneNodeAnimator* createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager=0);

	private:

		void recalculateIntermediateValues();

		core::vectorSIMDf Start;
		core::vectorSIMDf End;
		core::vectorSIMDf Vector;
		float TimeFactor;
		uint32_t StartTime;
		uint32_t TimeForWay;
		bool Loop;
		bool PingPong;
	};


} // end namespace scene
} // end namespace nbl

#endif

