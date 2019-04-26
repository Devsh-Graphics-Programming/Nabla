// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CSceneNodeAnimatorRotation.h"

namespace irr
{
namespace scene
{


//! constructor
CSceneNodeAnimatorRotation::CSceneNodeAnimatorRotation(uint32_t time, const core::vector3df& rotation)
: Rotation(rotation), StartTime(time)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSceneNodeAnimatorRotation");
	#endif
}


//! animates a scene node
void CSceneNodeAnimatorRotation::animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs)
{
	if (node) // thanks to warui for this fix
	{
		const uint32_t diffTime = timeMs - StartTime;

		if (diffTime != 0)
		{
			// clip the rotation to small values, to avoid
			// precision problems with huge floats.
			core::vector3df rot = node->getRotation() + Rotation*(diffTime*0.1f);
			if (rot.X>360.f)
				rot.X=fmodf(rot.X, 360.f);
			if (rot.Y>360.f)
				rot.Y=fmodf(rot.Y, 360.f);
			if (rot.Z>360.f)
				rot.Z=fmodf(rot.Z, 360.f);
			node->setRotation(rot);
			StartTime=timeMs;
		}
	}
}

ISceneNodeAnimator* CSceneNodeAnimatorRotation::createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager)
{
	CSceneNodeAnimatorRotation * newAnimator =
		new CSceneNodeAnimatorRotation(StartTime, Rotation);

	return newAnimator;
}


} // end namespace scene
} // end namespace irr

