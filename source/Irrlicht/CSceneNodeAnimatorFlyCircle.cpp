// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "irr/core/core.h"

#include "CSceneNodeAnimatorFlyCircle.h"

namespace irr
{
namespace scene
{


//! constructor
CSceneNodeAnimatorFlyCircle::CSceneNodeAnimatorFlyCircle(uint32_t time,
		const core::vector3df& center, float radius, float speed,
		const core::vectorSIMDf& direction, float radiusEllipsoid)
			: Center(center), Direction(direction), Radius(radius),
			RadiusEllipsoid(radiusEllipsoid), Speed(speed), StartTime(time)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSceneNodeAnimatorFlyCircle");
	#endif
	init();
}


void CSceneNodeAnimatorFlyCircle::init()
{
	Direction = core::normalize(Direction);

	if (Direction.Y != 0)
		VecV = core::normalize(core::cross(core::vectorSIMDf(50,0,0),Direction));
	else
		VecV = core::normalize(core::cross(core::vectorSIMDf(0,50,0),Direction));
	VecU = core::normalize(core::cross(VecV,Direction));
}


//! animates a scene node
void CSceneNodeAnimatorFlyCircle::animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs)
{
	if ( 0 == node )
		return;

	float time;

	// Check for the condition where the StartTime is in the future.
	if(StartTime > timeMs)
		time = ((int32_t)timeMs - (int32_t)StartTime) * Speed;
	else
		time = (timeMs-StartTime) * Speed;

//	node->setPosition(Center + Radius * ((VecU*cosf(time)) + (VecV*sinf(time))));
	float r2 = RadiusEllipsoid == 0.f ? Radius : RadiusEllipsoid;
	node->setPosition(Center + (VecU*Radius*cosf(time) + VecV*r2*sinf(time) ).getAsVector3df() );
}




ISceneNodeAnimator* CSceneNodeAnimatorFlyCircle::createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager)
{
	CSceneNodeAnimatorFlyCircle * newAnimator =
		new CSceneNodeAnimatorFlyCircle(StartTime, Center, Radius, Speed, Direction, RadiusEllipsoid);

	return newAnimator;
}


} // end namespace scene
} // end namespace irr

