// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CSceneNodeAnimatorFlyStraight.h"

namespace irr
{
namespace scene
{


//! constructor
CSceneNodeAnimatorFlyStraight::CSceneNodeAnimatorFlyStraight(const core::vectorSIMDf& startPoint,
				const core::vectorSIMDf& endPoint, uint32_t timeForWay,
				bool loop, uint32_t now, bool pingpong)
: ISceneNodeAnimatorFinishing(now + timeForWay),
	Start(startPoint), End(endPoint), TimeFactor(0.0f), StartTime(now),
	TimeForWay(timeForWay), Loop(loop), PingPong(pingpong)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSceneNodeAnimatorFlyStraight");
	#endif

	recalculateIntermediateValues();
}


void CSceneNodeAnimatorFlyStraight::recalculateIntermediateValues()
{
	Vector = End-Start;
	auto len = core::length(Vector);
	TimeFactor = len[0] / float(TimeForWay);
	Vector /= len;;
}


//! animates a scene node
void CSceneNodeAnimatorFlyStraight::animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs)
{
	if (!node)
		return;

	uint32_t t = (timeMs-StartTime);

	core::vectorSIMDf pos;

	if (!Loop && !PingPong && t >= TimeForWay)
	{
		pos = End;
		HasFinished = true;
	}
	else if (!Loop && PingPong && t >= TimeForWay * 2.f )
	{
		pos = Start;
		HasFinished = true;
	}
	else
	{
		float phase = fmodf( (float) t, (float) TimeForWay );
		auto rel = Vector * phase * TimeFactor;
		const bool pong = PingPong && fmodf( (float) t, (float) TimeForWay*2.f ) >= TimeForWay;

		if ( !pong )
		{
			pos += Start + rel;
		}
		else
		{
			pos = End - rel;
		}
	}

	node->setPosition(pos.getAsVector3df());
}


ISceneNodeAnimator* CSceneNodeAnimatorFlyStraight::createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager)
{
	CSceneNodeAnimatorFlyStraight * newAnimator =
		new CSceneNodeAnimatorFlyStraight(Start, End, TimeForWay, Loop, StartTime, PingPong);

	return newAnimator;
}


} // end namespace scene
} // end namespace irr

