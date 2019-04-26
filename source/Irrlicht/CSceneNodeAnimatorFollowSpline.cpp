// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CSceneNodeAnimatorFollowSpline.h"

namespace irr
{
namespace scene
{


//! constructor
CSceneNodeAnimatorFollowSpline::CSceneNodeAnimatorFollowSpline(uint32_t time,
	const core::vector<core::vector3df>& points, float speed,
	float tightness, bool loop, bool pingpong)
: ISceneNodeAnimatorFinishing(0), Points(points), Speed(speed), Tightness(tightness), StartTime(time)
, Loop(loop), PingPong(pingpong)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSceneNodeAnimatorFollowSpline");
	#endif
}


inline int32_t CSceneNodeAnimatorFollowSpline::clamp(int32_t idx, int32_t size)
{
	return ( idx<0 ? size+idx : ( idx>=size ? idx-size : idx ) );
}


//! animates a scene node
void CSceneNodeAnimatorFollowSpline::animateNode(IDummyTransformationSceneNode* node, uint32_t timeMs)
{
	if(!node)
		return;

	const uint32_t pSize = Points.size();
	if (pSize==0)
	{
		if ( !Loop )
			HasFinished = true;
		return;
	}
	if (pSize==1)
	{
		if ( timeMs > StartTime )
		{
			node->setPosition(Points[0]);
			if ( !Loop )
				HasFinished = true;
		}
		return;
	}

	const float dt = ( (timeMs-StartTime) * Speed * 0.001f );
	const int32_t unwrappedIdx = core::floor32( dt );
	if ( !Loop && unwrappedIdx >= (int32_t)pSize-1 )
	{
		node->setPosition(Points[pSize-1]);
		HasFinished = true;
		return;
	}
	const bool pong = PingPong && (unwrappedIdx/(pSize-1))%2;
	const float u =  pong ? 1.f-core::fract ( dt ) : core::fract ( dt );
	const int32_t idx = pong ?	(pSize-2) - (unwrappedIdx % (pSize-1))
						: (PingPong ? unwrappedIdx % (pSize-1)
									: unwrappedIdx % pSize);
	//const float u = 0.001f * fmodf( dt, 1000.0f );

	const core::vector3df& p0 = Points[ clamp( idx - 1, pSize ) ];
	const core::vector3df& p1 = Points[ clamp( idx + 0, pSize ) ]; // starting point
	const core::vector3df& p2 = Points[ clamp( idx + 1, pSize ) ]; // end point
	const core::vector3df& p3 = Points[ clamp( idx + 2, pSize ) ];

	// hermite polynomials
	const float h1 = 2.0f * u * u * u - 3.0f * u * u + 1.0f;
	const float h2 = -2.0f * u * u * u + 3.0f * u * u;
	const float h3 = u * u * u - 2.0f * u * u + u;
	const float h4 = u * u * u - u * u;

	// tangents
	const core::vector3df t1 = ( p2 - p0 ) * Tightness;
	const core::vector3df t2 = ( p3 - p1 ) * Tightness;

	// interpolated point
	node->setPosition(p1 * h1 + p2 * h2 + t1 * h3 + t2 * h4);
}


ISceneNodeAnimator* CSceneNodeAnimatorFollowSpline::createClone(IDummyTransformationSceneNode* node, ISceneManager* newManager)
{
	CSceneNodeAnimatorFollowSpline * newAnimator =
		new CSceneNodeAnimatorFollowSpline(StartTime, Points, Speed, Tightness);

	return newAnimator;
}


} // end namespace scene
} // end namespace irr

