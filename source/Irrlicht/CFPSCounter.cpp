// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CFPSCounter.h"
#include "irrMath.h"

namespace irr
{
namespace video
{


CFPSCounter::CFPSCounter()
:	FPS(60), Primitive(0), StartTime(0), FramesCounted(0),
	PrimitivesCounted(0), PrimitiveAverage(0), PrimitiveTotal(0)
{

}


//! returns current fps
int32_t CFPSCounter::getFPS() const
{
	return FPS;
}


//! returns current primitive count
uint32_t CFPSCounter::getPrimitive() const
{
	return Primitive;
}


//! returns average primitive count of last period
uint32_t CFPSCounter::getPrimitiveAverage() const
{
	return PrimitiveAverage;
}


//! returns accumulated primitive count since start
uint32_t CFPSCounter::getPrimitiveTotal() const
{
	return PrimitiveTotal;
}


//! to be called every frame
void CFPSCounter::registerFrame(uint32_t now, uint32_t primitivesDrawn)
{
	++FramesCounted;
	PrimitiveTotal += primitivesDrawn;
	PrimitivesCounted += primitivesDrawn;
	Primitive = primitivesDrawn;

	const uint32_t milliseconds = now - StartTime;

	if (milliseconds >= 1500 )
	{
		const float invMilli = core::reciprocal ( (float) milliseconds );

		FPS = core::ceil32 ( ( 1000 * FramesCounted ) * invMilli );
		PrimitiveAverage = core::ceil32 ( ( 1000 * PrimitivesCounted ) * invMilli );

		FramesCounted = 0;
		PrimitivesCounted = 0;
		StartTime = now;
	}
}


} // end namespace video
} // end namespace irr

