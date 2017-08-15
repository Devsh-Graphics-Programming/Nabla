// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_FPSCOUNTER_H_INCLUDED__
#define __C_FPSCOUNTER_H_INCLUDED__

#include "irrTypes.h"

namespace irr
{
namespace video
{


class CFPSCounter
{
public:
	CFPSCounter();

	//! returns current fps
	int32_t getFPS() const;

	//! returns primitive count
	uint32_t getPrimitive() const;

	//! returns average primitive count of last period
	uint32_t getPrimitiveAverage() const;

	//! returns accumulated primitive count since start
	uint32_t getPrimitiveTotal() const;

	//! to be called every frame
	void registerFrame(uint32_t now, uint32_t primitive);

private:

	int32_t FPS;
	uint32_t Primitive;
	uint32_t StartTime;

	uint32_t FramesCounted;
	uint32_t PrimitivesCounted;
	uint32_t PrimitiveAverage;
	uint32_t PrimitiveTotal;
};


} // end namespace video
} // end namespace irr


#endif

