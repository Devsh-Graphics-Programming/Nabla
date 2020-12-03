// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_FPSCOUNTER_H_INCLUDED__
#define __NBL_C_FPSCOUNTER_H_INCLUDED__

#include "nbl/core/Types.h"
#include "nbl/core/alloc/AlignedBase.h"

namespace nbl
{
namespace video
{


class CFPSCounter : public core::AllocationOverrideDefault
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
	void registerFrame(const std::chrono::high_resolution_clock::time_point& now, uint32_t primitive);

private:

	int32_t FPS;
	uint32_t Primitive;
	std::chrono::high_resolution_clock::time_point StartTime;

	uint32_t FramesCounted;
	uint32_t PrimitivesCounted;
	uint32_t PrimitiveAverage;
	uint32_t PrimitiveTotal;
};


} // end namespace video
} // end namespace nbl


#endif

