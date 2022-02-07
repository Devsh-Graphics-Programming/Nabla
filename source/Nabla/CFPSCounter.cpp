// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CFPSCounter.h"
#include "nbl/core/math/glslFunctions.tcc"

namespace nbl
{
namespace video
{
CFPSCounter::CFPSCounter()
    : FPS(60), Primitive(0), StartTime(std::chrono::high_resolution_clock::duration::zero()), FramesCounted(0),
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
void CFPSCounter::registerFrame(const std::chrono::high_resolution_clock::time_point& now, uint32_t primitivesDrawn)
{
    ++FramesCounted;
    PrimitiveTotal += primitivesDrawn;
    PrimitivesCounted += primitivesDrawn;
    Primitive = primitivesDrawn;

    auto delta = now - StartTime;

    if(delta >= std::chrono::milliseconds(1500))
    {
        const double invDelta = core::reciprocal((double)delta.count());

        FPS = core::ceil<double>((static_cast<double>(decltype(delta)::period::den) * FramesCounted) * invDelta);
        PrimitiveAverage = core::ceil<double>((static_cast<double>(decltype(delta)::period::den) * PrimitivesCounted) * invDelta);

        FramesCounted = 0;
        PrimitivesCounted = 0;
        StartTime = now;
    }
}

}  // end namespace video
}  // end namespace nbl
