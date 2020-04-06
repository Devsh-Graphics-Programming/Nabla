// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_IMAGE_MANIPULATOR_H_INCLUDED__
#define __I_IMAGE_MANIPULATOR_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/filters/CCopyImageFilter.h"

namespace irr
{
namespace asset
{

// scaled copies with filters
class CBlitImageFilter;

// specialized case of CBlitImageFilter
class CMipMapGenerationImageFilter;

} // end namespace asset
} // end namespace irr

#endif