// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

//! As of Irrlicht 1.6, position2d is a synonym for vector2d.
/** You should consider position2d to be deprecated, and use vector2d by preference. */

#ifndef __NBL_POSITION_H_INCLUDED__
#define __NBL_POSITION_H_INCLUDED__

#include "vector2d.h"

namespace nbl
{
namespace core
{
// Use typedefs where possible as they are more explicit...

//! \deprecated position2d is now a synonym for vector2d, but vector2d should be used directly.
typedef vector2d<float> position2df;

//! \deprecated position2d is now a synonym for vector2d, but vector2d should be used directly.
typedef vector2d<int32_t> position2di;
}  // namespace core
}  // namespace nbl

// ...and use a #define to catch the rest, for (e.g.) position2d<double>
#define position2d vector2d

#endif
