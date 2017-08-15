// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __S_K_2D_VERTEX_H_INCLUDED__
#define __S_K_2D_VERTEX_H_INCLUDED__

#include "vector2d.h"

typedef signed short TZBufferType;

namespace irr
{
namespace video
{

	struct S2DVertex
	{
		core::vector2d<int32_t> Pos;	// position
		core::vector2d<int32_t> TCoords;	// texture coordinates
		TZBufferType ZValue;		// zvalue
		uint16_t Color;
	};


} // end namespace video
} // end namespace irr

#endif

