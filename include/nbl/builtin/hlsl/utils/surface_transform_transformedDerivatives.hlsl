
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

namespace nbl
{
namespace hlsl
{
namespace surface_transform
{


switch (swapchainTransform) 
{
	case IDENTITY:
	    return ddxDdy;
	case HORIZONTAL_MIRROR:
	    return OUTPUT_TYPE(-ddxDdy[0], ddxDdy[1]);
	case HORIZONTAL_MIRROR_ROTATE_180:
	    return OUTPUT_TYPE(ddxDdy[0], -ddxDdy[1]);
	case ROTATE_180:
	    return OUTPUT_TYPE(-ddxDdy[0], -ddxDdy[1]);
	case ROTATE_90:
	    return OUTPUT_TYPE(ddxDdy[1], -ddxDdy[0]);
	case ROTATE_270:
	    return OUTPUT_TYPE(-ddxDdy[1], ddxDdy[0]);
	case HORIZONTAL_MIRROR_ROTATE_90:
	    return OUTPUT_TYPE(ddxDdy[1], ddxDdy[0]);
	case HORIZONTAL_MIRROR_ROTATE_270:
	    return OUTPUT_TYPE(-ddxDdy[1], -ddxDdy[0]);
	default:
	    return OUTPUT_TYPE(0);
}


}
}
}