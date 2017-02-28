// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __E_DRIVER_FEATURES_H_INCLUDED__
#define __E_DRIVER_FEATURES_H_INCLUDED__

namespace irr
{
namespace video
{

	//! enumeration for querying features of the video driver.
	enum E_VIDEO_DRIVER_FEATURE
	{
		//! Are stencilbuffers switched on and does the device support stencil buffers?
		EVDF_STENCIL_BUFFER = 0,

		//! Is GLSL supported?
		EVDF_ARB_GLSL,

		//! Is HLSL supported?
		EVDF_HLSL,

		//! Supports Alpha To Coverage
		EVDF_ALPHA_TO_COVERAGE,

		//! Supports Color masks (disabling color planes in output)
		EVDF_COLOR_MASK,

		//! Supports separate blend settings for multiple render targets
		EVDF_MRT_BLEND,

		//! Supports separate color masks for multiple render targets
		EVDF_MRT_COLOR_MASK,

		//! Supports separate blend functions for multiple render targets
		EVDF_MRT_BLEND_FUNC,

		//! Supports geometry shaders
		EVDF_GEOMETRY_SHADER,

		//! Supports occlusion queries
		EVDF_OCCLUSION_QUERY,

		//! Supports polygon offset/depth bias for avoiding z-fighting
		EVDF_POLYGON_OFFSET,

		//! Support for different blend functions. Without, only ADD is available
		EVDF_BLEND_OPERATIONS,

		//! Only used for counting the elements of this enum
		EVDF_COUNT
	};

} // end namespace video
} // end namespace irr


#endif

