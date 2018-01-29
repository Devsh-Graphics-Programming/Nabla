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
		//! Supports Alpha To Coverage
		EVDF_ALPHA_TO_COVERAGE = 0,

		//! Supports geometry shaders
		EVDF_GEOMETRY_SHADER,

		//!
		EVDF_TESSELLATION_SHADER,

		//! Only used for counting the elements of this enum
		EVDF_COUNT
	};

} // end namespace video
} // end namespace irr


#endif

