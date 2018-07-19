// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_VIDEO_CAPABILITY_REPORTER_H_INCLUDED__
#define __IRR_I_VIDEO_CAPABILITY_REPORTER_H_INCLUDED__

#include "rect.h"
#include "SColor.h"
#include "IGPUMappedBuffer.h"
#include "ITexture.h"
#include "IMultisampleTexture.h"
#include "ITextureBufferObject.h"
#include "IRenderBuffer.h"
#include "IFrameBuffer.h"
#include "irrArray.h"
#include "matrix4x3.h"
#include "plane3d.h"
#include "dimension2d.h"
#include "position2d.h"
#include "SMaterial.h"
#include "IDriverFence.h"
#include "SMesh.h"
#include "IGPUTimestampQuery.h"
#include "IOcclusionQuery.h"
#include "triangle3d.h"
#include "EDriverTypes.h"
#include "EDriverFeatures.h"
#include <string>

namespace irr
{
namespace video
{
	//! .
	class IVideoCapabilityReporter
	{
	public:
		//! Queries the features of the driver.
		/** Returns true if a feature is available
		\param feature Feature to query.
		\return True if the feature is available, false if not. */
		virtual bool queryFeature(const E_VIDEO_DRIVER_FEATURE& feature) const =0;

		//! Gets name of this video driver.
		/** \return Returns the name of the video driver, e.g. in case
		of the Direct3D8 driver, it would return "Direct3D 8.1". */
		virtual const wchar_t* getName() const =0;

		//! Returns the maximum amount of primitives
		/** (mostly vertices) which the device is able to render.
		\return Maximum amount of primitives. */
		virtual uint32_t getMaximalIndicesCount() const =0;


		//! Get the graphics card vendor name.
		virtual std::string getVendorInfo() =0;

		//! Get the maximum texture size supported.
		virtual const uint32_t* getMaxTextureSize(const ITexture::E_TEXTURE_TYPE& type) const =0;

		//!
		virtual uint32_t getRequiredUBOAlignment() const = 0;

		//!
		virtual uint32_t getRequiredSSBOAlignment() const = 0;

		//!
		virtual uint32_t getRequiredTBOAlignment() const = 0;
	};

} // end namespace video
} // end namespace irr


#endif

