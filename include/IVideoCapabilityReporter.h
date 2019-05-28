// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_VIDEO_CAPABILITY_REPORTER_H_INCLUDED__
#define __IRR_I_VIDEO_CAPABILITY_REPORTER_H_INCLUDED__

#include <string>
#include "ITexture.h"

#include "IrrCompileConfig.h"

namespace irr
{
namespace video
{
	//! .
	class IRR_FORCE_EBO IVideoCapabilityReporter
	{
	public:
		//! Get type of video driver
		/** \return Type of driver. */
		virtual E_DRIVER_TYPE getDriverType() const =0;

        //! enumeration for querying features of the video driver.
        enum E_DRIVER_FEATURE
        {
            //! Supports Alpha To Coverage (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
            EDF_ALPHA_TO_COVERAGE = 0,

            //! Supports geometry shaders (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
            EDF_GEOMETRY_SHADER,

            //! Supports tessellation shaders (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
            EDF_TESSELLATION_SHADER,

			//! Whether we can download sub-areas of an IGPUTexture
			EDF_GET_TEXTURE_SUB_IMAGE,

            //! Whether one cycle of read->write to the same pixel on an active FBO is supported (always in Vulkan)
            EDF_TEXTURE_BARRIER,

            //! If we can attach a stencil only texture to an FBO, if not must use Depth+Stencil
            EDF_STENCIL_ONLY_TEXTURE,

            //! Whether we can get gl_DrawIDARB in GLSL (if not see https://www.g-truc.net/post-0518.html for ways to circumvent)
            EDF_SHADER_DRAW_PARAMS,

            //! Whether we can indirectly tell how many indirect draws to issue (rather than issuing 0 triangle draw calls)
            EDF_MULTI_DRAW_INDIRECT_COUNT,

            //! Whether we can know if the whole warp has a condition true, false, mixed, etc. NV_gpu_shader5 or ARB_shader_group_vote
            EDF_SHADER_GROUP_VOTE,

            //! Whether we can know the warp/wavefront size and use ballot operations etc. NV_shader_thread_group or ARB_shader_ballot
            EDF_SHADER_GROUP_BALLOT,

            //! Whether we can use Kepler-style shuffle instructions in a shader NV_shader_thread_shuffle
            EDF_SHADER_GROUP_SHUFFLE,

            //! Whether we can force overlapping pixels to not rasterize in parallel, INTEL_fragment_shader_ordering, NV_fragment_shader_interlock or ARB_fragment_shader_interlock
            EDF_FRAGMENT_SHADER_INTERLOCK,

            //! Whether textures can be used by their hardware handles bindlessly (without specifying them in descriptor sets)
            EDF_BINDLESS_TEXTURE,

            //! Whether we can index samplers dynamically in a shader (automatically true if bindless is enabled)
            EDF_DYNAMIC_SAMPLER_INDEXING,

            //other feature ideas are; bindless buffers, sparse texture, sparse texture 2

            //! Only used for counting the elements of this enum
            EDF_COUNT
        };

		//! Queries the features of the driver.
		/** Returns true if a feature is available
		\param feature Feature to query.
		\return True if the feature is available, false if not. */
		virtual bool queryFeature(const E_DRIVER_FEATURE& feature) const {return false;}

		//! Gets name of this video driver.
		/** \return Returns the name of the video driver, e.g. in case
		of the Direct3D8 driver, it would return "Direct3D 8.1". */
		virtual const wchar_t* getName() const =0;

		//! Returns the maximum amount of primitives
		/** (mostly vertices) which the device is able to render.
		\return Maximum amount of primitives. */
		virtual uint32_t getMaximalIndicesCount() const =0;

		//! Get the current color format of the color buffer
		/** \return Color format of the color buffer. */
		virtual asset::E_FORMAT getColorFormat() const =0;

		//! Get the graphics card vendor name.
		virtual std::string getVendorInfo() =0;

        virtual uint32_t getMaxComputeWorkGroupSize(uint32_t _dimension) const = 0;

		//! Get the maximum texture size supported.
		virtual const uint32_t* getMaxTextureSize(const ITexture::E_TEXTURE_TYPE& type) const =0;

		//!
		virtual uint32_t getRequiredUBOAlignment() const = 0;

		//!
		virtual uint32_t getRequiredSSBOAlignment() const = 0;

		//!
		virtual uint32_t getRequiredTBOAlignment() const = 0;

		//!
		virtual uint32_t getMinimumMemoryMapAlignment() const { return _IRR_MIN_MAP_BUFFER_ALIGNMENT; }

        virtual uint16_t retrieveDisplayRefreshRate() const { return 0u; }

        virtual uint64_t getMaxUBOSize() const = 0;
        virtual uint64_t getMaxSSBOSize() const = 0;
        virtual uint64_t getMaxTBOSize() const = 0;
        virtual uint64_t getMaxBufferSize() const = 0;
	};

} // end namespace video
} // end namespace irr


#endif

