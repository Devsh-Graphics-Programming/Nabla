// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_VIDEO_CAPABILITY_REPORTER_H_INCLUDED__
#define __NBL_I_VIDEO_CAPABILITY_REPORTER_H_INCLUDED__



namespace nbl
{
namespace video
{
	//! .
	class NBL_FORCE_EBO IVideoCapabilityReporter
	{
	public:
		//! Get type of video driver
		/** \return Type of driver. */
		//virtual E_DRIVER_TYPE getDriverType() const =0;

        //! enumeration for querying features of the video driver.
        enum E_DRIVER_FEATURE
        {
            //! Supports Alpha To Coverage (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
            EDF_ALPHA_TO_COVERAGE = 0,

            //! Supports geometry shaders (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
            EDF_GEOMETRY_SHADER,

            //! Supports tessellation shaders (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
            EDF_TESSELLATION_SHADER,

            //! If we can attach a stencil only texture to an FBO, if not must use Depth+Stencil
            EDF_STENCIL_ONLY_TEXTURE,

            //! Whether we can get gl_DrawIDARB in GLSL (if not see https://www.g-truc.net/post-0518.html for ways to circumvent)
            EDF_SHADER_DRAW_PARAMS,

            //! Whether we can force overlapping pixels to not rasterize in parallel, INTEL_fragment_shader_ordering, NV_fragment_shader_interlock or ARB_fragment_shader_interlock
            EDF_FRAGMENT_SHADER_INTERLOCK,

            //! Whether textures can be used by their hardware handles bindlessly (without specifying them in descriptor sets) TODO: What to do about this?
            EDF_BINDLESS_TEXTURE,

            //! Whether we can index samplers dynamically in a shader TODO: only in Vulkan or NV_gpu_shader5
            EDF_DYNAMIC_SAMPLER_INDEXING,

            //! A way to pass information between fragment shader invocations covering the same pixel
            EDF_INPUT_ATTACHMENTS,

            //other feature ideas are; bindless buffers, sparse texture, sparse texture 2

            //! Only used for counting the elements of this enum
            EDF_COUNT
        };

        virtual uint16_t retrieveDisplayRefreshRate() const { return 0u; }
		virtual uint32_t getMaxTextureBindingsCompute() const { return 0u; }
	};

} // end namespace video
} // end namespace nbl


#endif

