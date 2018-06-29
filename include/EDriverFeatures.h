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
		//! Supports Alpha To Coverage (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
		EVDF_ALPHA_TO_COVERAGE = 0,

		//! Supports geometry shaders (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
		EVDF_GEOMETRY_SHADER,

		//! Supports tessellation shaders (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
		EVDF_TESSELLATION_SHADER,

		//! Whether one cycle of read->write to the same pixel on an active FBO is supported (always in Vulkan)
		EVDF_TEXTURE_BARRIER,

		//! If we can attach a stencil only texture to an FBO, if not must use Depth+Stencil or RenderBuffer
		EVDF_STENCIL_ONLY_TEXTURE,

		//! Whether we can get gl_DrawIDARB in GLSL (if not see https://www.g-truc.net/post-0518.html for ways to circumvent)
		EVDF_SHADER_DRAW_PARAMS,

		//! Whether we can indirectly tell how many indirect draws to issue (rather than issuing 0 triangle draw calls)
		EVDF_MULTI_DRAW_INDIRECT_COUNT,

		//! Whether we can know if the whole warp has a condition true, false, mixed, etc. NV_gpu_shader5 or ARB_shader_group_vote
		EVDF_SHADER_GROUP_VOTE,

		//! Whether we can know the warp/wavefront size and use ballot operations etc. NV_shader_thread_group or ARB_shader_ballot
		EVDF_SHADER_GROUP_BALLOT,

		//! Whether we can use Kepler-style shuffle instructions in a shader NV_shader_thread_shuffle
		EVDF_SHADER_GROUP_SHUFFLE,

		//! Whether we can force overlapping pixels to not rasterize in parallel, INTEL_fragment_shader_ordering, NV_fragment_shader_interlock or ARB_fragment_shader_interlock
		EVDF_FRAGMENT_SHADER_INTERLOCK,

		//other feature ideas are; bindless, sparse texture, sparse texture 2

		//! Only used for counting the elements of this enum
		EVDF_COUNT
	};

} // end namespace video
} // end namespace irr


#endif

