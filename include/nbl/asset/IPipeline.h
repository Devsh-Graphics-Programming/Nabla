// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_PIPELINE_H_INCLUDED_


#include "nbl/asset/IPipelineLayout.h"


namespace nbl::asset
{
//! Interface class for graphics and compute pipelines
/*
	A pipeline refers to a succession of fixed stages 
	through which a data input flows; each stage processes 
	the incoming data and hands it over to the next stage. 
	The final product will be either a 2D raster drawing image 
	(the graphics pipeline) or updated resources (buffers or images) 
	with computational logic and calculations (the compute pipeline).

	Vulkan supports multiple types of pipelines:
	- graphics pipeline
	- compute pipeline
	- TODO: Raytracing
*/
template<typename PipelineLayout>
class IPipeline
{
	public:
		// For now, due to API design we implicitly satisfy:
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-08771
		// to:
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pSpecializationInfo-06849
		struct SCreationParams
		{
			public:
				const PipelineLayout* layout = nullptr;

			protected:
				// This is not public to make sure that different pipelines only get the enums they support
				enum class FLAGS : uint64_t
				{
					NONE = 0, // disallowed in maintanance5
					DISABLE_OPTIMIZATIONS = 1<<0,
					ALLOW_DERIVATIVES = 1<<1,
					
					// I can just derive this
					//DERIVATIVE = 1<<2,

					// Graphics Pipelines only
					//VIEW_INDEX_FROM_DEVICE_INDEX = 1<<3,
					
					// Compute Pipelines only
					//DISPATCH_BASE = 1<<4,
					
					// This is for NV-raytracing extension. Now this is done via IDeferredOperation
					//DEFER_COMPILE_NV = 1<<5,

					// We use Renderdoc to take care of this for us,
					// we won't be parsing the statistics and internal representation ourselves.
					//CAPTURE_STATISTICS = 1<<6,
					//CAPTURE_INTERNAL_REPRESENTATIONS = 1<<7,

					// Will soon be deprecated due to
					// https://github.com/Devsh-Graphics-Programming/Nabla/issues/854
					FAIL_ON_PIPELINE_COMPILE_REQUIRED = 1<<8,
					EARLY_RETURN_ON_FAILURE = 1<<9,

					// Will be exposed later with the IPipelineLibrary asset implementation
					// https://github.com/Devsh-Graphics-Programming/Nabla/issues/853
					//LINK_TIME_OPTIMIZATION = 1<<10,

					// Won't be exposed because we'll introduce Libraries as a separate object/asset-type
					// https://github.com/Devsh-Graphics-Programming/Nabla/issues/853
					//CREATE_LIBRARY = 1<<11,

					// Ray Tracing Pipelines only
					//SKIP_BUILT_IN_PRIMITIVES = 1<<12,
					//SKIP_AABBS = 1<<13,
					//NO_NULL_ANY_HIT_SHADERS = 1<<14,
					//NO_NULL_CLOSEST_HIT_SHADERS = 1<<15,
					//NO_NULL_MISS_SHADERS = 1<<16,
					//NO_NULL_INTERSECTION_SHADERS = 1<<17,

					// There is a new Device Generated Commands extension with its own flag that will deprecate this
					//INDIRECT_BINDABLE_NV = 1<<18,

					// Ray Tracing Pipelines only
          // For debug tools
					//RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR = 1<<19,

					// Ray Tracing Pipelines only
					//ALLOW_MOTION = 1<<20,

					// Graphics Pipelineonly (we don't support subpass shading)
					//RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR = 1<<21,
					//RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT = 1<<22,

					// Will be exposed later with the IPipelineLibrary asset implementation
					// https://github.com/Devsh-Graphics-Programming/Nabla/issues/853
					//RETAIN_LINK_TIME_OPTIMIZATION_INFO = 1<<23,

					// Ray Tracing Pipelines only
					//RAY_TRACING_OPACITY_MICROMAP_BIT_EXT = 1<<24,

					// Not supported yet, and we will move to dynamic rendering, so this might never be supported
					//COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT = 1<<25,
					//DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT = 1<<26,

					// Not Supported Yet
					//NO_PROTECTED_ACCESS=1<<27,
					//RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV = 1<<28,
					//DESCRIPTOR_VUFFER_BIT=1<<29,
					//PROTECTED_ACCESS_ONLY=1<<30,
				};
		};

		inline const PipelineLayout* getLayout() const {return m_layout.get();}

	protected:
		inline IPipeline(core::smart_refctd_ptr<const PipelineLayout>&& _layout)
      : m_layout(std::move(_layout)) {}

		core::smart_refctd_ptr<const PipelineLayout> m_layout;
};

}
#endif