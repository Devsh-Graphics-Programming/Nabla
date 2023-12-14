// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_I_PIPELINE_H_INCLUDED_


#include "nbl/video/IGPUPipelineLayout.h"


namespace nbl::video
{
//! Interface class for graphics and compute pipelines
/*
	A pipeline refers to a succession of fixed stages 
	through which a data input flows; each stage processes 
	the incoming data and hands it over to the next stage. 
	The final product will be either a 2D raster drawing image 
	(the graphics pipeline) or updated resources (buffers or images) 
	with computational logic and calculations (the compute pipeline).

	Vulkan supports two types of pipeline:
	
	- graphics pipeline
	- compute pipeline
*/
template<typename CRTP>
class IPipeline : public IBackendObject
{
	public:
		// For now, due to API design we implicitly satisfy:
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-08771
		// to:
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pSpecializationInfo-06849
		struct SCreationParams
		{
			public:
				constexpr static inline int32_t NotDerivingFromPreviousPipeline = -1;

				inline bool isDerivative() const
				{
					return basePipelineIndex!=NotDerivingFromPreviousPipeline || basePipeline;
				}

				const IGPUPipelineLayout* layout = nullptr;
				// if you set this, then we don't take `basePipelineIndex` into account
				const CRTP* basePipeline = nullptr;
				int32_t basePipelineIndex = NotDerivingFromPreviousPipeline;

			protected:
				// TODO: split into separate enums per pipeline type!
				enum class FLAGS : uint64_t
				{
					NONE = 0,
					DISABLE_OPTIMIZATIONS = 1<<0,
					ALLOW_DERIVATIVES = 1<<1,
					
					//I can just guess this
					//DERIVATIVE = 1<<2,

					// Graphics Pipelines only
					//VIEW_INDEX_FROM_DEVICE_INDEX = 1<<3,
					
					// Compute Pipelines only
					//DISPATCH_BASE = 1<<4,
					
					// Weird extension
					//DEFER_COMPILE_NV = 1<<5,

					CAPTURE_STATISTICS = 1<<6,
					CAPTURE_INTERNAL_REPRESENTATIONS = 1<<7,
					FAIL_ON_PIPELINE_COMPILE_REQUIRED = 1<<8,
					EARLY_RETURN_ON_FAILURE = 1<<9,
					LINK_TIME_OPTIMIZATION = 1<<10,

					//Not Supported Yet
					//CREATE_LIBRARY = 1<<11,

					// Ray Tracing Pipelines only
					//RAY_TRACING_SKIP_TRIANGLES_BIT_KHR = 1<<12,
					//RAY_TRACING_SKIP_AABBS_BIT_KHR = 1<<13,
					//RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_BIT_KHR = 1<<14,
					//RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_BIT_KHR = 1<<15,
					//RAY_TRACING_NO_NULL_MISS_SHADERS_BIT_KHR = 1<<16,
					//RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_BIT_KHR = 1<<17,

					// Not Supported Yet
					//INDIRECT_BINDABLE_BIT_NV = 1<<18,

					// Ray Tracing Pipelines only
					//RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR = 1<<19,
					//RAY_TRACING_ALLOW_MOTION_BIT_NV = 1<<20,

					// Graphics Pipelineonly (we don't support subpass shading)
					//RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR = 1<<21,
					//RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT = 1<<22,

					RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT = 1<<23,

					// Ray Tracing Pipelines only
					//RAY_TRACING_OPACITY_MICROMAP_BIT_EXT = 1<<24,
					//RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV = 1<<25,

					// Not Supported Yet
					//NO_PROTECTED_ACCESS=1<<26,
					//PROTECTED_ACCESS_ONLY=1<<27,
				};
		};

	protected:
		using IBackendObject::IBackendObject;
};

}
#endif