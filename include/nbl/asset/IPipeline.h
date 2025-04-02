// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_PIPELINE_H_INCLUDED_


#include "nbl/asset/IPipelineLayout.h"
#include "nbl/asset/IShader.h"


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
class IPipelineBase
{
	public:
		struct SCreationParams
		{
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
		
		/*
			Specialization info contains things such as entry point to a shader,
			specialization map entry, required subgroup size, etc. for a blob of SPIR-V

			It also handles Specialization Constants.

			In Vulkan, all shaders get halfway-compiled into SPIR-V and
			then then lowered (compiled) into the HW ISA by the Vulkan driver.
			Normally, the half-way compile folds all constant values
			and optimizes the code that uses them.

			But, it would be nice every so often to have your Vulkan
			program sneak into the halfway-compiled SPIR-V binary and
			manipulate some constants at runtime. This is what
			Specialization Constants are for.

			So A Specialization Constant is a way of injecting an integer
			constant into a halfway-compiled version of a shader right
			before the lowering and linking when creating a pipeline.

			Without Specialization Constants, you would have to commit
			to a final value before the SPIR-V compilation
		*/
		struct SShaderSpecInfo final
		{
			//! Structure specifying a specialization map entry
			/*
				Note that if specialization constant ID is used
				in a shader, \bsize\b and \boffset'b must match 
				to \isuch an ID\i accordingly.

				By design the API satisfies:
				https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationInfo.html#VUID-VkSpecializationInfo-offset-00773
				https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationInfo.html#VUID-VkSpecializationInfo-pMapEntries-00774
			*/
			//!< The ID of the specialization constant in SPIR-V. If it isn't used in the shader, the map entry does not affect the behavior of the pipeline.
			using spec_constant_id_t = uint32_t;
			struct SSpecConstantValue
			{
				const void* data = nullptr;
				//!< The byte size of the specialization constant value within the supplied data buffer.
				uint32_t size = 0;

				inline operator bool() const {return data&&size;}
				
				auto operator<=>(const SSpecConstantValue&) const = default;
			};
			inline SSpecConstantValue getSpecializationByteValue(const spec_constant_id_t _specConstID) const
			{
				if (!entries)
					return { nullptr,0u };

				const auto found = entries->find(_specConstID);
				if (found != entries->end() && bool(found->second))
					return found->second;
				else
					return { nullptr,0u };
			}

			// Nabla requires device's reported subgroup size to be between 4 and 128
			enum class SUBGROUP_SIZE : uint8_t
			{
				// No constraint but probably means `gl_SubgroupSize` is Dynamically Uniform
				UNKNOWN = 0,
				// Allows the Subgroup Uniform `gl_SubgroupSize` to be non-Dynamically Uniform and vary between Device's min and max
				VARYING = 1,
				// The rest we encode as log2(x) of the required value
				REQUIRE_4 = 2,
				REQUIRE_8 = 3,
				REQUIRE_16 = 4,
				REQUIRE_32 = 5,
				REQUIRE_64 = 6,
				REQUIRE_128 = 7
			};

			//
			static constexpr int32_t INVALID_SPEC_INFO = -1;
			// Returns negative on failure, otherwise the size of the buffer required to reserve for the spec constant data 
			inline int32_t valid() const
			{
				if (!shader || hlsl::bitCount(stage)!=1)
					return INVALID_SPEC_INFO;

				// Impossible to check: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pName-00707
				if (entryPoint.empty())
					return INVALID_SPEC_INFO;

				// Shader stages already checked for validity w.r.t. features enabled, during unspec shader creation, only check:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-08988
				if (requireFullSubgroups)
				switch (stage)
				{
					case hlsl::ShaderStage::ESS_COMPUTE: [[fallthrough]];
					case hlsl::ShaderStage::ESS_TASK: [[fallthrough]];
					case hlsl::ShaderStage::ESS_MESH:
						break;
					default:
						return INVALID_SPEC_INFO;
						break;
				}
				// Impossible to efficiently check anything from:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-maxClipDistances-00708
				// to:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-06686
				// and from:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pNext-02756
				// to:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-module-08987
					
				int64_t specData = 0;
				if (entries)
				for (const auto& entry : *entries)
				{
					if (!entry.second)
						return INVALID_SPEC_INFO;
					specData += entry.second.size;
				}
				if (specData>0x7fffffff)
					return INVALID_SPEC_INFO;
				return static_cast<int32_t>(specData);
			}

			using spec_constant_map_t = core::unordered_map<spec_constant_id_t,SSpecConstantValue>;

			const IShader* shader = nullptr;
			// A name of the function where the entry point of an shader executable begins. It's often "main" function.
			std::string_view entryPoint = {};
			// stage must be set
			hlsl::ShaderStage stage = hlsl::ShaderStage::ESS_UNKNOWN;
			// there's some padding here
			SUBGROUP_SIZE requiredSubgroupSize : 3 = SUBGROUP_SIZE::UNKNOWN;	//!< Default value of 8 means no requirement
			// Valid only for Compute, Mesh and Task shaders
			uint8_t requireFullSubgroups : 1 = false;
			// Container choice implicitly satisfies:
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationInfo.html#VUID-VkSpecializationInfo-constantID-04911
			const spec_constant_map_t* entries = nullptr;
			// By requiring Nabla Core Profile features we implicitly satisfy:
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-02784
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-02785
			// Also because our API is sane, it satisfies the following by construction:
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pNext-02754
		};
};
template<typename PipelineLayout>
class IPipeline : public IPipelineBase
{
	public:
		// For now, due to API design we implicitly satisfy a bunch of VUIDs
		struct SCreationParams : protected IPipelineBase::SCreationParams
		{
			public:
				const PipelineLayout* layout = nullptr;
		};

		inline const PipelineLayout* getLayout() const {return m_layout.get();}

	protected:
		inline IPipeline(core::smart_refctd_ptr<const PipelineLayout>&& _layout)
      : m_layout(std::move(_layout)) {}

		core::smart_refctd_ptr<const PipelineLayout> m_layout;
};

}
#endif