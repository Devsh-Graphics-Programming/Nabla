// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_SHADER_H_INCLUDED_
#define _NBL_ASSET_I_SHADER_H_INCLUDED_


#include "nbl/core/declarations.h"

#include <algorithm>
#include <string>


namespace spirv_cross
{
	class ParsedIR;
	class Compiler;
	struct SPIRType;
}

namespace nbl::asset
{

//! Interface class for Unspecialized Shaders
/*
	The purpose for the class is for storing raw HLSL code
	to be compiled or already compiled (but unspecialized) 
	SPIR-V code.
*/

class IShader : public virtual core::IReferenceCounted // TODO: do we need this inheritance?
{
	public:
		// TODO: make this enum class
		enum E_SHADER_STAGE : uint32_t
		{
			ESS_UNKNOWN = 0,
			ESS_VERTEX = 1 << 0,
			ESS_TESSELLATION_CONTROL = 1 << 1,
			ESS_TESSELLATION_EVALUATION = 1 << 2,
			ESS_GEOMETRY = 1 << 3,
			ESS_FRAGMENT = 1 << 4,
			ESS_COMPUTE = 1 << 5,
			ESS_TASK = 1 << 6,
			ESS_MESH = 1 << 7,
			ESS_RAYGEN = 1 << 8,
			ESS_ANY_HIT = 1 << 9,
			ESS_CLOSEST_HIT = 1 << 10,
			ESS_MISS = 1 << 11,
			ESS_INTERSECTION = 1 << 12,
			ESS_CALLABLE = 1 << 13,
			ESS_ALL_GRAPHICS = 0x0000001F,
			ESS_ALL = 0x7fffffff
		};

		IShader(const E_SHADER_STAGE shaderStage, std::string&& filepathHint)
			: m_shaderStage(shaderStage), m_filepathHint(std::move(filepathHint)) {}

		inline E_SHADER_STAGE getStage() const { return m_shaderStage; }

		inline const std::string& getFilepathHint() const { return m_filepathHint; }
		

		enum class E_CONTENT_TYPE : uint8_t
		{
			ECT_UNKNOWN = 0,
			ECT_GLSL,
			ECT_HLSL,
			ECT_SPIRV,
		};
		
		struct SSpecInfoBase
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
				
			using spec_constant_map_t = core::unordered_map<spec_constant_id_t,SSpecConstantValue>;
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
		template<class ShaderType>
		struct SSpecInfo final : SSpecInfoBase
		{
			inline SSpecConstantValue getSpecializationByteValue(const spec_constant_id_t _specConstID) const
			{
				if (!entries)
					return {nullptr,0u};

				const auto found = entries->find(_specConstID);
				if (found!=entries->end() && bool(found->second))
					return found->second;
				else
					return {nullptr,0u};
			}

			// Returns negative on failure, otherwise the size of the buffer required to reserve for the spec constant data 
			inline int32_t valid() const
			{
				// Impossible to check: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pName-00707
				if (entryPoint.empty())
					return INVALID_SPEC_INFO;
					
				if (!shader)
					return INVALID_SPEC_INFO;
				const auto stage = shader->getStage();

				// Shader stages already checked for validity w.r.t. features enabled, during unspec shader creation, only check:
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-08988
				if (requireFullSubgroups)
				switch (stage)
				{
					case E_SHADER_STAGE::ESS_COMPUTE: [[fallthrough]];
					case E_SHADER_STAGE::ESS_TASK: [[fallthrough]];
					case E_SHADER_STAGE::ESS_MESH:
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

			inline bool equalAllButShader(const SSpecInfo<ShaderType>& other) const
			{
				if (entryPoint != other.entryPoint)
					return false;
				if ((!shader) != (!other.shader))
					return false;
				if (requiredSubgroupSize != other.requiredSubgroupSize)
					return false;
				if (requireFullSubgroups != other.requireFullSubgroups)
					return false;

				if (!entries)
					return !other.entries;
				if (entries->size()!=other.entries->size())
					return false;
				for (const auto& entry : *other.entries)
				{
					const auto found = entries->find(entry.first);
					if (found==entries->end())
						return false;
					if (found->second!=entry.second)
						return false;
				}

				return true;
			}

			inline operator SSpecInfo<const ShaderType>() const
			{
				return SSpecInfo<const ShaderType>{
					.entryPoint = entryPoint,
					.shader = shader,
					.entries = entries,
					.requiredSubgroupSize = requiredSubgroupSize,
					.requireFullSubgroups = requireFullSubgroups,
				};
			}
				

			std::string entryPoint = "main";						//!< A name of the function where the entry point of an shader executable begins. It's often "main" function.
			ShaderType* shader = nullptr;
			// Container choice implicitly satisfies:
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationInfo.html#VUID-VkSpecializationInfo-constantID-04911
			const spec_constant_map_t* entries = nullptr;
			// By requiring Nabla Core Profile features we implicitly satisfy:
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-02784
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-02785
			// Also because our API is sane, it satisfies the following by construction:
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pNext-02754
			SUBGROUP_SIZE requiredSubgroupSize : 3 = SUBGROUP_SIZE::UNKNOWN;	//!< Default value of 8 means no requirement
			// Valid only for Compute, Mesh and Task shaders
			uint8_t requireFullSubgroups : 1 = false;
			static constexpr int32_t INVALID_SPEC_INFO = -1;
		};

	protected:
		E_SHADER_STAGE m_shaderStage;
		std::string m_filepathHint;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IShader::E_SHADER_STAGE)

}

#endif
