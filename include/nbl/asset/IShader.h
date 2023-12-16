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
		class SSpecInfo
		{
			public:
				// Nabla requires device's reported subgroup size to be between 4 and 128
				enum class SUBGROUP_SIZE : uint8_t
				{
					// No constraint but probably means `gl_SubgroupSize` is Dynamically Uniform
					UNKNOWN=0,
					// Allows the Subgroup Uniform `gl_SubgroupSize` to be non-Dynamically Uniform and vary between Device's min and max
					VARYING=1,
					// The rest we encode as log2(x) of the required value
					REQUIRE_4=2,
					REQUIRE_8=3,
					REQUIRE_16=4,
					REQUIRE_32=5,
					REQUIRE_64=6,
					REQUIRE_128=7
				};

				//! Structure specifying a specialization map entry
				/*
					Note that if specialization constant ID is used
					in a shader, \bsize\b and \boffset'b must match 
					to \isuch an ID\i accordingly.
				*/
				struct SMapEntry
				{
					uint32_t specConstID;		//!< The ID of the specialization constant in SPIR-V. If it isn't used in the shader, the map entry does not affect the behavior of the pipeline.
					uint32_t offset;			//!< The byte offset of the specialization constant value within the supplied data buffer.		
					size_t size;				//!< The byte size of the specialization constant value within the supplied data buffer.
				
					auto operator<=>(const SMapEntry&) const = default;
				};

				bool operator<(const SSpecializationInfo& _rhs) const
				{
					if (entryPoint==_rhs.entryPoint)
					{
						size_t lhsSize = m_entries ? m_entries->size():0ull;
						size_t rhsSize = _rhs.m_entries ? _rhs.m_entries->size():0ull;
						if (lhsSize==rhsSize)
						{
							for (size_t i=0ull; i<lhsSize; ++i)
							{
								const auto& l = (*m_entries)[i];
								const auto& r = (*_rhs.m_entries)[i];

								if (l.specConstID==r.specConstID)
								{
									if (l.size==r.size)
									{
										int cmp = memcmp(reinterpret_cast<const uint8_t*>(m_backingBuffer->getPointer())+l.offset, reinterpret_cast<const uint8_t*>(_rhs.m_backingBuffer->getPointer())+r.offset, l.size);
										if (cmp==0)
											continue;
										return cmp<0;
									}
									return l.size<r.size;
								}
								return l.specConstID<r.specConstID;
							}
							// all entries equal if we got out the loop
							// return m_filePathHint<_rhs.m_filePathHint; // don't do this cause OpenGL program cache might get more entries in it (I think it contains only already include-resolved shaders)
						}
						return lhsSize<rhsSize;
					}
					return entryPoint<_rhs.entryPoint;
				}

				inline std::pair<const void*, size_t> getSpecializationByteValue(uint32_t _specConstID) const
				{
					if (!m_entries || !m_backingBuffer)
						return {nullptr, 0u};

					auto entry = std::lower_bound(m_entries->begin(), m_entries->end(), SMapEntry{ _specConstID,0xdeadbeefu,0xdeadbeefu/*To make GCC warnings shut up*/},
						[](const SMapEntry& lhs, const SMapEntry& rhs) -> bool
						{
							return lhs.specConstID<rhs.specConstID;
						}
					);
					if (entry != m_entries->end() && entry->specConstID==_specConstID && (entry->offset + entry->size) <= m_backingBuffer->getSize())
						return {reinterpret_cast<const uint8_t*>(m_backingBuffer->getPointer()) + entry->offset, entry->size};
					else
						return {nullptr, 0u};
				}

				//
				core::refctd_dynamic_array<SMapEntry>* getEntries() {return m_entries.get();}
				const core::refctd_dynamic_array<SMapEntry>* getEntries() const {return m_entries.get();}
				
				//
				ICPUBuffer* getBackingBuffer() {return m_backingBuffer.get();}
				const ICPUBuffer* getBackingBuffer() const {return m_backingBuffer.get();}

				//
				void setEntries(core::smart_refctd_dynamic_array<SMapEntry>&& _entries, core::smart_refctd_ptr<ICPUBuffer>&& _backingBuff)
				{
					m_entries = std::move(_entries);
					m_backingBuffer = std::move(_backingBuff);
				}

				std::string entryPoint = "main";									//!< A name of the function where the entry point of an shader executable begins. It's often "main" function.
				const ShaderType* shader;
		core::smart_refctd_dynamic_array<SMapEntry> m_entries;				//!< A specialization map entry
		core::smart_refctd_ptr<ICPUBuffer> m_backingBuffer;					//!< A buffer containing the actual constant values to specialize with
				SUBGROUP_SIZE requiredSubgroupSize : 3 = SUBGROUP_SIZE::UNKNOWN;	//!< Default value of 8 means no requirement
				// Valid only for Compute, Mesh and Task shaders
				uint8_t requireFullSubgroups : 1 = false;
		};

	protected:
		E_SHADER_STAGE m_shaderStage;
		std::string m_filepathHint;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IShader::E_SHADER_STAGE)

}

#endif
