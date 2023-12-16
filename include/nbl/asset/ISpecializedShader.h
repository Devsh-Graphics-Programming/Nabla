// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_SPECIALIZED_SHADER_H_INCLUDED_
#define _NBL_ASSET_I_SPECIALIZED_SHADER_H_INCLUDED_

#include "nbl/core/decl/Types.h"
#include "nbl/core/SRange.h"
#include "nbl/core/containers/refctd_dynamic_array.h"

#include <cstdint>
#include <filesystem>
#include <compare>

#include "nbl/asset/ICPUBuffer.h"

namespace nbl::asset
{

//! Interface class for Specialized Shaders
/*
	Specialized shaders are shaders prepared to be attached at pipeline
	process creation. SpecializedShader consists of unspecialzed Shader
	containing GLSL code or unspecialized SPIR-V, and creation information
	parameters such as entry point to a shader, stage of a shader and
	specialization map entry - that's why it's called specialized.

	@see IShader
	@see IReferenceCounted

	It also handles Specialization Constants.

	In Vulkan, all shaders get halfway-compiled into SPIR-V and
	then the rest-of-the-way compiled by the Vulkan driver.
	Normally, the half-way compile finalizes all constant values
	and compiles the code that uses them.
	But, it would be nice every so often to have your Vulkan
	program sneak into the halfway-compiled binary and
	manipulate some constants at runtime. This is what
	Specialization Constants are for. 
	
	So A Specialization Constant is
	a way of injecting an integer, Boolean, uint, float, or double
	constant into a halfway-compiled version of a shader right
	before the rest-of-the-way compilation.

	Without Specialization Constants, you would have to commit
	to a final value before the SPIR-V compile was done, which
	could have been a long time ago.
*/

class ISpecializedShader : public virtual core::IReferenceCounted // TODO: do we need this inheritance?
{
	public:
		//! Parameter class used in constructor of ISpecializedShader
		/*
			It holds shader stage type, specialization map entry, entry
			point of the shader and backing buffer.
		*/

		class SInfo
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

				SInfo() = default;
				//! _entries must be sorted!
				SInfo(core::smart_refctd_dynamic_array<SMapEntry>&& _entries,
					core::smart_refctd_ptr<ICPUBuffer>&& _backingBuff,
					const std::string& _entryPoint)
					: entryPoint{_entryPoint}
				{
					setEntries(std::move(_entries),std::move(_backingBuff));
				}
				~SInfo() = default;

				bool operator<(const SInfo& _rhs) const
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
				core::smart_refctd_dynamic_array<SMapEntry> m_entries;				//!< A specialization map entry
				core::smart_refctd_ptr<ICPUBuffer> m_backingBuffer;					//!< A buffer containing the actual constant values to specialize with
				SUBGROUP_SIZE m_requiredSubgroupSize : 3 = SUBGROUP_SIZE::UNKNOWN;	//!< Default value of 8 means no requirement
				// Valid only for Compute, Mesh and Task shaders
				uint8_t m_requireFullSubgroups : 1 = false;
		};

	protected:
		ISpecializedShader() = default;
		virtual ~ISpecializedShader() = default;
};

inline bool operator<(const ISpecializedShader::SInfo::SMapEntry& _a, const ISpecializedShader::SInfo::SMapEntry& _b)
{
    return _a.specConstID < _b.specConstID;
}

}

#endif
