#ifndef __IRR_I_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_I_SPECIALIZED_SHADER_H_INCLUDED__

#include <cstdint>


#include "irr/core/Types.h"
#include "irr/asset/ICPUBuffer.h"

namespace irr
{
namespace asset
{


class ISpecializedShader : public virtual core::IReferenceCounted
{
	public:
		enum E_SHADER_STAGE : uint32_t
		{
			ESS_VERTEX = 1 << 0,
			ESS_TESSELATION_CONTROL = 1 << 1,
			ESS_TESSELATION_EVALUATION = 1 << 2,
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
			ESS_UNKNOWN = 0,
			ESS_ALL_GRAPHICS = 0x1f,
			ESS_ALL = 0xffffffff
		};
		class SInfo
		{
			public:
				struct SMapEntry
				{
					uint32_t specConstID;
					uint32_t offset;
					size_t size;
				};
				//! _entries must be sorted!
				SInfo(core::vector<SMapEntry>&& _entries, core::smart_refctd_ptr<ICPUBuffer>&& _backingBuff, const std::string& _entryPoint, E_SHADER_STAGE _ss) :
					m_entries{std::move(_entries)}, m_backingBuffer(std::move(_backingBuff)), entryPoint{_entryPoint}, shaderStage{_ss}
				{
				}
				~SInfo() = default;

				inline std::pair<const void*, size_t> getSpecializationByteValue(uint32_t _specConstID) const
				{
					if (!m_backingBuffer)
						return {nullptr, 0u};

					auto entry = std::lower_bound(m_entries.begin(), m_entries.end(), SMapEntry{_specConstID,0xdeadbeefu,0xdeadbeefu/*To make GCC warnings shut up*/});
					if (entry != m_entries.end() && entry->specConstID == _specConstID && (entry->offset + entry->size) <= m_backingBuffer->getSize())
						return {reinterpret_cast<const uint8_t*>(m_backingBuffer->getPointer()) + entry->offset, entry->size};
					else
						return {nullptr, 0u};
				}

				std::string entryPoint;
				E_SHADER_STAGE shaderStage;
				core::vector<SMapEntry> m_entries; // TODO: change to smart refctd array
				core::smart_refctd_ptr<ICPUBuffer> m_backingBuffer;
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
}

#endif // __IRR_I_SPECIALIZED_SHADER_H_INCLUDED__
