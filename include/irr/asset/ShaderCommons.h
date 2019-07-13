#ifndef __IRR_SHADER_COMMONS_H_INCLUDED__
#define __IRR_SHADER_COMMONS_H_INCLUDED__

#include <cstdint>
#include "irr/core/Types.h"
#include "irr/asset/ICPUBuffer.h"

namespace irr { namespace asset
{

enum E_SHADER_STAGE : uint32_t
{
    ESS_VERTEX = 1<<0,
    ESS_TESSELATION_CONTROL = 1<<1,
    ESS_TESSELATION_EVALUATION = 1<<2,
    ESS_GEOMETRY = 1<<3,
    ESS_FRAGMENT = 1<<4,
    ESS_COMPUTE = 1<<5,
    ESS_TASK = 1<<6,
    ESS_MESH = 1<<7,
    ESS_RAYGEN = 1<<8,
    ESS_ANY_HIT = 1<<9,
    ESS_CLOSEST_HIT = 1<<10,
    ESS_MISS = 1<<11,
    ESS_INTERSECTION = 1<<12,
    ESS_CALLABLE = 1<<13,
    ESS_UNKNOWN = 0,
    ESS_ALL_GRAPHICS = 0x1f,
    ESS_ALL = 0xffffffff
};
//! might be moved to other file in the future
enum E_PIPELINE_CREATION : uint32_t
{
    EPC_DISABLE_OPTIMIZATIONS = 1<<0,
    EPC_ALLOW_DERIVATIVES = 1<<1,
    EPC_DERIVATIVE = 1<<2,
    EPC_VIEW_INDEX_FROM_DEVICE_INDEX = 1<<3,
    EPC_DISPATCH_BASE = 1<<4,
    EPC_DEFER_COMPILE_NV = 1<<5
};

struct SSpecializationMapEntry
{
    uint32_t specConstID;
    uint32_t offset;
    size_t size;
};
inline bool operator<(const SSpecializationMapEntry& _a, const SSpecializationMapEntry& _b)
{
    return _a.specConstID < _b.specConstID;
}

class ISpecializationInfo : public core::IReferenceCounted
{
protected:
    ~ISpecializationInfo()
    {
        if (m_backingBuffer)
            m_backingBuffer->drop();
    }

public:
    //! _entries must be sorted!
    ISpecializationInfo(core::vector<SSpecializationMapEntry>&& _entries, ICPUBuffer* _backingBuff, const std::string& _entryPoint, E_SHADER_STAGE _ss) : 
        m_entries{std::move(_entries)}, m_backingBuffer{_backingBuff}, entryPoint{_entryPoint}, shaderStage{_ss}
    {
        if (m_backingBuffer)
            m_backingBuffer->grab();
    }

    inline std::pair<const void*, size_t> getSpecializationByteValue(uint32_t _specConstID) const
    {
        if (!m_backingBuffer)
            return {nullptr, 0u};

        auto entry = std::lower_bound(m_entries.begin(), m_entries.end(), SSpecializationMapEntry{_specConstID,0xdeadbeefu,0xdeadbeefu/*To make GCC warnings shut up*/});
        if (entry != m_entries.end() && entry->specConstID == _specConstID && (entry->offset + entry->size) < m_backingBuffer->getSize())
            return {reinterpret_cast<const uint8_t*>(m_backingBuffer->getPointer()) + entry->offset, entry->size};
        else
            return {nullptr, 0u};
    }

public:
    std::string entryPoint;
    E_SHADER_STAGE shaderStage;

private:
    core::vector<SSpecializationMapEntry> m_entries;
    ICPUBuffer* m_backingBuffer = nullptr;
};

}}

#endif//__IRR_SHADER_COMMONS_H_INCLUDED__