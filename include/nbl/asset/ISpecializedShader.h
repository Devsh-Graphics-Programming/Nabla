// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_SPECIALIZED_SHADER_H_INCLUDED__
#define __NBL_ASSET_I_SPECIALIZED_SHADER_H_INCLUDED__

#include <cstdint>

#include "nbl/core/Types.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl
{
namespace asset
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

class ISpecializedShader : public virtual core::IReferenceCounted
{
public:
    //! An enum for specifing shader stage of unspecialized shader passed to the constructor of ISpecializedShader
    /*
			Since unspecialized shader contatins only a code, you have
			to state it's stage.
		*/

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

    //! Parameter class used in constructor of ISpecializedShader
    /*
			It holds shader stage type, specialization map entry, entry
			point of the shader and backing buffer.
		*/

    class SInfo
    {
    public:
        //! Structure specifying a specialization map entry
        /*
					Note that if specialization constant ID is used
					in a shader, \bsize\b and \boffset'b must match 
					to \isuch an ID\i accordingly.
				*/

        struct SMapEntry
        {
            uint32_t specConstID;  //!< The ID of the specialization constant in SPIR-V. If it isn't used in the shader, the map entry does not affect the behavior of the pipeline.
            uint32_t offset;  //!< The byte offset of the specialization constant value within the supplied data buffer.
            size_t size;  //!< The byte size of the specialization constant value within the supplied data buffer.
        };

        SInfo() = default;
        //! _entries must be sorted!
        SInfo(core::smart_refctd_dynamic_array<SMapEntry>&& _entries, core::smart_refctd_ptr<ICPUBuffer>&& _backingBuff, const std::string& _entryPoint, E_SHADER_STAGE _ss, const std::string& _filePathHint = "????")
            : entryPoint{_entryPoint}, shaderStage{_ss}, m_filePathHint(_filePathHint)
        {
            setEntries(std::move(_entries), std::move(_backingBuff));
        }
        ~SInfo() = default;

        bool operator<(const SInfo& _rhs) const
        {
            if(shaderStage == _rhs.shaderStage)
            {
                if(entryPoint == _rhs.entryPoint)
                {
                    size_t lhsSize = m_entries ? m_entries->size() : 0ull;
                    size_t rhsSize = _rhs.m_entries ? _rhs.m_entries->size() : 0ull;
                    if(lhsSize == rhsSize)
                    {
                        for(size_t i = 0ull; i < lhsSize; ++i)
                        {
                            const auto& l = (*m_entries)[i];
                            const auto& r = (*_rhs.m_entries)[i];

                            if(l.specConstID == r.specConstID)
                            {
                                if(l.size == r.size)
                                {
                                    int cmp = memcmp(reinterpret_cast<const uint8_t*>(m_backingBuffer->getPointer()) + l.offset, reinterpret_cast<const uint8_t*>(_rhs.m_backingBuffer->getPointer()) + r.offset, l.size);
                                    if(cmp == 0)
                                        continue;
                                    return cmp < 0;
                                }
                                return l.size < r.size;
                            }
                            return l.specConstID < r.specConstID;
                        }
                        // all entries equal if we got out the loop
                        // return m_filePathHint<_rhs.m_filePathHint; // don't do this cause OpenGL program cache might get more entries in it (I think it contains only already include-resolved shaders)
                    }
                    return lhsSize < rhsSize;
                }
                return entryPoint < _rhs.entryPoint;
            }
            return shaderStage < _rhs.shaderStage;
        }

        inline std::pair<const void*, size_t> getSpecializationByteValue(uint32_t _specConstID) const
        {
            if(!m_entries || !m_backingBuffer)
                return {nullptr, 0u};

            auto entry = std::lower_bound(m_entries->begin(), m_entries->end(), SMapEntry{_specConstID, 0xdeadbeefu, 0xdeadbeefu /*To make GCC warnings shut up*/},
                [](const SMapEntry& lhs, const SMapEntry& rhs) -> bool {
                    return lhs.specConstID < rhs.specConstID;
                });
            if(entry != m_entries->end() && entry->specConstID == _specConstID && (entry->offset + entry->size) <= m_backingBuffer->getSize())
                return {reinterpret_cast<const uint8_t*>(m_backingBuffer->getPointer()) + entry->offset, entry->size};
            else
                return {nullptr, 0u};
        }

        std::string entryPoint;  //!< A name of the function where the entry point of an shader executable begins. It's often "main" function.
        E_SHADER_STAGE shaderStage;  //!< A stage of the unspecialized shader passed to specialized one such as vertex, fragment, geometry shader and more.
        core::smart_refctd_dynamic_array<SMapEntry> m_entries;  //!< A specialization map entry
        core::smart_refctd_ptr<ICPUBuffer> m_backingBuffer;  //!< A buffer containing the actual constant values to specialize with
        std::string m_filePathHint;  //!< Only used to resolve `#include` directives in GLSL (not SPIR-V) shaders
        //
        core::refctd_dynamic_array<SMapEntry>* getEntries() { return m_entries.get(); }
        const core::refctd_dynamic_array<SMapEntry>* getEntries() const { return m_entries.get(); }

        //
        ICPUBuffer* getBackingBuffer() { return m_backingBuffer.get(); }
        const ICPUBuffer* getBackingBuffer() const { return m_backingBuffer.get(); }

        //
        void setEntries(core::smart_refctd_dynamic_array<SMapEntry>&& _entries, core::smart_refctd_ptr<ICPUBuffer>&& _backingBuff)
        {
            m_entries = std::move(_entries);
            m_backingBuffer = std::move(_backingBuff);
        }
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

#endif
