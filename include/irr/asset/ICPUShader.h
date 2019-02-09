#ifndef __IRR_I_CPU_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SHADER_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ISPIR_VProgram.h"
#include "irr/asset/ShaderCommons.h"

namespace spirv_cross
{
    class ParsedIR;
}

namespace irr { namespace asset
{

struct SIntrospectionData
{
    //
};

class ICPUShader : public IAsset
{
protected:
    virtual ~ICPUShader()
    {
        if (m_spirvBytecode)
            m_spirvBytecode->drop();
        if (m_parsed)
            delete m_parsed;
    }

public:
    ICPUShader(const void* _spirvBytecode, size_t _bytesize) : m_spirvBytecode(new ICPUBuffer(_bytesize))
    {
        memcpy(m_spirvBytecode->getPointer(), _spirvBytecode, _bytesize);
    }

    IAsset::E_TYPE getAssetType() const override { return IAsset::ET_SHADER; }
    size_t conservativeSizeEstimate() const override { return m_spirvBytecode ? m_spirvBytecode->conservativeSizeEstimate() : 0u; }
    void convertToDummyObject() override 
    { 
        if (m_spirvBytecode)
            m_spirvBytecode->drop();
    }

    const ICPUBuffer* getSPIR_VBytecode() const { return m_spirvBytecode; };

    const SIntrospectionData& enableIntrospection(const std::string& _entryPoint, E_SHADER_STAGE _stage);

protected:
    ICPUBuffer* m_spirvBytecode;
    spirv_cross::ParsedIR* m_parsed = nullptr;

    using CacheKey = std::pair<std::string, E_SHADER_STAGE>;
    std::unordered_map<CacheKey, SIntrospectionData> m_introspectionCache;
};

}}

#endif//__IRR_I_CPU_SHADER_H_INCLUDED__
