#ifndef __IRR_I_CPU_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SHADER_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ISPIR_VProgram.h"
#include "irr/asset/ShaderCommons.h"
#include "irr/asset/ShaderRes.h"
#include "irr/core/memory/new_delete.h"
#include "irr/asset/IParsedShaderSource.h"

namespace spirv_cross
{
    class ParsedIR;
    class Compiler;
    struct SPIRType;
}

namespace irr { namespace asset
{

struct SIntrospectionData
{
    struct SSpecConstant
    {
        uint32_t id;
        size_t byteSize;
        std::string name;
    };
    core::vector<SSpecConstant> specConstants;
    core::vector<SShaderResourceVariant> descriptorSetBindings[4];
    core::vector<SShaderInfoVariant> inputOutput;
    struct {
        bool present;
        SShaderPushConstant info;
    } pushConstant;

    bool canSpecializationlesslyCreateDescSetFrom() const
    {
        for (const auto& descSet : descriptorSetBindings)
        {
            auto found = std::find_if(descSet.begin(), descSet.end(), [](const SShaderResourceVariant& bnd) { return bnd.descCountIsSpecConstant; });
            if (found != descSet.end())
                return false;
        }
        return true;
    }
};

class ICPUShader : public IAsset
{
protected:
    virtual ~ICPUShader()
    {
        if (m_spirvBytecode)
            m_spirvBytecode->drop();
        if (m_parsed)
            m_parsed->drop();
    }

public:
    using SEntryPointStagePair = std::pair<std::string, E_SHADER_STAGE>;

    ICPUShader(const void* _spirvBytecode, size_t _bytesize) : m_spirvBytecode(new ICPUBuffer(_bytesize)), m_parsed(new IParsedShaderSource(m_spirvBytecode, core::defer))
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

    void enableIntrospection();
    const SIntrospectionData* getIntrospectionData(const SEntryPointStagePair& _entryPoint) const
    {
        auto found = m_introspectionCache.find(_entryPoint);
        return found == m_introspectionCache.end() ? nullptr : &found->second;
    }

    const core::vector<SEntryPointStagePair>& getStageEntryPoints();

    inline const IParsedShaderSource* getParsed() const
    {
        return m_parsed;
    }

protected:
    const core::vector<SEntryPointStagePair>& getStageEntryPoints(spirv_cross::Compiler& _comp);

protected:
    ICPUBuffer* m_spirvBytecode;
    IParsedShaderSource* m_parsed = nullptr;

    using CacheKey = SEntryPointStagePair;
    core::unordered_map<CacheKey, SIntrospectionData> m_introspectionCache;
    core::vector<SEntryPointStagePair> m_entryPoints;

protected:
    struct SIntrospectionPerformer
    {
        SIntrospectionData doIntrospection(spirv_cross::Compiler& _comp, const SEntryPointStagePair& _ep) const;
        void shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID) const;
        size_t calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType& _type) const;
    };
};

}}

#endif//__IRR_I_CPU_SHADER_H_INCLUDED__
