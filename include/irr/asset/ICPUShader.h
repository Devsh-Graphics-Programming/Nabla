#ifndef __IRR_I_CPU_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SHADER_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ISPIR_VProgram.h"
#include "irr/asset/ShaderCommons.h"
#include "irr/asset/ShaderRes.h"
#include "irr/core/memory/new_delete.h"

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
            _IRR_DELETE(m_parsed);
    }

public:
    using SEntryPointStagePair = std::pair<std::string, E_SHADER_STAGE>;

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

    void enableIntrospection();
    const SIntrospectionData* getIntrospectionData(const SEntryPointStagePair& _entryPoint) const
    {
        auto found = m_introspectionCache.find(_entryPoint);
        return found == m_introspectionCache.end() ? nullptr : &found->second;
    }

    const core::vector<SEntryPointStagePair>& getStageEntryPoints();
protected:
    const core::vector<SEntryPointStagePair>& getStageEntryPoints(spirv_cross::Compiler& _comp);

    //! returns pointer to parsed representation of contained SPIR-V code. Returned object is allocated by _IRR_NEW macro.
    spirv_cross::ParsedIR* parseSPIR_V() const;

protected:
    ICPUBuffer* m_spirvBytecode;
    mutable spirv_cross::ParsedIR* m_parsed = nullptr;

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
