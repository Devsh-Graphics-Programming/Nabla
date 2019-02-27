#ifndef __IRR_I_CPU_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SHADER_H_INCLUDED__

#include <algorithm>
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
    using SEntryPointStagePair = std::pair<std::string, E_SHADER_STAGE>;
}}//irr::asset
namespace std
{
    template<>
    struct hash<irr::asset::SEntryPointStagePair>
    {
        using T = irr::asset::SEntryPointStagePair;
        // based on boost::hash_combine
        size_t operator()(const T& x) const
        {
            size_t seed = hash<T::first_type>{}(x.first);
            return seed ^= hash<underlying_type_t<T::second_type>>{}(x.second) + 0x9e3779b9ull + (seed << 6) + (seed >> 2);
        }
    };
}//std

namespace irr { namespace asset
{

struct SIntrospectionData
{
    struct SSpecConstant
    {
        enum E_TYPE
        {
            ET_U64,
            ET_I64,
            ET_U32,
            ET_I32,
            ET_F64,
            ET_F32
        };

        uint32_t id;
        size_t byteSize;
        E_TYPE type;
        std::string name;
        union {
            uint64_t u64;
            int64_t i64;
            uint32_t u32;
            int32_t i32;
            double f64;
            float f32;
        } defaultValue;
    };
    //! Sorted by `id`
    core::vector<SSpecConstant> specConstants;
    //! Each vector is sorted by `binding`
    core::vector<SShaderResourceVariant> descriptorSetBindings[4];
    //! Sorted by `location`
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
        for (auto& introData : m_introspectionCache)
            SIntrospectionPerformer::deinitIntrospectionData(introData.second);
    }

public:
    ICPUShader(const void* _spirvBytecode, size_t _bytesize) : m_spirvBytecode(new ICPUBuffer(_bytesize)), m_parsed(new IParsedShaderSource(m_spirvBytecode, core::defer))
    {
        memcpy(m_spirvBytecode->getPointer(), _spirvBytecode, _bytesize);
    }

    IAsset::E_TYPE getAssetType() const override { return IAsset::ET_SHADER; }
    size_t conservativeSizeEstimate() const override { return m_spirvBytecode ? m_spirvBytecode->conservativeSizeEstimate() : 0u; }
    void convertToDummyObject() override { }

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

        static void deinitIntrospectionData(SIntrospectionData& _data);
        static void deinitShdrMemBlock(impl::SShaderMemoryBlock& _res);
    };
};

}}

#endif//__IRR_I_CPU_SHADER_H_INCLUDED__
