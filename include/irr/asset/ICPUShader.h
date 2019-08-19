#ifndef __IRR_I_CPU_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SHADER_H_INCLUDED__

#include <algorithm>
#include "irr/asset/IAsset.h"
#include "irr/asset/ISPIR_VProgram.h"
#include "irr/asset/ShaderCommons.h"
#include "irr/asset/ShaderRes.h"
#include "irr/core/memory/new_delete.h"
#include "irr/asset/IParsedShaderSource.h"
#include "irr/asset/IGLSLCompiler.h"

namespace spirv_cross
{
    class ParsedIR;
    class Compiler;
    struct SPIRType;
}
namespace irr 
{ 
namespace asset
{
    using SEntryPointStagePair = std::pair<std::string, E_SHADER_STAGE>;
}//asset
namespace io
{
    class IReadFile;
}//io
}//irr
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
        if (m_glslCompiler)
            m_glslCompiler->drop();
    }

public:
    ICPUShader(ICPUBuffer* _spirv) : m_glslCompiler(nullptr), m_spirvBytecode(_spirv), m_parsed(new IParsedShaderSource(m_spirvBytecode, core::defer)) 
    {
    }
    //! While creating from GLSL source, entry point name and stage must be given in advance
    ICPUShader(IGLSLCompiler* _glslcompiler, io::IReadFile* _glsl, const std::string& _entryPoint, E_SHADER_STAGE _stage);

    IAsset::E_TYPE getAssetType() const override { return IAsset::ET_SHADER; }
    size_t conservativeSizeEstimate() const override 
    { 
        if (m_glsl.size())
            return m_glsl.size();
        else if (m_spirvBytecode)
            return m_spirvBytecode->conservativeSizeEstimate();
        return 0u; //shouldnt ever reach this line
    }
    void convertToDummyObject() override { m_glsl.clear(); }

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
    IGLSLCompiler* m_glslCompiler;
    ICPUBuffer* m_spirvBytecode = nullptr;
    IParsedShaderSource* m_parsed = nullptr;
    std::string m_glsl;
    std::string m_glslOriginFilename;

    using CacheKey = SEntryPointStagePair;
    core::unordered_map<CacheKey, SIntrospectionData> m_introspectionCache;
    core::vector<SEntryPointStagePair> m_entryPoints;

protected:
    struct SIntrospectionPerformer
    {
        SIntrospectionData doIntrospection(spirv_cross::Compiler& _comp, const SEntryPointStagePair& _ep) const;
        void shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID, const core::unordered_map<uint32_t, const SIntrospectionData::SSpecConstant*>& _mapId2sconst) const;
        size_t calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType& _type) const;

        static void deinitIntrospectionData(SIntrospectionData& _data);
        static void deinitShdrMemBlock(impl::SShaderMemoryBlock& _res);
    };
};

}}

#endif//__IRR_I_CPU_SHADER_H_INCLUDED__
