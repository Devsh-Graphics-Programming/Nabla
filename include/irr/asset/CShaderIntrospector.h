#ifndef __IRR_C_SHADER_INTROSPECTOR_H_INCLUDED__
#define __IRR_C_SHADER_INTROSPECTOR_H_INCLUDED__

#include <cstdint>
#include "irr/core/Types.h"
#include "irr/asset/ShaderCommons.h"
#include "irr/asset/ShaderRes.h"
#include "irr/asset/ICPUShader.h"
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

class CIntrospectionData : public core::IReferenceCounted
{
protected:
    ~CIntrospectionData();

public:
    struct SSpecConstant
    {
        uint32_t id;
        size_t byteSize;
        E_GLSL_VAR_TYPE type;
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

class CShaderIntrospector : public core::Uncopyable
{
public:
    struct SEntryPoint_Stage_Extensions
    {
        std::string entryPoint;
        E_SHADER_STAGE stage;
        core::smart_refctd_dynamic_array<std::string> GLSLextensions;
    };

    //In the future there's also going list of enabled extensions
    CShaderIntrospector(const IGLSLCompiler* _glslcomp, const SEntryPoint_Stage_Extensions& _params) : m_glslCompiler(_glslcomp,core::dont_grab), m_params(_params) {}

    const CIntrospectionData* introspect(const ICPUShader* _shader);
private:
    core::smart_refctd_ptr<CIntrospectionData> doIntrospection(spirv_cross::Compiler& _comp, const SEntryPoint_Stage_Extensions& _ep) const;
    void shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID, const core::unordered_map<uint32_t, const CIntrospectionData::SSpecConstant*>& _mapId2sconst) const;
    size_t calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType& _type) const;

private:
    core::smart_refctd_ptr<const IGLSLCompiler> m_glslCompiler;
    SEntryPoint_Stage_Extensions m_params;
    core::unordered_map<core::smart_refctd_ptr<const ICPUShader>, core::smart_refctd_ptr<CIntrospectionData>> m_introspectionCache;
};

}//asset
}//irr

#endif