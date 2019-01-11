#ifndef __IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__

#include <cstdint>
#include "irr/core/Types.h"
#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/ISPIR_VProgram.h"

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
    ESS_ALL = 0xffffffff
};
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
    uint32_t size;
};
struct SSpecializationConstants
{
    core::vector<SSpecializationMapEntry> entries;
    ICPUBuffer* backingBuffer = nullptr;
};

struct SIntrospectionData
{
    //
};

class ICPUSpecializedShader : IAsset
{
protected:
    virtual ~ICPUSpecializedShader()
    {

    }

public:
    ICPUSpecializedShader(ISPIR_VProgram* _spirvProgram, const SSpecializationConstants& _spc, const std::string& _ep, E_PIPELINE_CREATION _pipelineCreationBits, E_SHADER_STAGE _type)
        : m_specConstants{_spc}, m_entryPoint{_ep}, m_pipelineCreationBits{_pipelineCreationBits}, m_shaderType{_type}, m_spirvProgram{_spirvProgram}
    {
        if (m_spirvProgram)
            m_spirvProgram->grab();
        if (m_specConstants.backingBuffer)
            m_specConstants.backingBuffer->grab();
    }

    void setSpecializationConstants(const SSpecializationConstants& _spc)
    {
        if (_spc.backingBuffer)
            _spc.backingBuffer->grab();
        if (m_specConstants.backingBuffer)
            m_specConstants.backingBuffer->drop();
        m_specConstants = _spc;
    }
    const SSpecializationConstants& getSpecializationConstants() const { return m_specConstants; }
    void setEntryPoint(const std::string& _ep) { m_entryPoint = _ep; } // could be validation whether it's a valid name within m_spirvProgram (requires retrosepction done)
    const std::string& getEntryPoint() const { return m_entryPoint; }
    void setPipelineCreationBits(E_PIPELINE_CREATION _pcb) { m_pipelineCreationBits = _pcb; }
    uint32_t getPipelineCreationBits() const { return m_pipelineCreationBits; }
    void setShaderType(E_SHADER_STAGE _type) { m_shaderType = _type; }
    E_SHADER_STAGE getShaderType() const { return m_shaderType; }

    const ISPIR_VProgram* getSPIR_VProgram() const { return m_spirvProgram; }

    const SIntrospectionData& enableIntrospection()
    {
        // get introspection result from ISPIR_VCross by passing bytecode to it
    }

private:
    SSpecializationConstants m_specConstants;
    std::string m_entryPoint;
    uint32_t m_pipelineCreationBits;
    E_SHADER_STAGE m_shaderType;
    ISPIR_VProgram* m_spirvProgram;

    SIntrospectionData m_introspectionData;
};

}}

#endif//__IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__
