#ifndef __IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUShader.h"
#include "irr/asset/ShaderCommons.h"

namespace irr { namespace asset
{

class ICPUSpecializedShader : IAsset
{
protected:
    virtual ~ICPUSpecializedShader()
    {
        if (m_specInfo)
            m_specInfo->drop();
        if (m_spirvBytecode)
            m_spirvBytecode->drop();
    }

public:
    ICPUSpecializedShader(ICPUShader* _unspecialized, const SSpecializationInfo* _spc)
        : m_specInfo{_spc}, m_spirvBytecode{_unspecialized->getSPIR_VBytecode()}
    {
        if (m_specInfo)
            m_specInfo->grab();
        if (m_spirvBytecode)
            m_spirvBytecode->grab();
    }

    const SSpecializationInfo* getSpecializationInfo() const { return m_specInfo; }

private:
    const SSpecializationInfo* m_specInfo;
    const ICPUBuffer* m_spirvBytecode;
    SIntrospectionData m_introspectionData;
};

}}

#endif//__IRR_I_CPU_SPECIALIZED_SHADER_H_INCLUDED__
