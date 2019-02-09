#ifndef __IRR_I_CPU_SHADER_H_INCLUDED__
#define __IRR_I_CPU_SHADER_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ISPIR_VProgram.h"

namespace irr { namespace asset
{

class ICPUShader : public IAsset
{
protected:
    virtual ~ICPUShader()
    {
        if (m_spirvBytecode)
            m_spirvBytecode->drop();
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

protected:
    ICPUBuffer* m_spirvBytecode;
};

}}

#endif//__IRR_I_CPU_SHADER_H_INCLUDED__
