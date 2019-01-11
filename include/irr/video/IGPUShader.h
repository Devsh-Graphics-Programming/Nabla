#ifndef __IRR_I_GPU_SHADER_H_INCLUDED__
#define __IRR_I_GPU_SHADER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/ISPIR_VProgram.h"

namespace irr { namespace video
{

class IGPUShader : public core::IReferenceCounted
{
protected:
    virtual ~IGPUShader()
    {
        if (m_spirvProgram)
            m_spirvProgram->drop();
    }

public:
    IGPUShader(asset::ISPIR_VProgram* _spirvProgram) : m_spirvProgram{_spirvProgram}
    {
        if (m_spirvProgram)
            m_spirvProgram->grab();
    }

    const asset::ISPIR_VProgram* getSPIR_VProgram() const { return m_spirvProgram; }

protected:
    asset::ISPIR_VProgram* m_spirvProgram;
};

}}

#endif//__IRR_I_GPU_SHADER_H_INCLUDED__
