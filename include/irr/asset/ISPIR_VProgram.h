#ifndef __IRR_I_SPIR_V_PROGRAM_H_INCLUDED__
#define __IRR_I_SPIR_V_PROGRAM_H_INCLUDED__

#include "irr/asset/ICPUBuffer.h"

namespace irr { namespace asset
{

class ISPIR_VProgram : public core::IReferenceCounted
{
protected:
    virtual ~ISPIR_VProgram()
    {
        if (m_bytecode)
            m_bytecode->drop();
    }

public:
    ISPIR_VProgram(ICPUBuffer* _bytecode) : m_bytecode{_bytecode}
    {
        if (m_bytecode)
            m_bytecode->grab();
    }

    const ICPUBuffer* getBytecode() const { return m_bytecode; }

protected:
    ICPUBuffer* m_bytecode;
};
}}

#endif//__IRR_I_SPIR_V_PROGRAM_H_INCLUDED__
