// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_SPIR_V_PROGRAM_H_INCLUDED__
#define __NBL_ASSET_I_SPIR_V_PROGRAM_H_INCLUDED__

#include "nbl/asset/ICPUBuffer.h"

namespace nbl
{
namespace asset
{
class ISPIR_VProgram : public core::IReferenceCounted
{
protected:
    virtual ~ISPIR_VProgram() = default;

public:
    ISPIR_VProgram(core::smart_refctd_ptr<ICPUBuffer>&& _bytecode)
        : m_bytecode{_bytecode}
    {
    }

    const ICPUBuffer* getBytecode() const { return m_bytecode.get(); }

protected:
    core::smart_refctd_ptr<ICPUBuffer> m_bytecode;
};

}
}

#endif
