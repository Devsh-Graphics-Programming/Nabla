#ifndef __IRR_I_GPU_SHADER_H_INCLUDED__
#define __IRR_I_GPU_SHADER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/ISPIR_VProgram.h"

namespace irr { namespace video
{

class IGPUShader : public core::IReferenceCounted
{
protected:
    virtual ~IGPUShader() = default;
};

}}

#endif//__IRR_I_GPU_SHADER_H_INCLUDED__
