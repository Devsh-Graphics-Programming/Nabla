#ifndef __IRR_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"

namespace irr { namespace video
{

class IGPUSpecializedShader : public core::IReferenceCounted
{
protected:
    virtual ~IGPUSpecializedShader() = default;
};

}}

#endif//__IRR_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__

