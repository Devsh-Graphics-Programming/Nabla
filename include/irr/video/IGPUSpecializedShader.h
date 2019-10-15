#ifndef __IRR_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/ShaderCommons.h"

namespace irr { namespace video
{

class IGPUSpecializedShader : public core::IReferenceCounted
{
public:
    IGPUSpecializedShader(asset::E_SHADER_STAGE _stage) : m_stage(_stage) {}

    asset::E_SHADER_STAGE getStage() const { return m_stage; }

protected:
    virtual ~IGPUSpecializedShader() = default;

    asset::E_SHADER_STAGE m_stage;
};

}}

#endif//__IRR_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__

