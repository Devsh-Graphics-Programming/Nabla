#ifndef __IRR_C_OPENGL_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SHADER_H_INCLUDED__

#include "irr/video/IGPUShader.h"
#include "irr/asset/ICPUShader.h"

namespace irr
{
namespace video
{

class COpenGLShader : public IGPUShader
{
public:
    COpenGLShader(core::smart_refctd_ptr<const asset::ICPUShader>&& _cpushader) : m_cpuShader(std::move(_cpushader)) {}

    const asset::ICPUShader* getCPUCounterpart() const { return m_cpuShader.get(); }

private:
    core::smart_refctd_ptr<const asset::ICPUShader> m_cpuShader;
};

}
}

#endif