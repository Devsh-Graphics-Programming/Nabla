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
    COpenGLShader(asset::ICPUShader* _cpushader) : m_cpuShader(_cpushader) {}

    asset::ICPUShader* getCPUCounterpart() { return m_cpuShader.get(); }
    const asset::ICPUShader* getCPUCounterpart() const { return m_cpuShader.get(); }

private:
    core::smart_refctd_ptr<asset::ICPUShader> m_cpuShader;
};

}
}

#endif