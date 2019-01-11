#ifndef __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/video/IGPUSpecializedShader.h"

namespace irr { namespace video
{

class COpenGLSpecializedShader : public IGPUSpecializedShader
{
public:
    COpenGLSpecializedShader(const asset::ISPIR_VProgram* _spirvProgram, const asset::SSpecializationConstants& _spc, const std::string& _ep, asset::E_PIPELINE_CREATION _pipelineCreationBits, asset::E_SHADER_STAGE _type)
    {
        // manipulate spir-v (specialize spec. constants)
        // down-compile to GLSL (SPIRV-Cross)
        // manipulation 2nd pass (on GLSL code)
        // feed to OpenGL and get GL name
    }
    COpenGLSpecializedShader(asset::ICPUSpecializedShader* _cpushader) :
        COpenGLSpecializedShader(_cpushader->getSPIR_VProgram(), _cpushader->getSpecializationConstants(), _cpushader->getEntryPoint(), static_cast<asset::E_PIPELINE_CREATION>(_cpushader->getPipelineCreationBits()), _cpushader->getShaderType())
    {}

    uint32_t getOpenGLName() const { return m_GLname; }

private:
    uint32_t m_GLname;
};

}}


#endif//__IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
