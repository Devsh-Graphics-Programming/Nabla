#ifndef __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "COpenGLExtensionHandler.h"
#include "irr/video/COpenGLShader.h"
#include "irr/asset/CShaderIntrospector.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{

class COpenGLSpecializedShader : public IGPUSpecializedShader
{
public:
    COpenGLSpecializedShader(const video::IVideoDriver* _driver, const asset::ICPUBuffer* _spirv, const asset::ISpecializationInfo* _specInfo, const asset::CIntrospectionData* _introspection);

    void setUniformsImitatingPushConstants(const uint8_t* _pcData);

    GLuint getOpenGLName() const { return m_GLname; }
    GLenum getStage() const { return m_stage; }

private:
    void buildUniformsList();

private:
    GLuint m_GLname;
    GLenum m_stage;
    //used for setting uniforms ("push constants")
    core::smart_refctd_ptr<asset::CIntrospectionData> m_introspectionData = nullptr;
    using SMember = asset::impl::SShaderMemoryBlock::SMember;
    struct SUniform {
        SMember m;
        GLint location;
    };
    core::vector<SUniform> m_uniformsList;
};

}
}
#endif

#endif//__IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
