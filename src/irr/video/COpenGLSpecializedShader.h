#ifndef __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "COpenGLExtensionHandler.h"
#include "irr/video/COpenGLShader.h"
#include "irr/asset/CShaderIntrospector.h"
#include "irr/core/memory/refctd_dynamic_array.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{

class COpenGLSpecializedShader : public IGPUSpecializedShader
{
public:
    struct SProgramBinary {
        GLenum format;
        core::smart_refctd_dynamic_array<uint8_t> binary;
    };

    COpenGLSpecializedShader(const asset::ICPUBuffer* _spirv, const asset::ISpecializationInfo* _specInfo, const asset::CIntrospectionData* _introspection);

    //! @returns GL name or zero if already compiled once or there were compilation errors.
    GLuint compile(uint32_t _GLSLversion);

    void setUniformsImitatingPushConstants(const uint8_t* _pcData, GLuint _GLname);

    inline const SProgramBinary& getProgramBinary() const { return m_binary; }

    GLenum getStage() const { return m_stage; }

private:
    void buildUniformsList(GLuint _GLname);

private:
    GLenum m_stage;
    //! Held until compilation of shader
    core::smart_refctd_ptr<const asset::ICPUBuffer> m_spirv;
    //! Held until compilation of shader
    core::smart_refctd_ptr<const asset::ISpecializationInfo> m_specInfo;
    //used for setting uniforms ("push constants")
    core::smart_refctd_ptr<asset::CIntrospectionData> m_introspectionData = nullptr;
    SProgramBinary m_binary;

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
