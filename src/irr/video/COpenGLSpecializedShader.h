#ifndef __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "COpenGLExtensionHandler.h"
#include "irr/video/COpenGLShader.h"
#include "irr/asset/CShaderIntrospector.h"
#include "irr/core/memory/refctd_dynamic_array.h"
#include "irr/video/IGPUMeshBuffer.h"
#include <algorithm>

#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{

class COpenGLSpecializedShader : public IGPUSpecializedShader
{
    struct SProgramBinary {
        GLenum format;
        core::smart_refctd_dynamic_array<uint8_t> binary;
    };

public:
    COpenGLSpecializedShader(size_t _ctxCount, uint32_t _ctxID, uint32_t _GLSLversion, const asset::ICPUBuffer* _spirv, const asset::ISpecializationInfo* _specInfo, const asset::CIntrospectionData* _introspection);

    inline GLuint getGLnameForCtx(uint32_t _ctxID) const
    {
        if ((*m_GLnames)[_ctxID])
            return (*m_GLnames)[_ctxID];

        const GLuint GLname = COpenGLExtensionHandler::extGlCreateProgram();
        COpenGLExtensionHandler::extGlProgramBinary(GLname, m_binary.format, m_binary.binary->data(), m_binary.binary->size());
        (*m_GLnames)[_ctxID] = GLname;
        return GLname;
    }

    void setUniformsImitatingPushConstants(const uint8_t* _pcData, uint32_t _ctxID) const;

    inline GLenum getOpenGLStage() const { return m_GLstage; }

protected:
    ~COpenGLSpecializedShader()
    {
        //shader programs can be shared so all names can be freed by any thread
        for (auto& n : *m_GLnames)
            COpenGLExtensionHandler::extGlDeleteProgram(n);
    }

private:
    //! @returns GL name or zero if already compiled once or there were compilation errors.
    GLuint compile(uint32_t _GLSLversion, const asset::ICPUBuffer* _spirv, const asset::ISpecializationInfo* _specInfo);
    void buildUniformsList(GLuint _GLname);

private:
    mutable core::smart_refctd_dynamic_array<GLuint> m_GLnames;
    GLenum m_GLstage;
    //used for setting uniforms ("push constants")
    core::smart_refctd_ptr<const asset::CIntrospectionData> m_introspectionData = nullptr;
    SProgramBinary m_binary;

    using SMember = asset::impl::SShaderMemoryBlock::SMember;
    struct SUniform {
        SUniform(const SMember& _m, GLint _loc, uint8_t* _valptr) : m(_m), location(_loc), value(_valptr) {}
        SMember m;
        GLint location;
        uint8_t* value;
    };
    mutable bool m_uniformsSetForTheVeryFirstTime = true;
    uint8_t m_uniformValues[IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE];
    core::vector<SUniform> m_uniformsList;
};

}
}
#endif

#endif//__IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
