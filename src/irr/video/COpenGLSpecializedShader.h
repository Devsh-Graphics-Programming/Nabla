#ifndef __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "COpenGLExtensionHandler.h"
#include "irr/video/COpenGLShader.h"
#include "irr/asset/CShaderIntrospector.h"
#include "irr/core/memory/refctd_dynamic_array.h"
#include <mutex>
#include <algorithm>

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

    COpenGLSpecializedShader(size_t _ctxCount, const asset::ICPUBuffer* _spirv, const asset::ISpecializationInfo* _specInfo, const asset::CIntrospectionData* _introspection);

    inline GLuint getGLnameForCtx(const void* _ctx)
    {
        std::unique_lock<std::mutex> lock(m_getGLnameMutex);

        auto found = std::find_if(m_GLnames.begin(), m_GLnames.end(), [&_ctx] (const CtxGLnamePair& n) { return n.first==_ctx; });
        if (found != m_GLnames.end())
            return found->second;

        if (m_binary.binary) {
            const GLuint GLname = COpenGLExtensionHandler::extGlCreateProgram();
            COpenGLExtensionHandler::extGlProgramBinary(GLname, m_binary.format, m_binary.binary->data(), m_binary.binary->size());
            m_GLnames.emplace_back(_ctx, GLname);
            return GLname;
        }

        const GLuint GLname = compile(COpenGLExtensionHandler::ShaderLanguageVersion);
        m_GLnames.emplace_back(_ctx, GLname);
        return GLname;
    }

    void setUniformsImitatingPushConstants(const uint8_t* _pcData, GLuint _GLname);

    inline GLenum getStage() const { return m_stage; }

protected:
    ~COpenGLSpecializedShader()
    {
        //shader programs can be shared so all names can be freed by any thread
        for (auto& n : m_GLnames)
            COpenGLExtensionHandler::extGlDeleteProgram(n.second);
    }

private:
    //! @returns GL name or zero if already compiled once or there were compilation errors.
    GLuint compile(uint32_t _GLSLversion);
    void buildUniformsList(GLuint _GLname);

private:
    using CtxGLnamePair = std::pair<const void*, GLuint>;
    core::vector<CtxGLnamePair> m_GLnames;
    GLenum m_stage;
    //! Held until compilation of shader
    core::smart_refctd_ptr<const asset::ICPUBuffer> m_spirv;
    //! Held until compilation of shader
    core::smart_refctd_ptr<const asset::ISpecializationInfo> m_specInfo;
    //used for setting uniforms ("push constants")
    core::smart_refctd_ptr<asset::CIntrospectionData> m_introspectionData = nullptr;
    SProgramBinary m_binary;

    std::mutex m_getGLnameMutex;

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
