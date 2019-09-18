#ifndef __IRR_C_OPENGL_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __IRR_C_OPENGL_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "irr/video/IGPURenderpassIndependentPipeline.h"
#include "COpenGLExtensionHandler.h"
#include "COpenGLSpecializedShader.h"

namespace irr {
namespace video
{

class COpenGLRenderpassIndependentPipeline : public IGPURenderpassIndependentPipeline
{
public:
    using IGPURenderpassIndependentPipeline::IGPURenderpassIndependentPipeline;

private:
    GLuint createGLobject(uint32_t _ctxID)
    {
        static_assert(SHADER_STAGE_COUNT == 5u, "SHADER_STAGE_COUNT is expected to be 5");
        const GLenum stages[SHADER_STAGE_COUNT]{ GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
        const GLenum stageFlags[SHADER_STAGE_COUNT]{ GL_VERTEX_SHADER_BIT, GL_TESS_CONTROL_SHADER_BIT, GL_TESS_EVALUATION_SHADER_BIT, GL_GEOMETRY_SHADER_BIT, GL_FRAGMENT_SHADER_BIT };

        GLuint pipeline = 0u;
        COpenGLExtensionHandler::extGlCreateProgramPipelines(1u, &pipeline);

        for (uint32_t ix = 0u; ix < SHADER_STAGE_COUNT; ++ix) {
            COpenGLSpecializedShader* glshdr = static_cast<COpenGLSpecializedShader*>(m_shaders[ix].get());
            GLuint progName = 0u;

            if (!glshdr || glshdr->getStage() != stages[ix])
                continue;
            progName = glshdr->getGLnameForCtx(_ctxID);

            if (progName)
                COpenGLExtensionHandler::extGlUseProgramStages(pipeline, stageFlags[ix], progName);
        }
        
        return pipeline;
    }
};

}}

#endif
