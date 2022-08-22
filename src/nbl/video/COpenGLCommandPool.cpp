#include "nbl/video/COpenGLCommandPool.h"

#include "nbl/video/COpenGLCommandBuffer.h"

namespace nbl::video
{

void COpenGLCommandPool::CBindFramebufferCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint GLname = 0u;
    if (m_fboHash != SOpenGLState::NULL_FBO_HASH)
    {
        auto* found = fboCache.get(m_fboHash);
        if (found)
        {
            GLname = *found;
        }
        else
        {
            GLname = m_fbo->createGLFBO(gl);
            if (GLname)
                fboCache.insert(m_fboHash, GLname);
        }

        assert(GLname != 0u); // TODO uncomment this
    }

    gl->glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, GLname);
}

void COpenGLCommandPool::CClearNamedFramebufferCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint fbo = 0u;
    if (m_fboHash != SOpenGLState::NULL_FBO_HASH)
    {
        auto* found = fboCache.get(m_fboHash);
        if (!found)
            return; // TODO(achal): Log warning?

        fbo = *found;

        const GLfloat depth = m_clearValue.depthStencil.depth;
        const GLint stencil = m_clearValue.depthStencil.stencil;

        switch (m_bufferType)
        {
        case GL_COLOR:
        {
            if (asset::isFloatingPointFormat(m_format) || asset::isNormalizedFormat(m_format))
            {
                const GLfloat* colorf = m_clearValue.color.float32;
                gl->extGlClearNamedFramebufferfv(fbo, m_bufferType, m_drawBufferIndex, colorf);
            }
            else if (asset::isIntegerFormat(m_format))
            {
                if (asset::isSignedFormat(m_format))
                {
                    const GLint* colori = m_clearValue.color.int32;
                    gl->extGlClearNamedFramebufferiv(fbo, m_bufferType, m_drawBufferIndex, colori);
                }
                else
                {
                    const GLuint* coloru = m_clearValue.color.uint32;
                    gl->extGlClearNamedFramebufferuiv(fbo, m_bufferType, m_drawBufferIndex, coloru);
                }
            }
        } break;

        case GL_DEPTH:
        {
            gl->extGlClearNamedFramebufferfv(fbo, m_bufferType, 0, &depth);
        } break;

        case GL_STENCIL:
        {
            gl->extGlClearNamedFramebufferiv(fbo, m_bufferType, 0, &stencil);
        } break;

        case GL_DEPTH_STENCIL:
        {
            gl->extGlClearNamedFramebufferfi(fbo, m_bufferType, 0, depth, stencil);
        } break;

        default:
            assert(!"Invalid Code Path.");
        }
    }
}

void COpenGLCommandPool::CViewportArrayVCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlViewportArrayv(m_first, m_count, m_params);
}

void COpenGLCommandPool::CDepthRangeArrayVCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlDepthRangeArrayv(m_first, m_count, m_params);
}

void COpenGLCommandPool::CPolygonModeCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlPolygonMode(GL_FRONT_AND_BACK, m_mode);
}

void COpenGLCommandPool::CEnableCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glGeneral.pglEnable(m_cap);
}

void COpenGLCommandPool::CDisableCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glGeneral.pglDisable(m_cap);
}

void COpenGLCommandPool::CCullFaceCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglCullFace(m_mode);
}

void COpenGLCommandPool::CStencilOpSeparateCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glFragment.pglStencilOpSeparate(m_face, m_sfail, m_dpfail, m_dppass);
}

void COpenGLCommandPool::CStencilFuncSeparateCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glFragment.pglStencilFuncSeparate(m_face, m_func, m_ref, m_mask);
}

void COpenGLCommandPool::CStencilMaskSeparateCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glFragment.pglStencilMaskSeparate(m_face, m_mask);
}

void COpenGLCommandPool::CDepthFuncCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglDepthFunc(m_func);
}

void COpenGLCommandPool::CFrontFaceCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglFrontFace(m_mode);
}

void COpenGLCommandPool::CPolygonOffsetCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglPolygonOffset(m_factor, m_units);
}

void COpenGLCommandPool::CLineWidthCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglLineWidth(m_width);
}

void COpenGLCommandPool::CMinSampleShadingCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlMinSampleShading(m_value);
}

void COpenGLCommandPool::CSampleMaskICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glFragment.pglSampleMaski(m_maskNumber, m_mask);
}

void COpenGLCommandPool::CDepthMaskCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglDepthMask(m_flag);
}

void COpenGLCommandPool::CLogicOpCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlLogicOp(m_opcode);
}

void COpenGLCommandPool::CEnableICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlEnablei(m_cap, m_index);
}

void COpenGLCommandPool::CDisableICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlDisablei(m_cap, m_index);
}

void COpenGLCommandPool::CBlendFuncSeparateICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlBlendFuncSeparatei(m_buf, m_srcRGB, m_dstRGB, m_srcAlpha, m_dstAlpha);
}

void COpenGLCommandPool::CColorMaskICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlColorMaski(m_buf, m_red, m_green, m_blue, m_alpha);
}

void COpenGLCommandPool::CMemoryBarrierCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glSync.pglMemoryBarrier(m_barrierBits);
}

void COpenGLCommandPool::CUseProgramComputeCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    const COpenGLComputePipeline* glppln = static_cast<const COpenGLComputePipeline*>(m_pipeline.get());
    const GLuint GLname = glppln->getShaderGLnameForCtx(0u, ctxid);
    gl->glShader.pglUseProgram(GLname);
}

void COpenGLCommandPool::CDispatchComputeCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache::fbo_cache_t& fboCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glCompute.pglDispatchCompute(m_numGroupsX, m_numGroupsY, m_numGroupsZ);
}

}