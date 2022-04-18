#include "nbl/video/COpenGLCommandPool.h"

#include "nbl/video/COpenGLCommandBuffer.h"

namespace nbl::video
{

void COpenGLCommandPool::CBeginRenderPassCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf)
{
    auto framebuf = core::smart_refctd_ptr_static_cast<const video::COpenGLFramebuffer>(m_framebuffer);

    ctxlocal->nextState.framebuffer.hash = framebuf->getHashValue();
    ctxlocal->nextState.framebuffer.fbo = std::move(framebuf);
    ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_FRAMEBUFFER, ctxid);

    GLuint fbo = ctxlocal->currentState.framebuffer.GLname;
    if (fbo)
    {
        IGPUCommandBuffer::SRenderpassBeginInfo beginInfo = {};
        beginInfo.clearValueCount = m_clearValueCount;
        beginInfo.clearValues = m_clearValues;
        beginInfo.framebuffer = m_framebuffer;
        beginInfo.renderArea = m_renderArea;
        beginInfo.renderpass = m_renderpass;
        COpenGLCommandBuffer::beginRenderpass_clearAttachments(gl, ctxlocal, ctxid, beginInfo, fbo, logger);
    }

    executingCmdbuf->currentlyRecordingRenderPass = m_renderpass.get();
}

void COpenGLCommandPool::CEndRenderPassCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxLocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf)
{
    ctxLocal->nextState.framebuffer.hash = SOpenGLState::NULL_FBO_HASH;
    ctxLocal->nextState.framebuffer.GLname = 0u;
    ctxLocal->nextState.framebuffer.fbo = nullptr;
    executingCmdbuf->currentlyRecordingRenderPass = nullptr;
}


}