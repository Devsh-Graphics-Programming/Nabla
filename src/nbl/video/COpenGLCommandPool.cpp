#include "nbl/video/COpenGLCommandPool.h"

#include "nbl/video/COpenGLCommandBuffer.h"

namespace nbl::video
{

void COpenGLCommandPool::CBindIndexBufferCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf)
{
    auto* buffer = static_cast<const COpenGLBuffer*>(m_indexBuffer.get());
    ctxlocal->nextState.vertexInputParams.vaoval.idxBinding = { m_offset, core::smart_refctd_ptr<const COpenGLBuffer>(buffer) };
    ctxlocal->nextState.vertexInputParams.vaoval.idxType = m_indexType;
}

void COpenGLCommandPool::CDrawCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf)
{
    ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);

    const asset::E_PRIMITIVE_TOPOLOGY primType = ctxlocal->currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
    GLenum glpt = COpenGLCommandBuffer::getGLprimitiveType(primType);

    gl->extGlDrawArraysInstancedBaseInstance(glpt, m_firstVertex, m_vertexCount, m_instanceCount, m_firstInstance);
}

void COpenGLCommandPool::CDrawIndexedCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf)
{
    ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);

    const asset::E_PRIMITIVE_TOPOLOGY primType = ctxlocal->currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
    GLenum glpt = COpenGLCommandBuffer::getGLprimitiveType(primType);
    GLenum idxType = GL_INVALID_ENUM;
    switch (ctxlocal->currentState.vertexInputParams.vaoval.idxType)
    {
    case asset::EIT_16BIT:
        idxType = GL_UNSIGNED_SHORT;
        break;
    case asset::EIT_32BIT:
        idxType = GL_UNSIGNED_INT;
        break;
    default: break;
    }

    if (idxType != GL_INVALID_ENUM)
    {
        const GLuint64 ixsz = idxType == GL_UNSIGNED_INT ? 4u : 2u;

        GLuint64 idxBufOffset = ctxlocal->currentState.vertexInputParams.vaoval.idxBinding.offset + ixsz * m_firstIndex;
        static_assert(sizeof(idxBufOffset) == sizeof(void*), "Bad reinterpret_cast");
        gl->extGlDrawElementsInstancedBaseVertexBaseInstance(glpt, m_indexCount, idxType, reinterpret_cast<void*>(idxBufOffset), m_instanceCount, m_vertexOffset, m_firstInstance);
    }
}

void COpenGLCommandPool::CDrawIndirectCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf)
{
    if (m_maxDrawCount == 0u)
        return;

    ctxlocal->nextState.vertexInputParams.indirectDrawBuf = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(m_buffer);

    ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);

    const asset::E_PRIMITIVE_TOPOLOGY primType = ctxlocal->currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
    GLenum glpt = COpenGLCommandBuffer::getGLprimitiveType(primType);

    GLuint64 offset = m_offset;
    static_assert(sizeof(offset) == sizeof(void*), "Bad reinterpret_cast");
    gl->extGlMultiDrawArraysIndirect(glpt, reinterpret_cast<void*>(offset), m_maxDrawCount, m_stride);
}

void COpenGLCommandPool::CDrawIndirectCountCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf)
{
    if (m_maxDrawCount == 0u)
        return;

    ctxlocal->nextState.vertexInputParams.indirectDrawBuf = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(m_buffer);
    ctxlocal->nextState.vertexInputParams.parameterBuf = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(m_countBuffer);

    ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);

    const asset::E_PRIMITIVE_TOPOLOGY primType = ctxlocal->currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
    GLenum glpt = COpenGLCommandBuffer::getGLprimitiveType(primType);

    GLuint64 offset = m_offset;
    static_assert(sizeof(offset) == sizeof(void*), "Bad reinterpret_cast");
    gl->extGlMultiDrawArraysIndirectCount(glpt, reinterpret_cast<void*>(offset), m_countBufferOffset, m_maxDrawCount, m_stride);
}

void COpenGLCommandPool::CDrawIndexedIndirectCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf)
{
    if (m_maxDrawCount == 0u)
        return;

    ctxlocal->nextState.vertexInputParams.indirectDrawBuf = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(m_buffer);

    ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);

    GLenum idxType = GL_INVALID_ENUM;
    switch (ctxlocal->currentState.vertexInputParams.vaoval.idxType)
    {
    case asset::EIT_16BIT:
        idxType = GL_UNSIGNED_SHORT;
        break;
    case asset::EIT_32BIT:
        idxType = GL_UNSIGNED_INT;
        break;
    default:
        break;
    }

    const asset::E_PRIMITIVE_TOPOLOGY primType = ctxlocal->currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
    GLenum glpt = COpenGLCommandBuffer::getGLprimitiveType(primType);

    GLuint64 offset = m_offset;
    static_assert(sizeof(offset) == sizeof(void*), "Bad reinterpret_cast");
    gl->extGlMultiDrawElementsIndirect(glpt, idxType, reinterpret_cast<void*>(offset), m_maxDrawCount, m_stride);
}

void COpenGLCommandPool::CDrawIndexedIndirectCountCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf)
{
    if (m_maxDrawCount == 0u)
        return;

    ctxlocal->nextState.vertexInputParams.indirectDrawBuf = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(m_buffer);
    ctxlocal->nextState.vertexInputParams.parameterBuf = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(m_countBuffer);

    ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);

    GLenum idxType = GL_INVALID_ENUM;
    switch (ctxlocal->currentState.vertexInputParams.vaoval.idxType)
    {
    case asset::EIT_16BIT:
        idxType = GL_UNSIGNED_SHORT;
        break;
    case asset::EIT_32BIT:
        idxType = GL_UNSIGNED_INT;
        break;
    default:
        break;
    }

    const asset::E_PRIMITIVE_TOPOLOGY primType = ctxlocal->currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
    GLenum glpt = COpenGLCommandBuffer::getGLprimitiveType(primType);

    GLuint64 offset = m_offset;
    static_assert(sizeof(offset) == sizeof(void*), "Bad reinterpret_cast");
    gl->extGlMultiDrawElementsIndirectCount(glpt, idxType, reinterpret_cast<void*>(offset), m_countBufferOffset, m_maxDrawCount, m_stride);
}

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