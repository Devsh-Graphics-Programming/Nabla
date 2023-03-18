#include "nbl/video/COpenGLCommandPool.h"

#include "nbl/video/COpenGLCommandBuffer.h"

namespace nbl::video
{

static GLuint getFBOGLName(const COpenGLImage* image, const uint32_t level, const uint32_t layer, SOpenGLContextDependentCache& queueCache, IOpenGL_FunctionTable* gl, const bool isColorImage)
{
    COpenGLFramebuffer::hash_t hash;
    if (isColorImage)
        hash = COpenGLFramebuffer::getHashColorImage(image, level, layer);
    else
        hash = COpenGLFramebuffer::getHashDepthStencilImage(image, level, layer);

    GLuint GLName;
    auto found = queueCache.fboCache.get(hash);
    if (found)
    {
        GLName = *found;
    }
    else
    {
        if (isColorImage)
            GLName = COpenGLFramebuffer::getNameColorImage(gl, image, level, layer);
        else
            GLName = COpenGLFramebuffer::getNameDepthStencilImage(gl, image, level, layer);

        if (GLName)
            queueCache.fboCache.insert(hash, GLName);
    }

    return GLName;
};

void COpenGLCommandPool::CBindFramebufferCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint GLname = 0u;
    if (m_fboHash != SOpenGLState::NULL_FBO_HASH)
    {
        auto* found = queueLocalCache.fboCache.get(m_fboHash);
        if (found)
        {
            GLname = *found;
        }
        else
        {
            GLname = m_fbo->createGLFBO(gl);
            if (GLname)
                queueLocalCache.fboCache.insert(m_fboHash, GLname);
        }

        assert(GLname != 0u); // TODO uncomment this
    }

    gl->glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, GLname);
}

void COpenGLCommandPool::CBlitNamedFramebufferCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint srcfbo = getFBOGLName(m_srcImage, m_srcLevel, m_srcLayer, queueCache, gl, true);
    if (!srcfbo)
        return;

    GLuint dstfbo = getFBOGLName(m_dstImage, m_dstLevel, m_dstLayer, queueCache, gl, true);
    if (!dstfbo)
        return;

    GLint sx0 = m_srcOffsets[0].x;
    GLint sy0 = m_srcOffsets[0].y;
    GLint sx1 = m_srcOffsets[1].x;
    GLint sy1 = m_srcOffsets[1].y;
    GLint dx0 = m_dstOffsets[0].x;
    GLint dy0 = m_dstOffsets[0].y;
    GLint dx1 = m_dstOffsets[1].x;
    GLint dy1 = m_dstOffsets[1].y;
    gl->extGlBlitNamedFramebuffer(srcfbo, dstfbo, sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, GL_COLOR_BUFFER_BIT, m_filter == asset::ISampler::ETF_NEAREST ? GL_NEAREST : GL_LINEAR);
}

void COpenGLCommandPool::CClearNamedFramebufferCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint fbo = 0u;
    if (m_fboHash != SOpenGLState::NULL_FBO_HASH)
    {
        auto* found = queueLocalCache.fboCache.get(m_fboHash);
        if (!found)
            return;

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

void COpenGLCommandPool::CViewportArrayVCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlViewportArrayv(m_first, m_count, m_params);
}

void COpenGLCommandPool::CDepthRangeArrayVCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlDepthRangeArrayv(m_first, m_count, m_params);
}

void COpenGLCommandPool::CPolygonModeCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlPolygonMode(GL_FRONT_AND_BACK, m_mode);
}

void COpenGLCommandPool::CEnableCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glGeneral.pglEnable(m_cap);
}

void COpenGLCommandPool::CDisableCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glGeneral.pglDisable(m_cap);
}

void COpenGLCommandPool::CCullFaceCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglCullFace(m_mode);
}

void COpenGLCommandPool::CStencilOpSeparateCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glFragment.pglStencilOpSeparate(m_face, m_sfail, m_dpfail, m_dppass);
}

void COpenGLCommandPool::CStencilFuncSeparateCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glFragment.pglStencilFuncSeparate(m_face, m_func, m_ref, m_mask);
}

void COpenGLCommandPool::CStencilMaskSeparateCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glFragment.pglStencilMaskSeparate(m_face, m_mask);
}

void COpenGLCommandPool::CDepthFuncCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglDepthFunc(m_func);
}

void COpenGLCommandPool::CFrontFaceCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglFrontFace(m_mode);
}

void COpenGLCommandPool::CPolygonOffsetCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglPolygonOffset(m_factor, m_units);
}

void COpenGLCommandPool::CLineWidthCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglLineWidth(m_width);
}

void COpenGLCommandPool::CMinSampleShadingCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlMinSampleShading(m_value);
}

void COpenGLCommandPool::CSampleMaskICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glFragment.pglSampleMaski(m_maskNumber, m_mask);
}

void COpenGLCommandPool::CDepthMaskCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglDepthMask(m_flag);
}

void COpenGLCommandPool::CLogicOpCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlLogicOp(m_opcode);
}

void COpenGLCommandPool::CEnableICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlEnablei(m_cap, m_index);
}

void COpenGLCommandPool::CDisableICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlDisablei(m_cap, m_index);
}

void COpenGLCommandPool::CBlendFuncSeparateICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlBlendFuncSeparatei(m_buf, m_srcRGB, m_dstRGB, m_srcAlpha, m_dstAlpha);
}

void COpenGLCommandPool::CColorMaskICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlColorMaski(m_buf, m_red, m_green, m_blue, m_alpha);
}

void COpenGLCommandPool::CMemoryBarrierCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glSync.pglMemoryBarrier(m_barrierBits);
}

void COpenGLCommandPool::CBindPipelineComputeCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    const GLuint GLname = m_glppln->getShaderGLnameForCtx(0u, ctxid);
    gl->glShader.pglUseProgram(GLname);
}

void COpenGLCommandPool::CDispatchComputeCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glCompute.pglDispatchCompute(m_numGroupsX, m_numGroupsY, m_numGroupsZ);
}

void COpenGLCommandPool::CDispatchComputeIndirectCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glCompute.pglDispatchComputeIndirect(m_indirect);
}

void COpenGLCommandPool::CSetUniformsImitatingPushConstantsComputeCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    const auto* pcState = queueLocalCache.pushConstantsState<asset::EPBP_COMPUTE>();
    assert(pcState);
    m_pipeline->setUniformsImitatingPushConstants(gl, ctxid, *pcState);
}

void COpenGLCommandPool::CSetUniformsImitatingPushConstantsGraphicsCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    const auto* pcState = queueLocalCache.pushConstantsState<asset::EPBP_GRAPHICS>();
    assert(pcState);
    m_pipeline->setUniformsImitatingPushConstants(gl, ctxid, *pcState);
}

void COpenGLCommandPool::CBindBufferCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glBuffer.pglBindBuffer(m_target, m_bufferGLName);
}

void COpenGLCommandPool::CBindImageTexturesCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlBindImageTextures(m_first, m_count, m_textures, m_formats);
}

void COpenGLCommandPool::CBindTexturesCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlBindTextures(m_first, m_count, m_textures, m_targets);
}

void COpenGLCommandPool::CBindSamplersCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlBindSamplers(m_first, m_count, m_samplers);
}

void COpenGLCommandPool::CBindBuffersRangeCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlBindBuffersRange(m_target, m_first, m_count, m_buffers, m_offsets, m_sizes);
}

void COpenGLCommandPool::CNamedBufferSubDataCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlNamedBufferSubData(m_bufferGLName, m_offset, m_size, m_data);
}

void COpenGLCommandPool::CResetQueryCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    m_queryPool->setLastQueueToUseForQuery(m_query, ctxid);
}

void COpenGLCommandPool::CQueryCounterCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint query = m_queryPool->getQueryAt(ctxid, m_query);
    gl->glQuery.pglQueryCounter(query, m_target);
}

void COpenGLCommandPool::CBeginQueryCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint query = m_queryPool->getQueryAt(ctxid, m_query);
    gl->glQuery.pglBeginQuery(m_target, query);
}

void COpenGLCommandPool::CEndQueryCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glQuery.pglEndQuery(m_target);
}

void COpenGLCommandPool::CGetQueryBufferObjectUICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    // COpenGLQueryPool::lastQueueToUseArray is set on the worker thread so it is important to retrieve its value on the worker thread as well, we cannot
    // do it on the main thread at command record time.
    const uint32_t lastQueueToUse = m_queryPool->getLastQueueToUseForQuery(m_queryIdx);

    if (ctxid != lastQueueToUse)
        return;

    GLuint query = m_queryPool->getQueryAt(lastQueueToUse, m_queryIdx);

    if (query == GL_NONE)
        return;

    if (m_use64Version)
        gl->extGlGetQueryBufferObjectui64v(query, m_buffer, m_pname, m_offset);
    else
        gl->extGlGetQueryBufferObjectuiv(query, m_buffer, m_pname, m_offset);
}

void COpenGLCommandPool::CBindPipelineGraphicsCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint pipelineGLName;

    auto found = queueLocalCache.graphicsPipelineCache.getMap().find(m_pipeline);
    if (found != queueLocalCache.graphicsPipelineCache.getMap().end())
    {
        pipelineGLName = found->second;
    }
    else
    {
        {
            constexpr size_t STAGE_CNT = COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT;
            static_assert(STAGE_CNT == 5u, "SHADER_STAGE_COUNT is expected to be 5");
            const GLenum stages[5]{ GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
            const GLenum stageFlags[5]{ GL_VERTEX_SHADER_BIT, GL_TESS_CONTROL_SHADER_BIT, GL_TESS_EVALUATION_SHADER_BIT, GL_GEOMETRY_SHADER_BIT, GL_FRAGMENT_SHADER_BIT };

            gl->glShader.pglGenProgramPipelines(1u, &pipelineGLName);

            auto pipelineHash = m_pipeline->getPipelineHash(ctxid);

            for (uint32_t ix = 0u; ix < STAGE_CNT; ++ix)
            {
                GLuint progName = pipelineHash[ix];
                if (progName)
                    gl->glShader.pglUseProgramStages(pipelineGLName, stageFlags[ix], progName);
            }
        }
        queueLocalCache.graphicsPipelineCache.getMap().insert({ m_pipeline, pipelineGLName });
    }

    gl->glShader.pglBindProgramPipeline(pipelineGLName);

    // this needs to be here to make sure interleaving the same compute pipeline with the same gfx pipeline works.
    // From the spec:
    // Warning: glUseProgram overrides glBindProgramPipeline.
    // That is, if you have a program in use and a program pipeline bound, all rendering
    // will use the program that is in use, not the pipeline programs.
    // So make sure that glUseProgram(0) has been called.
    gl->glShader.pglUseProgram(0);
}

void COpenGLCommandPool::CBindVertexArrayCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    bool brandNewVAO = false;//if VAO is taken from cache we don't have to modify VAO state that is part of hashval (everything except index and vertex buf bindings)

    auto it = queueLocalCache.vaoCache.get(m_vaoKey);
    GLuint vaoGLName;
    if (it)
    {
        vaoGLName = *it;
    }
    else
    {
        gl->extGlCreateVertexArrays(1u, &vaoGLName);
        queueLocalCache.vaoCache.insert(m_vaoKey, vaoGLName);
        brandNewVAO = true;
    }

    gl->glVertex.pglBindVertexArray(vaoGLName);

    bool updatedBindings[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT]{};
    for (uint32_t attr = 0u; attr < asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; ++attr)
    {
        if ((m_vaoKey.attribFormatAndComponentCount[attr] != asset::EF_UNKNOWN) && brandNewVAO)
        {
            gl->extGlEnableVertexArrayAttrib(vaoGLName, attr);
        }
        else
        {
            continue;
        }

        const uint32_t bnd = m_vaoKey.getBindingForAttrib(attr);

        if (brandNewVAO)
        {
            gl->extGlVertexArrayAttribBinding(vaoGLName, attr, bnd);

            const auto format = static_cast<asset::E_FORMAT>(m_vaoKey.attribFormatAndComponentCount[attr]);
            if (!gl->isGLES() && asset::isFloatingPointFormat(format) && asset::getTexelOrBlockBytesize(format) == asset::getFormatChannelCount(format) * sizeof(double))//DOUBLE
            {
                gl->extGlVertexArrayAttribLFormat(vaoGLName, attr, asset::getFormatChannelCount(format), GL_DOUBLE, m_vaoKey.getRelativeOffsetForAttrib(attr));
            }
            else if (asset::isFloatingPointFormat(format) || asset::isScaledFormat(format) || asset::isNormalizedFormat(format))//FLOATING-POINT, SCALED ("weak integer"), NORMALIZED
            {
                gl->extGlVertexArrayAttribFormat(vaoGLName, attr, asset::isBGRALayoutFormat(format) ? GL_BGRA : asset::getFormatChannelCount(format), formatEnumToGLenum(gl, format), asset::isNormalizedFormat(format) ? GL_TRUE : GL_FALSE, m_vaoKey.getRelativeOffsetForAttrib(attr));
            }
            else if (asset::isIntegerFormat(format))//INTEGERS
            {
                gl->extGlVertexArrayAttribIFormat(vaoGLName, attr, asset::getFormatChannelCount(format), formatEnumToGLenum(gl, format), m_vaoKey.getRelativeOffsetForAttrib(attr));
            }

            if (!updatedBindings[bnd])
            {
                gl->extGlVertexArrayBindingDivisor(vaoGLName, bnd, m_vaoKey.getDivisorForBinding(bnd));
                updatedBindings[bnd] = true;
            }
        }
    }
}

void COpenGLCommandPool::CVertexArrayVertexBufferCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    auto it = queueLocalCache.vaoCache.get(m_vaoKey);
    if (!it)
        return;

    GLuint vaoGLName = *it;
    gl->extGlVertexArrayVertexBuffer(vaoGLName, m_bindingIndex, m_bufferGLName, m_offset, m_stride);
}

void COpenGLCommandPool::CVertexArrayElementBufferCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    auto it = queueLocalCache.vaoCache.get(m_vaoKey);
    if (!it)
        return;

    GLuint vaoGLName = *it;
    gl->extGlVertexArrayElementBuffer(vaoGLName, m_bufferGLName);
}

void COpenGLCommandPool::CPixelStoreICmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glShader.pglPixelStorei(m_pname, m_param);
}

void COpenGLCommandPool::CDrawArraysInstancedBaseInstanceCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlDrawArraysInstancedBaseInstance(m_mode, m_first, m_count, m_instancecount, m_baseinstance);
}

void COpenGLCommandPool::CDrawElementsInstancedBaseVertexBaseInstanceCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlDrawElementsInstancedBaseVertexBaseInstance(m_mode, m_count, m_type, reinterpret_cast<void*>(m_idxBufOffset), m_instancecount, m_basevertex, m_baseinstance);
}

void COpenGLCommandPool::CCopyNamedBufferSubDataCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlCopyNamedBufferSubData(m_readBufferGLName, m_writeBufferGLName, m_readOffset, m_writeOffset, m_size);
}

void COpenGLCommandPool::CCompressedTextureSubImage2DCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlCompressedTextureSubImage2D(m_texture, m_target, m_level, m_xoffset, m_yoffset, m_width, m_height, m_format, m_imageSize, m_data);
}

void COpenGLCommandPool::CCompressedTextureSubImage3DCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlCompressedTextureSubImage3D(m_texture, m_target, m_level, m_xoffset, m_yoffset, m_zoffset, m_width, m_height, m_depth, m_format, m_imageSize, m_data);
}

void COpenGLCommandPool::CTextureSubImage2DCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlTextureSubImage2D(m_texture, m_target, m_level, m_xoffset, m_yoffset, m_width, m_height, m_format, m_type, m_pixels);
}

void COpenGLCommandPool::CTextureSubImage3DCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlTextureSubImage3D(m_texture, m_target, m_level, m_xoffset, m_yoffset, m_zoffset, m_width, m_height, m_depth, m_format, m_type, m_pixels);
}

void COpenGLCommandPool::CGetCompressedTextureSubImageCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glTexture.pglGetCompressedTextureSubImage(m_texture, m_level, m_xoffset, m_yoffset, m_zoffset, m_width, m_height, m_depth, m_bufSize, reinterpret_cast<void*>(m_bufferOffset));
}

void COpenGLCommandPool::CGetTextureSubImageCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->glTexture.pglGetTextureSubImage(m_texture, m_level, m_xoffset, m_yoffset, m_zoffset, m_width, m_height, m_depth, m_format, m_type, m_bufSize, reinterpret_cast<void*>(m_bufferOffset));
}

void COpenGLCommandPool::CReadPixelsCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint fbo = getFBOGLName(m_image, m_level, m_layer, queueCache, gl, true);

    if (!fbo)
    {
        logger.log("Couldn't retrieve attachment to download image to.", system::ILogger::ELL_ERROR);
        return;
    }

    GLint prevReadFB = 0;
    gl->glGeneral.pglGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevReadFB);
    {
        gl->glFramebuffer.pglBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        gl->glTexture.pglReadPixels(m_x, m_y, m_width, m_height, m_format, m_type, reinterpret_cast<void*>(m_bufOffset));
    }
    gl->glFramebuffer.pglBindFramebuffer(GL_READ_FRAMEBUFFER, prevReadFB);
}

void COpenGLCommandPool::CMultiDrawElementsIndirectCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    static_assert(sizeof(m_indirect) == sizeof(void*), "Bad reinterpret_cast");
    gl->extGlMultiDrawElementsIndirect(m_mode, m_type, reinterpret_cast<void*>(m_indirect), m_drawcount, m_stride);
}

void COpenGLCommandPool::CMultiDrawElementsIndirectCountCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    static_assert(sizeof(m_indirect) == sizeof(void*), "Bad reinterpret_cast");
    gl->extGlMultiDrawElementsIndirectCount(m_mode, m_type, reinterpret_cast<void*>(m_indirect), m_drawcount, m_maxdrawcount, m_stride);
}

void COpenGLCommandPool::CExecuteCommandsCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    for (auto i = 0; i < m_count; ++i)
        static_cast<const COpenGLCommandBuffer*>(m_commandBuffers[i])->executeAll(gl, queueCache, ctxid);
}

void COpenGLCommandPool::CMultiDrawArraysIndirectCountCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    static_assert(sizeof(m_indirect) == sizeof(void*), "Bad reinterpret_cast");
    gl->extGlMultiDrawArraysIndirectCount(m_mode, reinterpret_cast<void*>(m_indirect), m_drawcount, m_maxdrawcount, m_stride);
}

void COpenGLCommandPool::CMultiDrawArraysIndirectCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    static_assert(sizeof(m_indirect) == sizeof(void*), "Bad reinterpret_cast");
    gl->extGlMultiDrawArraysIndirect(m_mode, reinterpret_cast<void*>(m_indirect), m_drawcount, m_stride);
}

void COpenGLCommandPool::CCopyImageSubDataCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlCopyImageSubData(m_srcName, m_srcTarget, m_srcLevel, m_srcX, m_srcY, m_srcZ, m_dstName, m_dstTarget, m_dstLevel, m_dstX, m_dstY, m_dstZ, m_srcWidth, m_srcHeight, m_srcDepth);
}

void COpenGLCommandPool::CClearColorImageCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint fbo = getFBOGLName(m_image, m_level, m_layer, queueCache, gl, true);

    if (!fbo)
    {
        logger.log("Couldn't retrieve FBO to clear.", system::ILogger::ELL_ERROR);
        return;
    }

    const auto format = m_image->getCreationParameters().format;

    const bool is_fp = asset::isNormalizedFormat(format) || asset::isFloatingPointFormat(format);
    bool is_int = false;
    bool is_sint = false;
    if (!is_fp && asset::isIntegerFormat(format))
    {
        is_int = true;
        is_sint = asset::isSignedFormat(format);
    }

    if (is_fp)
    {
        gl->extGlClearNamedFramebufferfv(fbo, GL_COLOR, 0, m_clearColorValue.float32);
    }
    else if (is_int)
    {
        if (is_sint)
        {
            gl->extGlClearNamedFramebufferiv(fbo, GL_COLOR, 0, m_clearColorValue.int32);
        }
        else
        {
            gl->extGlClearNamedFramebufferuiv(fbo, GL_COLOR, 0, m_clearColorValue.uint32);
        }
    }
}

void COpenGLCommandPool::CClearDepthStencilImageCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    GLuint fbo = getFBOGLName(m_image, m_level, m_layer, queueCache, gl, false);

    if (!fbo)
    {
        logger.log("Couldn't retrieve FBO to clear.", system::ILogger::ELL_ERROR);
        return;
    }

    const auto fmt = m_image->getCreationParameters().format;

    const bool is_depth = asset::isDepthOnlyFormat(fmt);
    bool is_stencil = false;
    bool is_depth_stencil = false;
    if (!is_depth)
    {
        is_stencil = asset::isStencilOnlyFormat(fmt);
        if (!is_stencil)
            is_depth_stencil = asset::isDepthOrStencilFormat(fmt);
    }

    if (is_depth)
    {
        gl->extGlClearNamedFramebufferfv(fbo, GL_DEPTH, 0, &m_depthStencilClearValue.depth);
    }
    else if (is_stencil)
    {
        static_assert(sizeof(GLint) == sizeof(m_depthStencilClearValue.stencil), "Bad reinterpret_cast!");
        gl->extGlClearNamedFramebufferiv(fbo, GL_STENCIL, 0, reinterpret_cast<const GLint*>(&m_depthStencilClearValue.stencil));
    }
    else if (is_depth_stencil)
    {
        gl->extGlClearNamedFramebufferfi(fbo, GL_DEPTH_STENCIL, 0, m_depthStencilClearValue.depth, m_depthStencilClearValue.stencil);
    }
}

void COpenGLCommandPool::CClearNamedBufferSubDataCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlClearNamedBufferSubData(m_bufferGLName, m_internalformat, m_offset, m_size, m_format, m_type, &m_data);
}

void COpenGLCommandPool::CGenerateTextureMipmapCmd::operator()(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger)
{
    gl->extGlGenerateTextureMipmap(m_texture, m_target);
}

}