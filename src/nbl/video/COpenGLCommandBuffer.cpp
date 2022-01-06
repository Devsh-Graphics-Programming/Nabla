#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IOpenGL_LogicalDevice.h"

#include "nbl/video/COpenGLCommandBuffer.h"
#include "nbl/video/COpenGLQueryPool.h"
#include "nbl/video/COpenGLCommon.h"

namespace nbl::video
{

COpenGLCommandBuffer::COpenGLCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, E_LEVEL lvl, IGPUCommandPool* _cmdpool, system::logger_opt_smart_ptr&& logger, const COpenGLFeatureMap* _features)
    : IGPUCommandBuffer(std::move(dev), lvl, _cmdpool), m_logger(std::move(logger)), m_features(_features)
{
}

COpenGLCommandBuffer::~COpenGLCommandBuffer()
{
    freeSpaceInCmdPool();
}

    void COpenGLCommandBuffer::freeSpaceInCmdPool()
    {
        auto* pool = getGLCommandPool();
        for (auto& cmd : m_commands)
        {
            switch (cmd.type)
            {
            case impl::ECT_SET_VIEWPORT:
            {
                auto& c = cmd.get<impl::ECT_SET_VIEWPORT>();
                pool->free_n<asset::SViewport>(c.viewports, c.viewportCount);
            }
            break;
            case impl::ECT_COPY_BUFFER:
            {
                auto& c = cmd.get<impl::ECT_COPY_BUFFER>();
                pool->free_n<asset::SBufferCopy>(c.regions, c.regionCount);
            }
            break;
            case impl::ECT_COPY_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_COPY_IMAGE>();
                pool->free_n<asset::IImage::SImageCopy>(c.regions, c.regionCount);
            }
            break;
            case impl::ECT_COPY_BUFFER_TO_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_COPY_BUFFER_TO_IMAGE>();
                pool->free_n<asset::IImage::SBufferCopy>(c.regions, c.regionCount);
            }
            break;
            case impl::ECT_COPY_IMAGE_TO_BUFFER:
            {
                auto& c = cmd.get<impl::ECT_COPY_IMAGE_TO_BUFFER>();
                pool->free_n<asset::IImage::SBufferCopy>(c.regions, c.regionCount);
            }
            break;
            case impl::ECT_BLIT_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_BLIT_IMAGE>();
                pool->free_n<asset::SImageBlit>(c.regions, c.regionCount);
            }
            break;
            case impl::ECT_RESOLVE_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_RESOLVE_IMAGE>();
                pool->free_n<asset::SImageResolve>(c.regions, c.regionCount);
            }
            break;
            case impl::ECT_SET_SCISSORS:
            {
                auto& c = cmd.get<impl::ECT_SET_SCISSORS>();
                pool->free_n<VkRect2D>(c.scissors, c.scissorCount);
            }
            break;
            case impl::ECT_CLEAR_COLOR_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_CLEAR_COLOR_IMAGE>();
                pool->free_n<asset::IImage::SSubresourceRange>(c.ranges, c.rangeCount);
            }
            break;
            case impl::ECT_CLEAR_DEPTH_STENCIL_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_CLEAR_DEPTH_STENCIL_IMAGE>();
                pool->free_n<asset::IImage::SSubresourceRange>(c.ranges, c.rangeCount);
            }
            break;
            case impl::ECT_CLEAR_ATTACHMENTS:
            {
                auto& c = cmd.get<impl::ECT_CLEAR_ATTACHMENTS>();
                pool->free_n<asset::SClearAttachment>(c.attachments, c.attachmentCount);
                pool->free_n<asset::SClearRect>(c.rects, c.rectCount);
            }
            break;
            case impl::ECT_UPDATE_BUFFER:
            {
                auto& c = cmd.get<impl::ECT_UPDATE_BUFFER>();
                pool->free_n<uint8_t>(reinterpret_cast<const uint8_t*>(c.data), c.dataSize);
            }
            break;
            case impl::ECT_BEGIN_RENDERPASS:
            {
                auto& c = cmd.get<impl::ECT_BEGIN_RENDERPASS>();
                if (c.renderpassBegin.clearValueCount > 0u)
                    pool->free_n<asset::SClearValue>(c.renderpassBegin.clearValues, c.renderpassBegin.clearValueCount);
            }
            break;
            default: break; // other commands dont use cmd pool
            }
        }
    }

    bool COpenGLCommandBuffer::reset(uint32_t _flags)
    {
        if (!(m_cmdpool->getCreationFlags() & IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT))
            return false;

        freeSpaceInCmdPool();
        m_commands.clear();
        IGPUCommandBuffer::reset(_flags);

        return true;
    }

    void COpenGLCommandBuffer::copyBufferToImage(const SCmd<impl::ECT_COPY_BUFFER_TO_IMAGE>& c, IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid)
    {
        IGPUImage* dstImage = c.dstImage.get();
        const IGPUBuffer* srcBuffer = c.srcBuffer.get();
        if (!dstImage->validateCopies(c.regions, c.regions + c.regionCount, srcBuffer))
            return;

        const auto params = dstImage->getCreationParameters();
        const auto type = params.type;
        const auto format = params.format;
        const bool compressed = asset::isBlockCompressionFormat(format);
        auto dstImageGL = static_cast<COpenGLImage*>(dstImage);
        GLuint dst = dstImageGL->getOpenGLName();
        GLenum glfmt, gltype;
        getOpenGLFormatAndParametersFromColorFormat(gl, format, glfmt, gltype);

        const auto bpp = asset::getBytesPerPixel(format);
        const auto blockDims = asset::getBlockDimensions(format);

        ctxlocal->nextState.pixelUnpack.buffer = core::smart_refctd_ptr<const COpenGLBuffer>(static_cast<const COpenGLBuffer*>(srcBuffer));
        for (auto it = c.regions; it != c.regions + c.regionCount; it++)
        {
            // TODO: check it->bufferOffset is aligned to data type of E_FORMAT
            //assert(?);

            uint32_t pitch = ((it->bufferRowLength ? it->bufferRowLength : it->imageExtent.width) * bpp).getIntegerApprox();
            int32_t alignment = 0x1 << core::min(core::min<uint32_t>(core::findLSB(it->bufferOffset),core::findLSB(pitch)), 3u);
            ctxlocal->nextState.pixelUnpack.alignment = alignment;
            ctxlocal->nextState.pixelUnpack.rowLength = it->bufferRowLength;
            ctxlocal->nextState.pixelUnpack.imgHeight = it->bufferImageHeight;

            if (compressed)
            {
                ctxlocal->nextState.pixelUnpack.BCwidth = blockDims[0];
                ctxlocal->nextState.pixelUnpack.BCheight = blockDims[1];
                ctxlocal->nextState.pixelUnpack.BCdepth = blockDims[2];

                ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_PIXEL_PACK_UNPACK, ctxid);

                uint32_t imageSize = pitch;
                switch (type)
                {
                case IGPUImage::ET_1D:
                    imageSize *= it->imageSubresource.layerCount;
                    gl->extGlCompressedTextureSubImage2D(dst, GL_TEXTURE_1D_ARRAY, it->imageSubresource.mipLevel,
                        it->imageOffset.x, it->imageSubresource.baseArrayLayer,
                        it->imageExtent.width, it->imageSubresource.layerCount,
                        dstImageGL->getOpenGLSizedFormat(), imageSize, reinterpret_cast<const void*>(it->bufferOffset));
                    break;
                case IGPUImage::ET_2D:
                    imageSize *= (it->bufferImageHeight ? it->bufferImageHeight : it->imageExtent.height);
                    imageSize *= it->imageSubresource.layerCount;
                    gl->extGlCompressedTextureSubImage3D(dst, GL_TEXTURE_2D_ARRAY, it->imageSubresource.mipLevel,
                        it->imageOffset.x, it->imageOffset.y, it->imageSubresource.baseArrayLayer,
                        it->imageExtent.width, it->imageExtent.height, it->imageSubresource.layerCount,
                        dstImageGL->getOpenGLSizedFormat(), imageSize, reinterpret_cast<const void*>(it->bufferOffset));
                    break;
                case IGPUImage::ET_3D:
                    imageSize *= (it->bufferImageHeight ? it->bufferImageHeight : it->imageExtent.height);
                    imageSize *= it->imageExtent.depth;
                    gl->extGlCompressedTextureSubImage3D(dst, GL_TEXTURE_3D, it->imageSubresource.mipLevel,
                        it->imageOffset.x, it->imageOffset.y, it->imageOffset.z,
                        it->imageExtent.width, it->imageExtent.height, it->imageExtent.depth,
                        dstImageGL->getOpenGLSizedFormat(), imageSize, reinterpret_cast<const void*>(it->bufferOffset));
                    break;
                }
            }
            else
            {
                ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_PIXEL_PACK_UNPACK, ctxid);
                switch (type)
                {
                case IGPUImage::ET_1D:
                    gl->extGlTextureSubImage2D(dst, GL_TEXTURE_1D_ARRAY, it->imageSubresource.mipLevel,
                        it->imageOffset.x, it->imageSubresource.baseArrayLayer,
                        it->imageExtent.width, it->imageSubresource.layerCount,
                        glfmt, gltype, reinterpret_cast<const void*>(it->bufferOffset));
                    break;
                case IGPUImage::ET_2D:
                    gl->extGlTextureSubImage3D(dst, GL_TEXTURE_2D_ARRAY, it->imageSubresource.mipLevel,
                        it->imageOffset.x, it->imageOffset.y, it->imageSubresource.baseArrayLayer,
                        it->imageExtent.width, it->imageExtent.height, it->imageSubresource.layerCount,
                        glfmt, gltype, reinterpret_cast<const void*>(it->bufferOffset));
                    break;
                case IGPUImage::ET_3D:
                    gl->extGlTextureSubImage3D(dst, GL_TEXTURE_3D, it->imageSubresource.mipLevel,
                        it->imageOffset.x, it->imageOffset.y, it->imageOffset.z,
                        it->imageExtent.width, it->imageExtent.height, it->imageExtent.depth,
                        glfmt, gltype, reinterpret_cast<const void*>(it->bufferOffset));
                    break;
                }
            }
        }
    }

    void COpenGLCommandBuffer::copyImageToBuffer(const SCmd<impl::ECT_COPY_IMAGE_TO_BUFFER>& c, IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid)
    {
        const auto* srcImage = c.srcImage.get();
        auto* dstBuffer = c.dstBuffer.get();
        if (!srcImage->validateCopies(c.regions, c.regions + c.regionCount, dstBuffer))
            return;

        const auto params = srcImage->getCreationParameters();
        const auto type = params.type;
        const auto format = params.format;
        const uint32_t fmtBytesize = nbl::asset::getTexelOrBlockBytesize(format);
        const bool compressed = asset::isBlockCompressionFormat(format);
        GLuint src = static_cast<const COpenGLImage*>(srcImage)->getOpenGLName();
        GLenum glfmt, gltype;
        getOpenGLFormatAndParametersFromColorFormat(gl, format, glfmt, gltype);

        const auto bpp = asset::getBytesPerPixel(format);
        const auto blockDims = asset::getBlockDimensions(format);

        const bool usingGetTexSubImage = (gl->features->Version >= 450 || gl->features->FeatureAvailable[gl->features->EOpenGLFeatures::NBL_ARB_get_texture_sub_image]);

        GLint prevReadFB = 0;
        if (!usingGetTexSubImage)
            gl->glGeneral.pglGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevReadFB);

        ctxlocal->nextState.pixelPack.buffer = core::smart_refctd_ptr<const COpenGLBuffer>(static_cast<COpenGLBuffer*>(dstBuffer));
        for (auto it = c.regions; it != c.regions + c.regionCount; it++)
        {
            // TODO: check it->bufferOffset is aligned to data type of E_FORMAT
            //assert(?);

            uint32_t pitch = ((it->bufferRowLength ? it->bufferRowLength : it->imageExtent.width) * bpp).getIntegerApprox();
            int32_t alignment = 0x1 << core::min(core::max(core::findLSB(it->bufferOffset), core::findLSB(pitch)), 3u);
            ctxlocal->nextState.pixelPack.alignment = alignment;
            ctxlocal->nextState.pixelPack.rowLength = it->bufferRowLength;
            ctxlocal->nextState.pixelPack.imgHeight = it->bufferImageHeight;

            auto yStart = type == IGPUImage::ET_1D ? it->imageSubresource.baseArrayLayer : it->imageOffset.y;
            auto yRange = type == IGPUImage::ET_1D ? it->imageSubresource.layerCount : it->imageExtent.height;
            auto zStart = type == IGPUImage::ET_2D ? it->imageSubresource.baseArrayLayer : it->imageOffset.z;
            auto zRange = type == IGPUImage::ET_2D ? it->imageSubresource.layerCount : it->imageExtent.depth;

            if (usingGetTexSubImage)
            {
                if (compressed)
                {
                    ctxlocal->nextState.pixelPack.BCwidth = blockDims[0];
                    ctxlocal->nextState.pixelPack.BCheight = blockDims[1];
                    ctxlocal->nextState.pixelPack.BCdepth = blockDims[2];
                }

                ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_PIXEL_PACK_UNPACK, ctxid);

                if (compressed)
                {
                    gl->glTexture.pglGetCompressedTextureSubImage(src, it->imageSubresource.mipLevel, it->imageOffset.x, yStart, zStart, it->imageExtent.width, yRange, zRange,
                        dstBuffer->getSize() - it->bufferOffset, reinterpret_cast<void*>(it->bufferOffset));
                }
                else
                {
                    gl->glTexture.pglGetTextureSubImage(src, it->imageSubresource.mipLevel, it->imageOffset.x, yStart, zStart, it->imageExtent.width, yRange, zRange,
                        glfmt, gltype, dstBuffer->getSize() - it->bufferOffset, reinterpret_cast<void*>(it->bufferOffset));
                }
            }
            else
            {
                ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_PIXEL_PACK_UNPACK, ctxid);

                // TODO this isnt totally valid right?
                const size_t bytesPerLayer = pitch * yRange * (compressed ? nbl::asset::getFormatChannelCount(format) : nbl::asset::getTexelOrBlockBytesize(format)); // all block compressed formats are decoded into 1 byte per channel format
                for (uint32_t z = 0u; z < zRange; ++z)
                {
                    GLuint fbo = ctxlocal->getSingleColorAttachmentFBO(gl, srcImage, it->imageSubresource.mipLevel, it->imageSubresource.baseArrayLayer + z);
                    if (!fbo)
                    {
                        // TODO log something
                        continue;
                    }

                    gl->glFramebuffer.pglBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);

                    size_t bufOffset = it->bufferOffset + z * bytesPerLayer;
                    bufOffset = core::alignUp(bufOffset, alignment); // ??? am i doing it right?
                    gl->glTexture.pglReadPixels(it->imageOffset.x, yStart, it->imageExtent.width, yRange, glfmt, gltype, reinterpret_cast<void*>(bufOffset));
                }
            }
        }

        if (!usingGetTexSubImage)
            gl->glFramebuffer.pglBindFramebuffer(GL_READ_FRAMEBUFFER, prevReadFB);
    }

    void COpenGLCommandBuffer::beginRenderpass_clearAttachments(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const SRenderpassBeginInfo& info, GLuint fbo, const system::logger_opt_ptr logger)
    {
        auto& rp = info.framebuffer->getCreationParameters().renderpass;
        auto& sub = rp->getSubpasses().begin()[0];
        auto* color = sub.colorAttachments;
        auto* depthstencil = sub.depthStencilAttachment;
        auto* descriptions = rp->getAttachments().begin();

        for (uint32_t i = 0u; i < sub.colorAttachmentCount; ++i)
        {
            const uint32_t a = color[i].attachment;

            if (descriptions[a].loadOp == asset::IRenderpass::ELO_CLEAR)
            {
                if (a < info.clearValueCount)
                {
                    const GLfloat* colorf = info.clearValues[a].color.float32;
                    const GLint* colori = info.clearValues[a].color.int32;
                    const GLuint* coloru = info.clearValues[a].color.uint32;

                    asset::E_FORMAT fmt = descriptions[a].format;

                    if (asset::isFloatingPointFormat(fmt) || asset::isNormalizedFormat(fmt))
                    {
                        gl->extGlClearNamedFramebufferfv(fbo, GL_COLOR, i, colorf);
                    }
                    else if (asset::isIntegerFormat(fmt))
                    {
                        if (asset::isSignedFormat(fmt))
                        {
                            gl->extGlClearNamedFramebufferiv(fbo, GL_COLOR, i, colori);
                        }
                        else
                        {
                            gl->extGlClearNamedFramebufferuiv(fbo, GL_COLOR, i, coloru);
                        }
                    }
                }
                else
                {
                    logger.log("Begin renderpass command: not enough clear values provided, an attachment not cleared!", system::ILogger::ELL_ERROR);
                }
            }
        }
        if (depthstencil)
        {
            auto* depthstencilDescription = descriptions + depthstencil->attachment;
            if (depthstencilDescription->loadOp == asset::IRenderpass::ELO_CLEAR)
            {
                if (depthstencil->attachment < info.clearValueCount)
                {
                    const auto& clear = info.clearValues[depthstencil->attachment].depthStencil;
                    asset::E_FORMAT fmt = depthstencilDescription->format;

                    // isnt there a way in vulkan to clear only depth or only stencil part?? TODO

                    const bool is_depth = asset::isDepthOnlyFormat(fmt);
                    const bool is_stencil = asset::isStencilOnlyFormat(fmt);
                    const bool is_depth_stencil = asset::isDepthOrStencilFormat(fmt);

                    const auto state_backup = ctxlocal->backupAndFlushStateClear(gl, ctxid, false, (is_depth | is_depth_stencil), (is_stencil | is_depth_stencil));

                    GLfloat depth = clear.depth;
                    GLint stencil = clear.stencil;
                    if (is_depth)
                    {
                        gl->extGlClearNamedFramebufferfv(fbo, GL_DEPTH, 0, &depth);
                    }
                    else if (is_stencil)
                    {
                        gl->extGlClearNamedFramebufferiv(fbo, GL_STENCIL, 0, &stencil);
                    }
                    else if (is_depth_stencil)
                    {
                        gl->extGlClearNamedFramebufferfi(fbo, GL_DEPTH_STENCIL, 0, depth, stencil);
                    }

                    ctxlocal->restoreStateAfterClear(state_backup);
                }
                else
                {
                    logger.log("Begin renderpass command: not enough clear values provided, an attachment not cleared!", system::ILogger::ELL_ERROR);
                }
            }
        }
    }

    void COpenGLCommandBuffer::clearAttachments(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t count, const asset::SClearAttachment* attachments)
    {
        auto& framebuffer = ctxlocal->currentState.framebuffer.fbo;
        const GLuint fbo = ctxlocal->currentState.framebuffer.GLname;
        if (!framebuffer || !fbo)
            return;
        auto& rp = framebuffer->getCreationParameters().renderpass;
        auto& sub = rp->getSubpasses().begin()[0];
        auto* color = sub.colorAttachments;
        auto* depthstencil = sub.depthStencilAttachment;
        auto* descriptions = rp->getAttachments().begin();

        for (uint32_t i = 0u; i < count; ++i)
        {
            auto& attachment = attachments[i];
            if (attachment.aspectMask & asset::IImage::EAF_COLOR_BIT)
            {
                uint32_t num = attachment.colorAttachment;
                uint32_t a = color[num].attachment;
                {
                    asset::E_FORMAT fmt = descriptions[a].format;

                    if (asset::isFloatingPointFormat(fmt) || asset::isNormalizedFormat(fmt))
                    {
                        gl->extGlClearNamedFramebufferfv(fbo, GL_COLOR, num, attachment.clearValue.color.float32);
                    }
                    else if (asset::isIntegerFormat(fmt))
                    {
                        if (asset::isSignedFormat(fmt))
                        {
                            gl->extGlClearNamedFramebufferiv(fbo, GL_COLOR, num, attachment.clearValue.color.int32);
                        }
                        else
                        {
                            gl->extGlClearNamedFramebufferuiv(fbo, GL_COLOR, num, attachment.clearValue.color.uint32);
                        }
                    }
                }
            }
            else if (attachment.aspectMask & (asset::IImage::EAF_DEPTH_BIT | asset::IImage::EAF_STENCIL_BIT))
            {
                auto aspectMask = (attachment.aspectMask & (asset::IImage::EAF_DEPTH_BIT | asset::IImage::EAF_STENCIL_BIT));
                if (aspectMask == (asset::IImage::EAF_DEPTH_BIT | asset::IImage::EAF_STENCIL_BIT))
                    gl->extGlClearNamedFramebufferfi(fbo, GL_DEPTH_STENCIL, 0, attachment.clearValue.depthStencil.depth, attachment.clearValue.depthStencil.stencil);
                else if (aspectMask == asset::IImage::EAF_DEPTH_BIT)
                    gl->extGlClearNamedFramebufferfv(fbo, GL_DEPTH, 0, &attachment.clearValue.depthStencil.depth);
                else if (aspectMask == asset::IImage::EAF_STENCIL_BIT)
                    gl->extGlClearNamedFramebufferiv(fbo, GL_STENCIL, 0, reinterpret_cast<const GLint*>(&attachment.clearValue.depthStencil.stencil));
            }
        }
    }

    bool COpenGLCommandBuffer::pushConstants_validate(const IGPUPipelineLayout* _layout, uint32_t _stages, uint32_t _offset, uint32_t _size, const void* _values)
    {
        if (!_layout || !_values)
            return false;
        if (!_size)
            return false;
        if (!_stages)
            return false;
        if (!core::is_aligned_to(_offset, 4u))
            return false;
        if (!core::is_aligned_to(_size, 4u))
            return false;
        if (_offset >= IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)
            return false;
        if ((_offset + _size) > IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)
            return false;

        asset::SPushConstantRange updateRange;
        updateRange.offset = _offset;
        updateRange.size = _size;

#ifdef _NBL_DEBUG
        //TODO validation:
        /*
        For each byte in the range specified by offset and size and for each shader stage in stageFlags,
        there must be a push constant range in layout that includes that byte and that stage
        */
        for (const auto& rng : _layout->getPushConstantRanges())
        {
            /*
            For each byte in the range specified by offset and size and for each push constant range that overlaps that byte,
            stageFlags must include all stages in that push constant ranges VkPushConstantRange::stageFlags
            */
            if (updateRange.overlap(rng) && ((_stages & rng.stageFlags) != rng.stageFlags))
                return false;
        }
#endif//_NBL_DEBUG

        return true;
    }

    void COpenGLCommandBuffer::executeAll(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid) const
    {
        for (const SCommand& cmd : m_commands)
        {
            switch (cmd.type)
            {
            case impl::ECT_BIND_INDEX_BUFFER:
            {
                auto& c = cmd.get<impl::ECT_BIND_INDEX_BUFFER>();
                auto* buffer = static_cast<const COpenGLBuffer*>(c.buffer.get());
                ctxlocal->nextState.vertexInputParams.vaoval.idxBinding = { c.offset, core::smart_refctd_ptr<const COpenGLBuffer>(buffer) };
                ctxlocal->nextState.vertexInputParams.vaoval.idxType = c.indexType;
            }
            break;
            case impl::ECT_DRAW:
            {
                auto& c = cmd.get<impl::ECT_DRAW>();

                ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);

                const asset::E_PRIMITIVE_TOPOLOGY primType = ctxlocal->currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
                GLenum glpt = getGLprimitiveType(primType);

                gl->extGlDrawArraysInstancedBaseInstance(glpt, c.firstVertex, c.vertexCount, c.instanceCount, c.firstInstance);
            }
            break;
            case impl::ECT_DRAW_INDEXED:
            {
                auto& c = cmd.get<impl::ECT_DRAW_INDEXED>();

                ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);

                const asset::E_PRIMITIVE_TOPOLOGY primType = ctxlocal->currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
                GLenum glpt = getGLprimitiveType(primType);
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

                    GLuint64 idxBufOffset = ctxlocal->currentState.vertexInputParams.vaoval.idxBinding.offset + ixsz*c.firstIndex;
                    static_assert(sizeof(idxBufOffset) == sizeof(void*), "Bad reinterpret_cast");
                    gl->extGlDrawElementsInstancedBaseVertexBaseInstance(glpt, c.indexCount, idxType, reinterpret_cast<void*>(idxBufOffset), c.instanceCount, c.vertexOffset, c.firstInstance);
                }
            }
            break;
            case impl::ECT_DRAW_INDIRECT:
            {
                auto& c = cmd.get<impl::ECT_DRAW_INDIRECT>();
                if (c.maxDrawCount==0u)
                    break;

                ctxlocal->nextState.vertexInputParams.indirectDrawBuf = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(c.buffer);
                ctxlocal->nextState.vertexInputParams.parameterBuf = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(c.countBuffer);

                ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);
                
                const asset::E_PRIMITIVE_TOPOLOGY primType = ctxlocal->currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
                GLenum glpt = getGLprimitiveType(primType);

                GLuint64 offset = c.offset;
                static_assert(sizeof(offset) == sizeof(void*), "Bad reinterpret_cast");
                if (ctxlocal->currentState.vertexInputParams.parameterBuf)
                    gl->extGlMultiDrawArraysIndirectCount(glpt, reinterpret_cast<void*>(offset), c.countBufferOffset, c.maxDrawCount, c.stride);
                else
                    gl->extGlMultiDrawArraysIndirect(glpt, reinterpret_cast<void*>(offset), c.maxDrawCount, c.stride);
            }
            break;
            case impl::ECT_DRAW_INDEXED_INDIRECT:
            {
                auto& c = cmd.get<impl::ECT_DRAW_INDEXED_INDIRECT>();
                if (c.maxDrawCount==0u)
                    break;

                ctxlocal->nextState.vertexInputParams.indirectDrawBuf = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(c.buffer);
                ctxlocal->nextState.vertexInputParams.parameterBuf = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(c.countBuffer);

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
                GLenum glpt = getGLprimitiveType(primType);

                GLuint64 offset = c.offset;
                static_assert(sizeof(offset) == sizeof(void*), "Bad reinterpret_cast");
                if (ctxlocal->currentState.vertexInputParams.parameterBuf)
                    gl->extGlMultiDrawElementsIndirectCount(glpt, idxType, reinterpret_cast<void*>(offset), c.countBufferOffset, c.maxDrawCount, c.stride);
                else
                    gl->extGlMultiDrawElementsIndirect(glpt, idxType, reinterpret_cast<void*>(offset), c.maxDrawCount, c.stride);
            }
            break;
            case impl::ECT_SET_VIEWPORT:
            {
                auto& c = cmd.get<impl::ECT_SET_VIEWPORT>();
                if (c.firstViewport < SOpenGLState::MAX_VIEWPORT_COUNT)
                {
                    uint32_t count = std::min(c.viewportCount, SOpenGLState::MAX_VIEWPORT_COUNT);
                    if (c.firstViewport + count > SOpenGLState::MAX_VIEWPORT_COUNT)
                        count = SOpenGLState::MAX_VIEWPORT_COUNT - c.firstViewport;

                    uint32_t first = c.firstViewport;
                    for (uint32_t i = 0u; i < count; ++i)
                    {
                        auto& vp = ctxlocal->nextState.rasterParams.viewport[first + i];
                        auto& vpd = ctxlocal->nextState.rasterParams.viewport_depth[first + i];

                        vp.x = c.viewports[i].x;
                        vp.y = c.viewports[i].y;
                        vp.width = c.viewports[i].width;
                        vp.height = c.viewports[i].height;
                        vpd.minDepth = c.viewports[i].minDepth;
                        vpd.maxDepth = c.viewports[i].maxDepth;
                    }
                }
            }
            break;
            case impl::ECT_SET_LINE_WIDTH:
            {
                auto& c = cmd.get<impl::ECT_SET_LINE_WIDTH>();
                ctxlocal->nextState.rasterParams.lineWidth = c.lineWidth;
            }
            break;
            case impl::ECT_SET_DEPTH_BIAS:
            {
                auto& c = cmd.get<impl::ECT_SET_DEPTH_BIAS>();
                // TODO what about c.depthBiasClamp
                ctxlocal->nextState.rasterParams.polygonOffset.factor = c.depthBiasSlopeFactor;
                ctxlocal->nextState.rasterParams.polygonOffset.units = c.depthBiasConstantFactor;
            }
            break;
            case impl::ECT_SET_BLEND_CONSTANTS:
            {
                auto& c = cmd.get<impl::ECT_SET_BLEND_CONSTANTS>();
                // TODO, cant see such thing in opengl
            }
            break;
            case impl::ECT_COPY_BUFFER:
            {
                auto& c = cmd.get<impl::ECT_COPY_BUFFER>();
                GLuint readb = static_cast<const COpenGLBuffer*>(c.srcBuffer.get())->getOpenGLName();
                GLuint writeb = static_cast<COpenGLBuffer*>(c.dstBuffer.get())->getOpenGLName();
                for (uint32_t i = 0u; i < c.regionCount; ++i)
                {
                    const asset::SBufferCopy& cp = c.regions[i];
                    gl->extGlCopyNamedBufferSubData(readb, writeb, cp.srcOffset, cp.dstOffset, cp.size);
                }
            }
            break;
            case impl::ECT_COPY_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_COPY_IMAGE>();
                IGPUImage* dstImage = c.dstImage.get();
                const IGPUImage* srcImage = c.srcImage.get();
                if (!dstImage->validateCopies(c.regions, c.regions + c.regionCount, srcImage))
                    return;

                auto src = static_cast<const COpenGLImage*>(srcImage);
                auto dst = static_cast<COpenGLImage*>(dstImage);
                IGPUImage::E_TYPE srcType = srcImage->getCreationParameters().type;
                IGPUImage::E_TYPE dstType = dstImage->getCreationParameters().type;
                constexpr GLenum type2Target[3u] = { GL_TEXTURE_1D_ARRAY,GL_TEXTURE_2D_ARRAY,GL_TEXTURE_3D };
                for (auto it = c.regions; it != c.regions + c.regionCount; it++)
                {
                    gl->extGlCopyImageSubData(src->getOpenGLName(), type2Target[srcType], it->srcSubresource.mipLevel,
                        it->srcOffset.x, srcType == IGPUImage::ET_1D ? it->srcSubresource.baseArrayLayer : it->srcOffset.y, srcType == IGPUImage::ET_2D ? it->srcSubresource.baseArrayLayer : it->srcOffset.z,
                        dst->getOpenGLName(), type2Target[dstType], it->dstSubresource.mipLevel,
                        it->dstOffset.x, dstType == IGPUImage::ET_1D ? it->dstSubresource.baseArrayLayer : it->dstOffset.y, dstType == IGPUImage::ET_2D ? it->dstSubresource.baseArrayLayer : it->dstOffset.z,
                        it->extent.width, dstType == IGPUImage::ET_1D ? it->dstSubresource.layerCount : it->extent.height, dstType == IGPUImage::ET_2D ? it->dstSubresource.layerCount : it->extent.depth);
                }
            }
            break;
            case impl::ECT_COPY_BUFFER_TO_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_COPY_BUFFER_TO_IMAGE>();

                copyBufferToImage(c, gl, ctxlocal, ctxid);
            }
            break;
            case impl::ECT_COPY_IMAGE_TO_BUFFER:
            {
                auto& c = cmd.get<impl::ECT_COPY_IMAGE_TO_BUFFER>();

                copyImageToBuffer(c, gl, ctxlocal, ctxid);
            }
            break;
            case impl::ECT_BLIT_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_BLIT_IMAGE>();

                for (uint32_t i = 0u; i < c.regionCount; ++i)
                {
                    auto& info = c.regions[i];

                    // already made sure in validation (see ICommandBuffer::blitImage), thats why only assert here
                    assert(info.dstSubresource.layerCount == info.srcSubresource.layerCount);

                    for (uint32_t l = 0u; l < info.dstSubresource.layerCount; ++l)
                    {
                        GLuint srcfbo = ctxlocal->getSingleColorAttachmentFBO(gl, c.srcImage.get(), info.srcSubresource.mipLevel, info.srcSubresource.baseArrayLayer + l);
                        GLuint dstfbo = ctxlocal->getSingleColorAttachmentFBO(gl, c.dstImage.get(), info.dstSubresource.mipLevel, info.dstSubresource.baseArrayLayer + l);

                        blit(gl, srcfbo, dstfbo, info.srcOffsets, info.dstOffsets, c.filter);
                    }
                }
            }
            break;
            case impl::ECT_RESOLVE_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_RESOLVE_IMAGE>();

                for (uint32_t i = 0u; i < c.regionCount; ++i)
                {
                    auto& info = c.regions[i];

                    assert(info.dstSubresource.layerCount == info.srcSubresource.layerCount);

                    asset::VkOffset3D srcoffsets[2]{ info.srcOffset,info.srcOffset };
                    srcoffsets[1].x += info.extent.width;
                    srcoffsets[1].y += info.extent.height;
                    srcoffsets[1].z += info.extent.depth;
                    asset::VkOffset3D dstoffsets[2]{ info.dstOffset,info.dstOffset };
                    dstoffsets[1].x += info.extent.width;
                    dstoffsets[1].y += info.extent.height;
                    dstoffsets[1].z += info.extent.depth;

                    for (uint32_t l = 0u; l < info.dstSubresource.layerCount; ++l)
                    {
                        GLuint srcfbo = ctxlocal->getSingleColorAttachmentFBO(gl, c.srcImage.get(), info.srcSubresource.mipLevel, info.srcSubresource.baseArrayLayer + l);
                        GLuint dstfbo = ctxlocal->getSingleColorAttachmentFBO(gl, c.dstImage.get(), info.dstSubresource.mipLevel, info.dstSubresource.baseArrayLayer + l);

                        blit(gl, srcfbo, dstfbo, srcoffsets, dstoffsets, asset::ISampler::ETF_NEAREST);
                    }
                }
            }
            break;
            case impl::ECT_BIND_VERTEX_BUFFERS:
            {
                auto& c = cmd.get<impl::ECT_BIND_VERTEX_BUFFERS>();

                for (uint32_t i = 0u; i < c.count; ++i)
                {
                    auto& binding = ctxlocal->nextState.vertexInputParams.vaoval.vtxBindings[c.first + i];
                    binding.buffer = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(c.buffers[i]);
                    binding.offset = c.offsets[i];
                }
            }
            break;
            case impl::ECT_SET_SCISSORS:
            {
                auto& c = cmd.get<impl::ECT_SET_SCISSORS>();
                // TODO ?
            }
            break;
            case impl::ECT_SET_DEPTH_BOUNDS:
            {
                auto& c = cmd.get<impl::ECT_SET_DEPTH_BOUNDS>();
                // TODO ?
            }
            break;
            case impl::ECT_SET_STENCIL_COMPARE_MASK:
            {
                auto& c = cmd.get<impl::ECT_SET_STENCIL_COMPARE_MASK>();
                if (c.faceMask & asset::ESFF_FRONT_BIT)
                    ctxlocal->nextState.rasterParams.stencilFunc_front.mask = c.cmpMask;
                if (c.faceMask & asset::ESFF_BACK_BIT)
                    ctxlocal->nextState.rasterParams.stencilFunc_back.mask = c.cmpMask;;
            }
            break;
            case impl::ECT_SET_STENCIL_WRITE_MASK:
            {
                auto& c = cmd.get<impl::ECT_SET_STENCIL_WRITE_MASK>();
                if (c.faceMask & asset::ESFF_FRONT_BIT)
                    ctxlocal->nextState.rasterParams.stencilWriteMask_front = c.writeMask;
                if (c.faceMask & asset::ESFF_BACK_BIT)
                    ctxlocal->nextState.rasterParams.stencilWriteMask_back = c.writeMask;
            }
            break;
            case impl::ECT_SET_STENCIL_REFERENCE:
            {
                auto& c = cmd.get<impl::ECT_SET_STENCIL_REFERENCE>();
                if (c.faceMask & asset::ESFF_FRONT_BIT)
                    ctxlocal->nextState.rasterParams.stencilFunc_front.ref = c.reference;
                if (c.faceMask & asset::ESFF_BACK_BIT)
                    ctxlocal->nextState.rasterParams.stencilFunc_back.ref = c.reference;
            }
            break;
            case impl::ECT_DISPATCH:
            {
                auto& c = cmd.get<impl::ECT_DISPATCH>();
                
                ctxlocal->flushStateCompute(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);

                gl->glCompute.pglDispatchCompute(c.groupCountX, c.groupCountY, c.groupCountZ);
            }
            break;
            case impl::ECT_DISPATCH_INDIRECT:
            {
                auto& c = cmd.get<impl::ECT_DISPATCH_INDIRECT>();
                ctxlocal->nextState.dispatchIndirect.buffer = core::smart_refctd_ptr_static_cast<const COpenGLBuffer>(c.buffer);
                
                ctxlocal->flushStateCompute(gl, SOpenGLContextLocalCache::GSB_ALL, ctxid);

                gl->glCompute.pglDispatchComputeIndirect(static_cast<GLintptr>(c.offset));
            }
            break;
            case impl::ECT_DISPATCH_BASE:
            {
                auto& c = cmd.get<impl::ECT_DISPATCH_BASE>();
                // no such thing in opengl (easy to emulate tho)
                // maybe spirv-cross emits some uniforms for this?
            }
            break;
            case impl::ECT_SET_EVENT:
            {
                auto& c = cmd.get<impl::ECT_SET_EVENT>();
                //https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdSetEvent2KHR.html
                // A memory dependency is defined between the event signal operation and commands that occur earlier in submission order.
                //gl->glSync.pglMemoryBarrier(c.barrierBits); @Crisspl?
            }
            break;
            case impl::ECT_RESET_EVENT:
            {
                auto& c = cmd.get<impl::ECT_RESET_EVENT>();
                // currently no-op
            }
            break;
            case impl::ECT_WAIT_EVENTS:
            {
                auto& c = cmd.get<impl::ECT_WAIT_EVENTS>();
                gl->glSync.pglMemoryBarrier(c.barrier);
            }
            break;
            case impl::ECT_PIPELINE_BARRIER:
            {
                auto& c = cmd.get<impl::ECT_PIPELINE_BARRIER>();
                gl->glSync.pglMemoryBarrier(c.barrier);
            }
            break;
            case impl::ECT_BEGIN_RENDERPASS:
            {
                auto& c = cmd.get<impl::ECT_BEGIN_RENDERPASS>();
                auto framebuf = core::smart_refctd_ptr_static_cast<const COpenGLFramebuffer>(c.renderpassBegin.framebuffer);

                ctxlocal->nextState.framebuffer.hash = framebuf->getHashValue();
                ctxlocal->nextState.framebuffer.fbo = std::move(framebuf);
                ctxlocal->flushStateGraphics(gl, SOpenGLContextLocalCache::GSB_FRAMEBUFFER, ctxid);

                GLuint fbo = ctxlocal->currentState.framebuffer.GLname;
                if (fbo)
                    beginRenderpass_clearAttachments(gl, ctxlocal, ctxid, c.renderpassBegin, fbo, m_logger.getOptRawPtr());
            }
            break;
            case impl::ECT_NEXT_SUBPASS:
            {
                auto& c = cmd.get<impl::ECT_NEXT_SUBPASS>();
                // TODO (when we support subpasses) some barriers based on subpass dependencies?
                // not needed now tho, we dont support multiple subpasses yet
            }
            break;
            case impl::ECT_END_RENDERPASS:
            {
                auto& c = cmd.get<impl::ECT_END_RENDERPASS>();
                ctxlocal->nextState.framebuffer.hash = SOpenGLState::NULL_FBO_HASH;
                ctxlocal->nextState.framebuffer.GLname = 0u;
                ctxlocal->nextState.framebuffer.fbo = nullptr;
            }
            break;
            case impl::ECT_SET_DEVICE_MASK:
            {
                auto& c = cmd.get<impl::ECT_SET_DEVICE_MASK>();
                // no-op
            }
            break;
            case impl::ECT_BIND_GRAPHICS_PIPELINE:
            {
                auto& c = cmd.get<impl::ECT_BIND_GRAPHICS_PIPELINE>();

                ctxlocal->updateNextState_pipelineAndRaster(c.pipeline.get(), ctxid);

                auto* rpindependent = c.pipeline->getRenderpassIndependentPipeline();
                auto* glppln = static_cast<const COpenGLRenderpassIndependentPipeline*>(rpindependent);
                ctxlocal->nextState.vertexInputParams.vaokey = glppln->getVAOHash();
            }
            break;
            case impl::ECT_BIND_COMPUTE_PIPELINE:
            {
                auto& c = cmd.get<impl::ECT_BIND_COMPUTE_PIPELINE>();

                const COpenGLComputePipeline* glppln = static_cast<const COpenGLComputePipeline*>(c.pipeline.get());
                ctxlocal->nextState.pipeline.compute.usedShader = glppln ? glppln->getShaderGLnameForCtx(0u, ctxid) : 0u;
                ctxlocal->nextState.pipeline.compute.pipeline = core::smart_refctd_ptr<const COpenGLComputePipeline>(glppln);
            }
            break;
            case impl::ECT_RESET_QUERY_POOL:
            {
                auto& c = cmd.get<impl::ECT_RESET_QUERY_POOL>();
                _NBL_TODO();
            }
            break;
            case impl::ECT_BEGIN_QUERY:
            {
                auto& c = cmd.get<impl::ECT_BEGIN_QUERY>();
                const COpenGLQueryPool* qp = static_cast<const COpenGLQueryPool*>(c.queryPool.get());
                qp->beginQuery(gl, c.query, c.flags);
            }
            break;
            case impl::ECT_END_QUERY:
            {
                auto& c = cmd.get<impl::ECT_END_QUERY>();
                const COpenGLQueryPool* qp = static_cast<const COpenGLQueryPool*>(c.queryPool.get());
                qp->endQuery(gl, c.query);
            }
            break;
            case impl::ECT_COPY_QUERY_POOL_RESULTS:
            {
                auto& c = cmd.get<impl::ECT_COPY_QUERY_POOL_RESULTS>();
                
                const COpenGLBuffer* buffer = static_cast<const COpenGLBuffer*>(c.dstBuffer.get());
                GLuint bufferId = buffer->getOpenGLName();

                const COpenGLQueryPool* qp = static_cast<const COpenGLQueryPool*>(c.queryPool.get());
                
                auto queryPoolQueriesCount = qp->getCreationParameters().queryCount;
                auto queriesRange = qp->getQueries(); // queriesRange.size() is a multiple of queryPoolQueriesCount
                auto queries = queriesRange.begin();
                
                IQueryPool::E_QUERY_TYPE queryType = qp->getCreationParameters().queryType;
                bool use64Version = (c.flags & IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_64_BIT) == IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_64_BIT;
                bool availabilityFlag = (c.flags & IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_WITH_AVAILABILITY_BIT) == IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_WITH_AVAILABILITY_BIT;
                bool waitForAllResults = (c.flags & IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_WAIT_BIT) == IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_WAIT_BIT;
                bool partialResults = (c.flags & IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_PARTIAL_BIT) == IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_PARTIAL_BIT;

                if(c.firstQuery + c.queryCount > queryPoolQueriesCount)
                {
                    assert(false && "The sum of firstQuery and queryCount must be less than or equal to the number of queries in queryPool");
                    break;
                }
                if(partialResults && queryType == IQueryPool::E_QUERY_TYPE::EQT_TIMESTAMP) {
                    assert(false && "QUERY_RESULT_PARTIAL_BIT must not be used if the pool’s queryType is QUERY_TYPE_TIMESTAMP.");
                    break;
                }

                size_t currentDataPtrOffset = c.dstOffset;
                size_t queryElementDataSize = (use64Version) ? sizeof(GLuint64) : sizeof(GLuint); // each query might write to multiple values/elements

                GLenum pname;
                if(availabilityFlag)
                    pname = GL_QUERY_RESULT_AVAILABLE;
                else if(waitForAllResults)
                    pname = GL_QUERY_RESULT;
                else if(partialResults)
                    pname = GL_QUERY_NO_WAIT;

                auto getQueryBufferObject = [&](GLuint queryId, GLuint buffer, GLenum pname, GLintptr offset) -> void 
                {
                    if(use64Version)
                    {
                        gl->extGlGetQueryBufferObjectui64v(queryId, buffer, pname, offset);
                    }
                    else
                    {
                        gl->extGlGetQueryBufferObjectuiv(queryId, buffer, pname, offset);
                    }
                };

                for(uint32_t i = 0; i < c.queryCount; ++i)
                {
                    if(queryType == IQueryPool::E_QUERY_TYPE::EQT_TIMESTAMP || queryType == IQueryPool::E_QUERY_TYPE::EQT_OCCLUSION)
                    {
                        assert(queryPoolQueriesCount == queriesRange.size());
                        assert(c.stride >= queryElementDataSize);
                        GLuint query = queries[i+c.firstQuery];

                        getQueryBufferObject(query, bufferId, pname, currentDataPtrOffset);
                    }
                    else if(queryType == IQueryPool::E_QUERY_TYPE::EQT_TRANSFORM_FEEDBACK_STREAM_EXT)
                    {
                        assert(queryPoolQueriesCount * 2 == queriesRange.size());
                        assert(c.stride >= queryElementDataSize * 2);

                        GLuint query1 = queries[i+c.firstQuery];
                        GLuint query2 = queries[i+c.firstQuery + queryPoolQueriesCount];

                        // Write All
                        getQueryBufferObject(query1, bufferId, pname, currentDataPtrOffset);
                        getQueryBufferObject(query2, bufferId, pname, currentDataPtrOffset + queryElementDataSize);
                    }
                    else
                    {
                        assert(false && "QueryType is not supported.");
                    }

                    currentDataPtrOffset += c.stride;
                }
            }
            break;
            case impl::ECT_WRITE_TIMESTAMP:
            {
                auto& c = cmd.get<impl::ECT_WRITE_TIMESTAMP>();
                const COpenGLQueryPool* qp = static_cast<const COpenGLQueryPool*>(c.queryPool.get());
                GLuint query = qp->getQueryAt(c.query);
                assert(qp->getCreationParameters().queryType == IQueryPool::E_QUERY_TYPE::EQT_TIMESTAMP);
                gl->glQuery.pglQueryCounter(query, GL_TIMESTAMP);
            }
            break;
            case impl::ECT_BEGIN_QUERY_INDEXED:
            {
                auto& c = cmd.get<impl::ECT_BEGIN_QUERY_INDEXED>();
                const COpenGLQueryPool* qp = static_cast<const COpenGLQueryPool*>(c.queryPool.get());
                qp->beginQueryIndexed(gl, c.query, c.index, c.flags);
            }
            break;
            case impl::ECT_END_QUERY_INDEXED:
            {
                auto& c = cmd.get<impl::ECT_END_QUERY_INDEXED>();
                const COpenGLQueryPool* qp = static_cast<const COpenGLQueryPool*>(c.queryPool.get());
                qp->endQueryIndexed(gl, c.query, c.index);
            }
            break;
            case impl::ECT_BIND_DESCRIPTOR_SETS:
            {
                auto& c = cmd.get<impl::ECT_BIND_DESCRIPTOR_SETS>();

                asset::E_PIPELINE_BIND_POINT pbp = c.pipelineBindPoint;

                const IGPUPipelineLayout* layouts[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
                for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
                    layouts[i] = ctxlocal->nextState.descriptorsParams[pbp].descSets[i].pplnLayout.get();
                const IGPUDescriptorSet* descriptorSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
                for (uint32_t i = 0u; i < c.dsCount; ++i)
                    descriptorSets[i] = c.descriptorSets[i].get();
                bindDescriptorSets_generic(c.layout.get(), c.firstSet, c.dsCount, descriptorSets, layouts);

                for (uint32_t i=0u; i<IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
                    if (!layouts[i])
                        ctxlocal->nextState.descriptorsParams[pbp].descSets[i] = { nullptr, nullptr, {}, 0u }; // TODO: have a default constructor that makes sense and prevents us from screwing up

                uint32_t offsetOfDynamicOffsets = 0u;
                for (uint32_t i = 0u; i < c.dsCount; i++)
                {
                    auto glDS = static_cast<const COpenGLDescriptorSet*>(descriptorSets[i]);
                    if(glDS)
                    {
                        const auto dynamicOffsetCount = glDS->getDynamicOffsetCount();

                        auto& stateDS = ctxlocal->nextState.descriptorsParams[pbp].descSets[c.firstSet+i];
                        stateDS.pplnLayout = core::smart_refctd_ptr<const COpenGLPipelineLayout>(static_cast<const COpenGLPipelineLayout*>(c.layout.get()));
                        stateDS.set = core::smart_refctd_ptr<const COpenGLDescriptorSet>(glDS);
                        std::copy_n(c.dynamicOffsets+offsetOfDynamicOffsets,dynamicOffsetCount,stateDS.dynamicOffsets);
                        stateDS.revision = glDS->getRevision();

                        offsetOfDynamicOffsets += dynamicOffsetCount;
                    }
                }
                assert(offsetOfDynamicOffsets==c.dynamicOffsetCount);
            }
            break;
            case impl::ECT_PUSH_CONSTANTS:
            {
                auto& c = cmd.get<impl::ECT_PUSH_CONSTANTS>();

                if (pushConstants_validate(c.layout.get(), c.stageFlags.value, c.offset, c.size, c.values))
                {
                    asset::SPushConstantRange updtRng;
                    updtRng.offset = c.offset;
                    updtRng.size = c.size;

                    if (c.stageFlags.value & asset::IShader::ESS_ALL_GRAPHICS)
                        ctxlocal->pushConstants<asset::EPBP_GRAPHICS>(static_cast<const COpenGLPipelineLayout*>(c.layout.get()), c.stageFlags.value, c.offset, c.size, c.values);
                    if (c.stageFlags.value & asset::IShader::ESS_COMPUTE)
                        ctxlocal->pushConstants<asset::EPBP_COMPUTE>(static_cast<const COpenGLPipelineLayout*>(c.layout.get()), c.stageFlags.value, c.offset, c.size, c.values);
                }
            }
            break;
            case impl::ECT_CLEAR_COLOR_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_CLEAR_COLOR_IMAGE>();

                const auto format = c.image->getCreationParameters().format;

                const bool is_fp = asset::isNormalizedFormat(format) || asset::isFloatingPointFormat(format);
                bool is_int = false;
                bool is_sint = false;
                if (!is_fp && asset::isIntegerFormat(format))
                {
                    is_int = true;
                    is_sint = asset::isSignedFormat(format);
                }

                const auto state_backup = ctxlocal->backupAndFlushStateClear(gl, ctxid, true, false, false);

                for (uint32_t i = 0u; i < c.rangeCount; ++i)
                {
                    auto& info = c.ranges[i];

                    for (uint32_t m = 0u; m < info.levelCount; ++m)
                    {
                        for (uint32_t l = 0u; l < info.layerCount; ++l)
                        {
                            GLuint fbo = ctxlocal->getSingleColorAttachmentFBO(gl, c.image.get(), info.baseMipLevel + m, info.baseArrayLayer + l);

                            if (is_fp)
                            {
                                gl->extGlClearNamedFramebufferfv(fbo, GL_COLOR, 0, c.color.float32);
                            }
                            else if (is_int)
                            {
                                if (is_sint)
                                {
                                    gl->extGlClearNamedFramebufferiv(fbo, GL_COLOR, 0, c.color.int32);
                                }
                                else
                                {
                                    gl->extGlClearNamedFramebufferuiv(fbo, GL_COLOR, 0, c.color.uint32);
                                }
                            }
                        }
                    }
                }

                ctxlocal->restoreStateAfterClear(state_backup);
            }
            break;
            case impl::ECT_CLEAR_DEPTH_STENCIL_IMAGE:
            {
                auto& c = cmd.get<impl::ECT_CLEAR_DEPTH_STENCIL_IMAGE>();
                const auto fmt = c.image->getCreationParameters().format;

                const bool is_depth = asset::isDepthOnlyFormat(fmt);
                bool is_stencil = false;
                bool is_depth_stencil = false;
                if (!is_depth)
                {
                    is_stencil = asset::isStencilOnlyFormat(fmt);
                    if (!is_stencil)
                        is_depth_stencil = asset::isDepthOrStencilFormat(fmt);
                }

                const auto state_backup = ctxlocal->backupAndFlushStateClear(gl, ctxid, false, (is_depth | is_depth_stencil), (is_stencil | is_depth_stencil));

                for (uint32_t i = 0u; i < c.rangeCount; ++i)
                {
                    const auto& info = c.ranges[i];

                    for (uint32_t m = 0u; m < info.levelCount; ++m)
                    {
                        for (uint32_t l = 0u; l < info.layerCount; ++l)
                        {
                            const GLuint fbo = ctxlocal->getDepthStencilAttachmentFBO(gl, c.image.get(), info.baseMipLevel + m, info.baseArrayLayer + l);

                            if (is_depth)
                            {
                                gl->extGlClearNamedFramebufferfv(fbo, GL_DEPTH, 0, &c.depthStencil.depth);
                            }
                            else if (is_stencil)
                            {
                                static_assert(sizeof(GLint) == sizeof(c.depthStencil.stencil), "Bad reinterpret_cast!");
                                gl->extGlClearNamedFramebufferiv(fbo, GL_STENCIL, 0, reinterpret_cast<const GLint*>(&c.depthStencil.stencil));
                            }
                            else if (is_depth_stencil)
                            {
                                gl->extGlClearNamedFramebufferfi(fbo, GL_DEPTH_STENCIL, 0, c.depthStencil.depth, c.depthStencil.stencil);
                            }
                        }
                    }
                }

                ctxlocal->restoreStateAfterClear(state_backup);
            }
            break;
            case impl::ECT_CLEAR_ATTACHMENTS:
            {
                auto& c = cmd.get<impl::ECT_CLEAR_ATTACHMENTS>();
                clearAttachments(gl, ctxlocal, c.attachmentCount, c.attachments);
            }
            break;
            case impl::ECT_FILL_BUFFER:
            {
                auto& c = cmd.get<impl::ECT_FILL_BUFFER>();

                GLuint buf = static_cast<const COpenGLBuffer*>(c.dstBuffer.get())->getOpenGLName();
                gl->extGlClearNamedBufferSubData(buf, GL_R32UI, c.dstOffset, c.size, GL_RED, GL_UNSIGNED_INT, &c.data);
            }
            break;
            case impl::ECT_UPDATE_BUFFER:
            {
                auto& c = cmd.get<impl::ECT_UPDATE_BUFFER>();

                GLuint buf = static_cast<const COpenGLBuffer*>(c.dstBuffer.get())->getOpenGLName();
                const float* ptr = reinterpret_cast<const float*>(c.data);
                gl->extGlNamedBufferSubData(buf, c.dstOffset, c.dataSize, c.data);
            }
            break;
            case impl::ECT_EXECUTE_COMMANDS:
            {
                auto& c = cmd.get<impl::ECT_EXECUTE_COMMANDS>();

                static_cast<COpenGLCommandBuffer*>(c.cmdbuf.get())->executeAll(gl, ctxlocal, ctxid);
            }
            break;
            case impl::ECT_REGENERATE_MIPMAPS:
            {
                auto& c = cmd.get<impl::ECT_REGENERATE_MIPMAPS>();
                auto* glimg = static_cast<COpenGLImage*>(c.imgview.get());

                gl->extGlGenerateTextureMipmap(glimg->getOpenGLName(), glimg->getOpenGLTarget());
            }
            break;
            }
        }
    }

    void COpenGLCommandBuffer::blit(IOpenGL_FunctionTable* gl, GLuint src, GLuint dst, const asset::VkOffset3D srcOffsets[2], const asset::VkOffset3D dstOffsets[2], asset::ISampler::E_TEXTURE_FILTER filter)
    {
        GLint sx0 = srcOffsets[0].x;
        GLint sy0 = srcOffsets[0].y;
        GLint sx1 = srcOffsets[1].x;
        GLint sy1 = srcOffsets[1].y;
        GLint dx0 = dstOffsets[0].x;
        GLint dy0 = dstOffsets[0].y;
        GLint dx1 = dstOffsets[1].x;
        GLint dy1 = dstOffsets[1].y;
        gl->extGlBlitNamedFramebuffer(src, dst, sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, GL_COLOR_BUFFER_BIT, filter==asset::ISampler::ETF_NEAREST?GL_NEAREST:GL_LINEAR);
    }
}