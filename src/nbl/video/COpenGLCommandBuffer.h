#ifndef __NBL_C_OPENGL_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_OPENGL_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <variant>

#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/IGPUMeshBuffer.h"

#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/SOpenGLContextLocalCache.h"
#include "nbl/video/IQueryPool.h"
#include "nbl/video/COpenGLCommandPool.h"

namespace nbl::video
{

class COpenGLCommandBuffer final : public IGPUCommandBuffer
{
protected:
    ~COpenGLCommandBuffer()
    {
        // We don't do anything here because the deletion of all command segment lists will happen in ~IGPUCommandBuffer through releaseResourcesBackToPool_impl.
    }

    static bool pushConstants_validate(const IGPUPipelineLayout* _layout, uint32_t _stages, uint32_t _offset, uint32_t _size, const void* _values);

    static void blit(IOpenGL_FunctionTable* gl, GLuint src, GLuint dst, const asset::VkOffset3D srcOffsets[2], const asset::VkOffset3D dstOffsets[2], asset::ISampler::E_TEXTURE_FILTER filter);

    static inline GLbitfield barriersToMemBarrierBits(
        const SOpenGLBarrierHelper& helper,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers)
    {
        const GLbitfield bufferBits = helper.AllBarrierBits^SOpenGLBarrierHelper::ImageTransferBits;
        constexpr GLbitfield imageBits = SOpenGLBarrierHelper::ImageDescriptorAccessBits|SOpenGLBarrierHelper::ImageTransferBits;

        // ignoring source access flags
        GLbitfield bufaccess = 0u;
        for (uint32_t i = 0u; i < bufferMemoryBarrierCount; ++i)
            bufaccess |= helper.accessFlagsToMemoryBarrierBits(pBufferMemoryBarriers[i].barrier);
        GLbitfield imgaccess = 0u;
        for (uint32_t i = 0u; i < imageMemoryBarrierCount; ++i)
            imgaccess |= helper.accessFlagsToMemoryBarrierBits(pImageMemoryBarriers[i].barrier);
        GLbitfield membarrier = 0u;

        for (uint32_t i = 0u; i < memoryBarrierCount; ++i)
            membarrier |= helper.accessFlagsToMemoryBarrierBits(pMemoryBarriers[i]);

        GLbitfield bufbarrier = bufferBits & bufaccess;
        GLbitfield imgbarrier = imageBits & imgaccess;

        return bufbarrier | imgbarrier | membarrier;
    }

    COpenGLCommandPool* getGLCommandPool() const { return static_cast<COpenGLCommandPool*>(m_cmdpool.get()); }

    const COpenGLFeatureMap* m_features;
    mutable core::bitflag<IQueryPool::E_QUERY_TYPE> queriesActive;
    mutable std::tuple<IQueryPool const *,uint32_t/*query ix*/,renderpass_t const *,uint32_t/*subpass ix*/> currentlyRecordingQueries[IQueryPool::EQT_COUNT];

public:
    static bool beginRenderpass_clearAttachments(SOpenGLContextIndependentCache* stateCache, const SRenderpassBeginInfo& info, const system::logger_opt_ptr logger, IGPUCommandPool* cmdpool, IGPUCommandPool::CCommandSegment*& segmentListHead, IGPUCommandPool::CCommandSegment*& segmentListTail, const E_API_TYPE apiType, const COpenGLFeatureMap* features);

    static inline GLenum getGLprimitiveType(asset::E_PRIMITIVE_TOPOLOGY pt)
    {
        using namespace asset;
        switch (pt)
        {
        case EPT_POINT_LIST:
            return GL_POINTS;
        case EPT_LINE_LIST:
            return GL_LINES;
        case EPT_LINE_STRIP:
            return GL_LINE_STRIP;
        case EPT_TRIANGLE_LIST:
            return GL_TRIANGLES;
        case EPT_TRIANGLE_STRIP:
            return GL_TRIANGLE_STRIP;
        case EPT_TRIANGLE_FAN:
            return GL_TRIANGLE_FAN;
        case EPT_LINE_LIST_WITH_ADJACENCY:
            return GL_LINES_ADJACENCY;
        case EPT_LINE_STRIP_WITH_ADJACENCY:
            return GL_LINE_STRIP_ADJACENCY;
        case EPT_TRIANGLE_LIST_WITH_ADJACENCY:
            return GL_TRIANGLES_ADJACENCY;
        case EPT_TRIANGLE_STRIP_WITH_ADJACENCY:
            return GL_TRIANGLE_STRIP_ADJACENCY;
        case EPT_PATCH_LIST:
            return GL_PATCHES;
        default:
            return GL_INVALID_ENUM;
        }
    }

    mutable renderpass_t const * currentlyRecordingRenderPass = nullptr;

    void executeAll(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocal, uint32_t ctxid) const;

    COpenGLCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, E_LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool, system::logger_opt_smart_ptr&& logger, const COpenGLFeatureMap* _features);

    bool begin_impl(core::bitflag<E_USAGE> flags, const SInheritanceInfo* inheritanceInfo) override final;

    bool end_impl() override final
    {
        assert(queriesActive.value == 0u); // No Queries should be active when command buffer ends
        return true;
    }

    void releaseResourcesBackToPool_impl() override final;

    inline void checkForParentPoolReset_impl() const override { assert(!"Deprecated"); }

    inline void bindIndexBuffer_impl(const buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType) override
    {
        auto* glbuffer = static_cast<const COpenGLBuffer*>(buffer);
        m_stateCache.nextState.vertexInputParams.vaoval.idxBinding = { offset, core::smart_refctd_ptr<const COpenGLBuffer>(glbuffer) };
        m_stateCache.nextState.vertexInputParams.vaoval.idxType = indexType;
    }

    bool draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) override
    {
        if (!m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHead, m_GLSegmentListTail, getAPIType(), m_features))
            return false;

        const asset::E_PRIMITIVE_TOPOLOGY primType = m_stateCache.currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
        GLenum glpt = getGLprimitiveType(primType);

        if (!m_cmdpool->emplace<COpenGLCommandPool::CDrawArraysInstancedBaseInstanceCmd>(m_GLSegmentListHead, m_GLSegmentListTail, glpt, firstVertex, vertexCount, instanceCount, firstInstance))
            return false;

        return true;
    }

    bool drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) override
    {
        if (!m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHead, m_GLSegmentListTail, getAPIType(), m_features))
            return false;

        const asset::E_PRIMITIVE_TOPOLOGY primType = m_stateCache.currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
        GLenum glpt = getGLprimitiveType(primType);
        GLenum idxType = GL_INVALID_ENUM;
        switch (m_stateCache.currentState.vertexInputParams.vaoval.idxType)
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

            GLuint64 idxBufOffset = m_stateCache.currentState.vertexInputParams.vaoval.idxBinding.offset + ixsz * firstIndex;
            static_assert(sizeof(idxBufOffset) == sizeof(void*), "Bad reinterpret_cast");
            if (!m_cmdpool->emplace<COpenGLCommandPool::CDrawElementsInstancedBaseVertexBaseInstanceCmd>(m_GLSegmentListHead, m_GLSegmentListTail, glpt, indexCount, idxType, idxBufOffset, instanceCount, vertexOffset, firstInstance))
                return false;
        }

        return true;
    }

    bool drawIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override;
    bool drawIndexedIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override;
    bool drawIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override;
    bool drawIndexedIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override;

    bool setViewport(uint32_t firstViewport, uint32_t viewportCount, const asset::SViewport* pViewports) override;
    bool setLineWidth(float lineWidth) override;
    bool setDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor) override;
    bool setBlendConstants(const float blendConstants[4]) override;

    bool copyBuffer_impl(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions) override;
    bool copyImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) override;
    bool copyBufferToImage_impl(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override;
    bool copyImageToBuffer_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override;
    bool blitImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) override;
    bool resolveImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions) override;

    void bindVertexBuffers_impl(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets) override;

    bool setScissor(uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors) override;
    bool setDepthBounds(float minDepthBounds, float maxDepthBounds) override;
    bool setStencilCompareMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t compareMask) override;
    bool setStencilWriteMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t writeMask) override;
    bool setStencilReference(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t reference) override;

    bool dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override;
    bool dispatchIndirect_impl(const buffer_t* buffer, size_t offset) override;
    bool dispatchBase(uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override;

    bool setEvent_impl(event_t* _event, const SDependencyInfo& depInfo) override;
    bool resetEvent_impl(event_t* _event, asset::E_PIPELINE_STAGE_FLAGS stageMask) override;
    bool waitEvents_impl(uint32_t eventCount, event_t* const* const pEvents, const SDependencyInfo* depInfos);

    bool pipelineBarrier_impl(core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers) override
    {
        const SOpenGLBarrierHelper helper(m_features);
        GLbitfield barrier = helper.pipelineStageFlagsToMemoryBarrierBits(srcStageMask.value, dstStageMask.value);
        barrier &= barriersToMemBarrierBits(helper,memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);

        if (!m_cmdpool->emplace<COpenGLCommandPool::CMemoryBarrierCmd>(m_GLSegmentListHead, m_GLSegmentListTail, barrier))
            return false;

        return true;
    }

    bool beginRenderPass_impl(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) override
    {
        const auto* glfb = static_cast<const COpenGLFramebuffer*>(pRenderPassBegin->framebuffer.get());
        m_stateCache.nextState.framebuffer.hash = glfb->getHashValue();
        m_stateCache.nextState.framebuffer.fbo = glfb;
        if (!m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_FRAMEBUFFER, m_cmdpool.get(), m_GLSegmentListHead, m_GLSegmentListTail, getAPIType(), m_features))
            return false;

        if (!beginRenderpass_clearAttachments(&m_stateCache, *pRenderPassBegin, m_logger.getOptRawPtr(), m_cmdpool.get(), m_GLSegmentListHead, m_GLSegmentListTail, getAPIType(), m_features))
            return false;

        currentlyRecordingRenderPass = pRenderPassBegin->renderpass.get();

        return true;
    }

    inline bool nextSubpass(asset::E_SUBPASS_CONTENTS contents) override
    {
        // TODO (when we support subpasses) some barriers based on subpass dependencies?
        // not needed now tho, we dont support multiple subpasses yet

        return true;
    }

    inline bool endRenderPass() override
    {
        m_stateCache.nextState.framebuffer.hash = SOpenGLState::NULL_FBO_HASH;
        m_stateCache.nextState.framebuffer.fbo = nullptr;

        currentlyRecordingRenderPass = nullptr;

        return true;
    }

    bool bindGraphicsPipeline_impl(const graphics_pipeline_t* pipeline) override;
    void bindComputePipeline_impl(const compute_pipeline_t* pipeline) override;
        
    bool resetQueryPool_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) override;
    bool beginQuery_impl(IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS> flags) override;
    bool endQuery_impl(IQueryPool* queryPool, uint32_t query) override;
    bool copyQueryPoolResults_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags) override;
    bool writeTimestamp_impl(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* queryPool, uint32_t query) override;

    bool bindDescriptorSets_impl(asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const pipeline_layout_t* layout_, uint32_t firstSet_, const uint32_t descriptorSetCount_,
        const descriptor_set_t* const* const descriptorSets_, const uint32_t dynamicOffsetCount_ = 0u, const uint32_t* dynamicOffsets_ = nullptr) override;
    bool pushConstants_impl(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) override;
    bool clearColorImage_impl(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override;
    bool clearDepthStencilImage_impl(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override;
    bool clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects) override;
    bool fillBuffer_impl(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data) override;
    bool updateBuffer_impl(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) override;
    bool executeCommands_impl(uint32_t count, IGPUCommandBuffer* const* const cmdbufs) override;

    inline const void* getNativeHandle() const override {return nullptr;}

private:
    SOpenGLContextIndependentCache m_stateCache;
    IGPUCommandPool::CCommandSegment* m_GLSegmentListHead = {};
    IGPUCommandPool::CCommandSegment* m_GLSegmentListTail = nullptr;
};

}

#endif
