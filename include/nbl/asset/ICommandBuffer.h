#ifndef __NBL_I_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_I_COMMAND_BUFFER_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include "nbl/asset/IImage.h"
#include "nbl/asset/IRenderpass.h"
#include "nbl/asset/ISampler.h"
#include "nbl/asset/ISpecializedShader.h"
#include "nbl/asset/ECommonEnums.h"
#include "nbl/video/IGPUMeshBuffer.h"

#include <type_traits>

namespace nbl::asset
{

//TODO move and possibly rename
struct SBufferCopy
{
    size_t srcOffset;
    size_t dstOffset;
    size_t size;
};
struct SImageBlit
{
    asset::IImage::SSubresourceLayers srcSubresource;
    asset::VkOffset3D srcOffsets[2];
    asset::IImage::SSubresourceLayers dstSubresource;
    asset::VkOffset3D dstOffsets[2];
};
struct SImageResolve
{
    asset::IImage::SSubresourceLayers srcSubresource;
    asset::VkOffset3D srcOffset;
    asset::IImage::SSubresourceLayers dstSubresource;
    asset::VkOffset3D dstOffset;
    asset::VkExtent3D extent;
};
struct SViewport
{
    float x, y;
    float width, height;
    float minDepth, maxDepth;
};
struct VkOffset2D
{
    int32_t x;
    int32_t y;
};
struct VkExtent2D
{
    uint32_t width;
    uint32_t height;
};
struct VkRect2D
{
    VkOffset2D    offset;
    VkExtent2D    extent;
};

struct SMemoryBarrier
{
    asset::E_ACCESS_FLAGS srcAccessMask;
    asset::E_ACCESS_FLAGS dstAccessMask;
};

union SClearColorValue
{
    float float32[4];
    int32_t int32[4];
    uint32_t uint32[4];
};
struct SClearDepthStencilValue
{
    float depth;
    uint32_t stencil;
};
union SClearValue
{
    SClearColorValue color;
    SClearDepthStencilValue depthStencil;
};

struct SClearAttachment
{
    asset::IImage::E_ASPECT_FLAGS aspectMask;
    uint32_t colorAttachment;
    SClearValue clearValue;
};

struct SClearRect
{
    VkRect2D rect;
    uint32_t baseArrayLayer;
    uint32_t layerCount;
};

template <
    typename BufferType, 
    typename ImageType, 
    typename ImageViewType,
    typename RenderpassType, 
    typename FramebufferType, 
    typename GraphicsPipelineType, 
    typename ComputePipelineType,
    typename DescSetType,
    typename PipelineLayoutType,
    typename EventType,
    typename CommandBufferType
>
class ICommandBuffer
{
protected:
    using buffer_t = BufferType;
    using image_t = ImageType;
    using image_view_t = ImageViewType;
    using renderpass_t = RenderpassType;
    using framebuffer_t = FramebufferType;
    using graphics_pipeline_t = GraphicsPipelineType;
    using compute_pipeline_t = ComputePipelineType;
    using descriptor_set_t = DescSetType;
    using pipeline_layout_t = PipelineLayoutType;
    using event_t = EventType;
    using cmdbuf_t = CommandBufferType;

public:
    _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_PUSH_CONSTANT_BYTESIZE = 128u;

    enum E_RESET_FLAGS : uint32_t
    {
        ERF_RELEASE_RESOURCES_BIT = 0x01
    };

    enum E_STATE : uint32_t
    {
        ES_INITIAL,
        ES_RECORDING,
        ES_EXECUTABLE,
        ES_PENDING,
        ES_INVALID
    };

    enum E_USAGE : uint32_t
    {
        EU_ONE_TIME_SUBMIT_BIT = 0x01,
        EU_RENDER_PASS_CONTINUE_BIT = 0x02,
        EU_SIMULTANEOUS_USE_BIT = 0x04
    };

    enum E_LEVEL : uint32_t
    {
        EL_PRIMARY,
        EL_SECONDARY
    };

    struct SBufferMemoryBarrier
    {
        SMemoryBarrier barrier;
        uint32_t srcQueueFamilyIndex;
        uint32_t dstQueueFamilyIndex;
        core::smart_refctd_ptr<const buffer_t> buffer;
        size_t offset;
        size_t size;
    };
    struct SImageMemoryBarrier
    {
        SMemoryBarrier barrier;
        asset::E_IMAGE_LAYOUT oldLayout;
        asset::E_IMAGE_LAYOUT newLayout;
        uint32_t srcQueueFamilyIndex;
        uint32_t dstQueueFamilyIndex;
        core::smart_refctd_ptr<const image_t> image;
        asset::IImage::SSubresourceRange subresourceRange;
    };
    struct SRenderpassBeginInfo
    {
        core::smart_refctd_ptr<const renderpass_t> renderpass;
        core::smart_refctd_ptr<const framebuffer_t> framebuffer;
        VkRect2D renderArea;
        uint32_t clearValueCount;
        const SClearValue* clearValues;
    };
    struct SDependencyInfo
    {
        E_DEPENDENCY_FLAGS dependencyFlags;
        uint32_t memBarrierCount;
        const SMemoryBarrier* memBarriers;
        uint32_t bufBarrierCount;
        const SBufferMemoryBarrier* bufBarriers;
        uint32_t imgBarrierCount;
        const SImageMemoryBarrier* imgBarriers;
    };

    E_STATE getState() const { return m_state; }

    E_LEVEL getLevel() const { return m_level; }

    // hm now i think having begin(), reset() and end() as command buffer API is a little weird

    //! `_flags` takes bits from E_USAGE
    virtual void begin(uint32_t _flags)
    {
        assert(m_state != ES_PENDING);
        m_state = ES_RECORDING;
        m_recordingFlags = _flags;
    }

    //! `_flags` takes bits from E_RESET_FLAGS
    virtual bool reset(uint32_t _flags)
    {
        if (m_state==ES_PENDING);
            return false;
        m_state = ES_INITIAL;

        return true;
    }

    virtual void end()
    {
        assert(m_state!=ES_PENDING);
        m_state = ES_EXECUTABLE;
    }

    virtual bool bindIndexBuffer(const buffer_t* buffer, size_t offset, E_INDEX_TYPE indexType) = 0;
    virtual bool draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) = 0;
    virtual bool drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) = 0;
    virtual bool drawIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
    virtual bool drawIndexedIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
    virtual bool drawMeshBuffer(const nbl::video::IGPUMeshBuffer* meshBuffer) = 0;

    virtual bool setViewport(uint32_t firstViewport, uint32_t viewportCount, const SViewport* pViewports) = 0;

    virtual bool setLineWidth(float lineWidth) = 0;
    virtual bool setDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor) = 0;

    virtual bool setBlendConstants(const float blendConstants[4]) = 0;

    virtual bool copyBuffer(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const SBufferCopy* pRegions) = 0;
    virtual bool copyImage(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) = 0;
    virtual bool copyBufferToImage(const buffer_t* srcBuffer, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) = 0;
    virtual bool copyImageToBuffer(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) = 0;
    virtual bool blitImage(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter)
    {
        for (uint32_t i = 0u; i < regionCount; ++i)
        {
            if (pRegions[i].dstSubresource.aspectMask != pRegions[i].srcSubresource.aspectMask)
                return false;
            if (pRegions[i].dstSubresource.layerCount != pRegions[i].srcSubresource.layerCount)
                return false;
        }
        return true;
    }
    virtual bool resolveImage(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const SImageResolve* pRegions) = 0;

    virtual bool bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, const buffer_t*const *const pBuffers, const size_t* pOffsets) = 0;

    virtual bool setScissor(uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors) = 0;
    virtual bool setDepthBounds(float minDepthBounds, float maxDepthBounds) = 0;
    virtual bool setStencilCompareMask(E_STENCIL_FACE_FLAGS faceMask, uint32_t compareMask) = 0;
    virtual bool setStencilWriteMask(E_STENCIL_FACE_FLAGS faceMask, uint32_t writeMask) = 0;
    virtual bool setStencilReference(E_STENCIL_FACE_FLAGS faceMask, uint32_t reference) = 0;

    virtual bool dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) = 0;
    virtual bool dispatchIndirect(const buffer_t* buffer, size_t offset) = 0;
    virtual bool dispatchBase(uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) = 0;

    virtual bool setEvent(event_t* event, const SDependencyInfo& depInfo) = 0;
    virtual bool resetEvent(event_t* event, asset::E_PIPELINE_STAGE_FLAGS stageMask) = 0;

    virtual bool waitEvents(uint32_t eventCount, event_t*const *const pEvents, const SDependencyInfo* depInfos) = 0;

    virtual bool pipelineBarrier(std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        std::underlying_type_t<E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers) = 0;

    virtual bool beginRenderPass(const SRenderpassBeginInfo* pRenderPassBegin, E_SUBPASS_CONTENTS content) = 0;
    virtual bool nextSubpass(E_SUBPASS_CONTENTS contents) = 0;
    virtual bool endRenderPass() = 0;

    virtual bool setDeviceMask(uint32_t deviceMask) { m_deviceMask = deviceMask; return true; }

    //those two instead of bindPipeline(E_PIPELINE_BIND_POINT, pipeline)
    virtual bool bindGraphicsPipeline(const graphics_pipeline_t* pipeline) = 0;
    virtual bool bindComputePipeline(const compute_pipeline_t* pipeline) = 0;

    //virtual bool resetQueryPool(IGPUQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) = 0;
    //virtual bool beginQuery(IGPUQueryPool* queryPool, uint32_t entry, std::underlying_type_t<E_QUERY_CONTROL_FLAGS> flags) = 0;
    //virtual bool endQuery(IGPUQueryPool* queryPool, uint32_t query) = 0;
    //virtual bool copyQueryPoolResults(IGPUQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, std::underlying_type_t<E_QUERY_RESULT_FLAGS> flags) = 0;
    //virtual bool writeTimestamp(std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> pipelineStage, IGPUQueryPool* queryPool, uint32_t query) = 0;

    // E_PIPELINE_BIND_POINT needs to be in asset namespace or divide this into two functions (for graphics and compute)
    virtual bool bindDescriptorSets(E_PIPELINE_BIND_POINT pipelineBindPoint, const pipeline_layout_t* layout, uint32_t firstSet, uint32_t descriptorSetCount,
        const descriptor_set_t*const *const pDescriptorSets, core::smart_refctd_dynamic_array<uint32_t> dynamicOffsets = nullptr
    ) = 0;
    virtual bool pushConstants(const pipeline_layout_t* layout, std::underlying_type_t<asset::ISpecializedShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) = 0;

    virtual bool clearColorImage(image_t* image, asset::E_IMAGE_LAYOUT imageLayout, const SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) = 0;
    virtual bool clearDepthStencilImage(image_t* image, asset::E_IMAGE_LAYOUT imageLayout, const SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) = 0;
    virtual bool clearAttachments(uint32_t attachmentCount, const SClearAttachment* pAttachments, uint32_t rectCount, const SClearRect* pRects) = 0;
    virtual bool fillBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data) = 0;
    virtual bool updateBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) = 0;

    virtual bool executeCommands(uint32_t count, cmdbuf_t*const *const cmdbufs)
    {
        for (uint32_t i = 0u; i < count; ++i)
        {
            if (cmdbufs[i]->getLevel() != EL_SECONDARY)
                return false;
        }
        return true;
    }

    virtual bool regenerateMipmaps(image_view_t* imgview) = 0;

protected:
    ICommandBuffer(E_LEVEL lvl) : m_level(lvl) {}
    virtual ~ICommandBuffer() = default;

    void setState(E_STATE _state)
    {
        m_state = _state;
    }

    E_LEVEL m_level;
    // Flags from E_USAGE
    uint32_t m_recordingFlags = 0u;
    E_STATE m_state = ES_INITIAL;
    //future
    uint32_t m_deviceMask = ~0u;
};

}

#endif