#ifndef __NBL_I_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_I_COMMAND_BUFFER_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include "nbl/asset/IImage.h"
#include "nbl/asset/IRenderpass.h"
#include "nbl/asset/ISampler.h"
#include "nbl/asset/ISpecializedShader.h"

#include <type_traits>

namespace nbl {
namespace asset
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
enum E_STENCIL_FACE_FLAGS : uint32_t
{
    ESFF_FRONT_BIT = 0x01,
    ESFF_BACK_BIT = 0x02,
    ESFF_FACE_AND_FRONT = 0x03
};
enum E_QUERY_CONTROL_FLAGS : uint32_t
{
    EQCF_PRECISE_BIT = 0x01
};
enum E_QUERY_RESULT_FLAGS : uint32_t
{
    EQRF_64_BIT = 0x01,
    EQRF_WAIT_BIT = 0x02,
    EQRF_WITH_AVAILABILITY_BIT = 0x04,
    EQRF_PARTIAL_BIT = 0x08
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
    uint32_t uint32_t[4];
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

enum E_SUBPASS_CONTENTS : uint32_t
{
    ESC_INLINE = 0,
    ESC_SECONDARY_COMMAND_BUFFERS = 1
};

template <
    typename BufferType, 
    typename ImageType, 
    typename RenderpassType, 
    typename FramebufferType, 
    typename GraphicsPipelineType, 
    typename ComputePipelineType,
    typename DescSetType,
    typename PipelineLayoutType,
    typename EventType
>
class ICommandBuffer
{
protected:
    using buffer_t = BufferType;
    using image_t = ImageType;
    using renderpass_t = RenderpassType;
    using framebuffer_t = FramebufferType;
    using graphics_pipeline_t = GraphicsPipelineType;
    using compute_pipeline_t = ComputePipelineType;
    using descriptor_set_t = DescSetType;
    using pipeline_layout_t = PipelineLayoutType;
    using event_t = EventType;

public:
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
        const buffer_t* buffer;
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
        const image_t* image;
        asset::IImage::SSubresourceRange subresourceRange;
    };
    struct SRenderpassBeginInfo
    {
        const renderpass_t* renderpass;
        const framebuffer_t* framebuffer;
        VkRect2D renderArea;
        uint32_t clearValueCount;
        const SClearValue* clearValues;
    };

    virtual E_LEVEL getLevel() const = 0;

    //! `_flags` takes bits from E_RESET_FLAGS
    virtual void reset(uint32_t _flags)
    {
        assert(m_state!=ES_PENDING);
        m_state = ES_INITIAL;
    }

    virtual void end()
    {
        assert(m_state!=ES_PENDING);
        m_state = ES_EXECUTABLE;
    }

    virtual void bindIndexBuffer(buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType) = 0;
    virtual void draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) = 0;
    virtual void drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) = 0;
    virtual void drawIndirect(buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
    virtual void drawIndexedIndirect(buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;

    virtual void setViewport(uint32_t firstViewport, uint32_t viewportCount, const SViewport* pViewports) = 0;

    virtual void setLineWidth(float lineWidth) = 0;
    virtual void setDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor) = 0;

    virtual void setBlendConstants(const float blendConstants[4]) = 0;

    virtual void copyBuffer(buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const SBufferCopy* pRegions) = 0;
    virtual void copyImage(image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) = 0;
    virtual void copyBufferToImage(buffer_t* srcBuffer, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) = 0;
    virtual void copyImageToBuffer(buffer_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) = 0;
    virtual void blitImage(image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) = 0;
    virtual void resolveImage(image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const SImageResolve* pRegions) = 0;

    virtual void bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, buffer_t** pBuffers, const size_t* pOffsets) = 0;

    virtual void setScissor(uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors) = 0;
    virtual void setDepthBounds(float minDepthBounds, float maxDepthBounds) = 0;
    virtual void setStencilCompareMask(E_STENCIL_FACE_FLAGS faceMask, uint32_t compareMask) = 0;
    virtual void setStencilWriteMask(E_STENCIL_FACE_FLAGS faceMask, uint32_t writeMask) = 0;
    virtual void setStencilReference(E_STENCIL_FACE_FLAGS faceMask, uint32_t reference) = 0;

    virtual void dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) = 0;
    virtual void dispatchIndirect(buffer_t* buffer, size_t offset) = 0;
    virtual void dispatchBase(uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) = 0;

    virtual void setEvent(event_t* event, asset::E_PIPELINE_STAGE_FLAGS stageMask) = 0;
    virtual void resetEvent(event_t* event, asset::E_PIPELINE_STAGE_FLAGS stageMask) = 0;

    virtual void waitEvents(uint32_t eventCount, event_t** pEvents, std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        uint32_t memoryBarrierCount, const SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers
    ) = 0;

    virtual void pipelineBarrier(uint32_t eventCount, const VkEvent* pEvents, std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        std::underlying_type_t<E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers) = 0;

    virtual void beginRenderPass(const SRenderpassBeginInfo* pRenderPassBegin, E_SUBPASS_CONTENTS content) = 0;
    virtual void nextSubpass(E_SUBPASS_CONTENTS contents) = 0;
    virtual void endRenderPass() = 0;

    virtual void setDeviceMask(uint32_t deviceMask) { m_deviceMask = deviceMask; }

    //those two instead of bindPipeline(E_PIPELINE_BIND_POINT, pipeline)
    virtual void bindGraphicsPipeline(graphics_pipeline_t* pipeline) = 0;
    virtual void bindComputePipeline(compute_pipeline_t* pipeline) = 0;

    //virtual void resetQueryPool(IGPUQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) = 0;
    //virtual void beginQuery(IGPUQueryPool* queryPool, uint32_t entry, std::underlying_type_t<E_QUERY_CONTROL_FLAGS> flags) = 0;
    //virtual void endQuery(IGPUQueryPool* queryPool, uint32_t query) = 0;
    //virtual void copyQueryPoolResults(IGPUQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, std::underlying_type_t<E_QUERY_RESULT_FLAGS> flags) = 0;
    //virtual void writeTimestamp(std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> pipelineStage, IGPUQueryPool* queryPool, uint32_t query) = 0;

    // E_PIPELINE_BIND_POINT needs to be in asset namespace or divide this into two functions (for graphics and compute)
    /*virtual void bindDescriptorSets(E_PIPELINE_BIND_POINT pipelineBindPoint, pipeline_layout_t* layout, uint32_t firstSet, uint32_t descriptorSetCount,
        descriptor_set_t** pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t pDynamicOffsets
    ) = 0;*/
    virtual void pushConstants(pipeline_layout_t* layout, std::underlying_type_t<asset::ISpecializedShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) = 0;

    virtual void clearColorImage(image_t* image, asset::E_IMAGE_LAYOUT imageLayout, const SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) = 0;
    virtual void clearDepthStencilImage(image_t* image, asset::E_IMAGE_LAYOUT imageLayout, const SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) = 0;
    virtual void clearAttachments(uint32_t attachmentCount, const SClearAttachment* pAttachments, uint32_t rectCount, const SClearRect* pRects) = 0;
    virtual void fillBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data) = 0;
    virtual void updateBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) = 0;

protected:
    virtual ~ICommandBuffer() = default;

    //! `_flags` takes bits from E_USAGE
    virtual void begin(uint32_t _flags)
    {
        assert(m_state!=ES_PENDING);
        assert(m_state!=ES_RECORDING);

        m_state = ES_RECORDING;
        m_recordingFlags = _flags;
    }

    void setState(E_STATE _state)
    {
        m_state = _state;
    }

    // Flags from E_USAGE
    uint32_t m_recordingFlags = 0u;
    E_STATE m_state = ES_INITIAL;
    //future
    uint32_t m_deviceMask = ~0u;
};

}}

#endif