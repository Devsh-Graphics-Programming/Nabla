#ifndef __NBL_I_GPU_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_BUFFER_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include "IGPUBuffer.h"
#include <nbl/asset/IMeshBuffer.h>
#include <nbl/video/IGPUImage.h>
#include <nbl/video/IGPUSampler.h>
#include "IVideoDriver.h"

#include <type_traits>

namespace nbl
{
namespace video
{
//TODO move and possibly rename
struct VkBufferCopy
{
    size_t srcOffset;
    size_t dstOffset;
    size_t size;
};
struct VkImageBlit
{
    asset::IImage::SSubresourceLayers srcSubresource;
    asset::VkOffset3D srcOffsets[2];
    asset::IImage::SSubresourceLayers dstSubresource;
    asset::VkOffset3D dstOffsets[2];
};
struct VkImageResolve
{
    asset::IImage::SSubresourceLayers srcSubresource;
    asset::VkOffset3D srcOffset;
    asset::IImage::SSubresourceLayers dstSubresource;
    asset::VkOffset3D dstOffset;
    asset::VkExtent3D extent;
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
    VkOffset2D offset;
    VkExtent2D extent;
};
enum E_STENCIL_FACE_FLAGS : uint32_t
{
    ESFF_FRONT_BIT = 0x01,
    ESFF_BACK_BIT = 0x02,
    ESFF_FACE_AND_FRONT = 0x03
};
enum E_PIPELINE_STAGE_FLAGS : uint32_t
{
    EPSF_TOP_OF_PIPE_BIT = 0x00000001,
    EPSF_DRAW_INDIRECT_BIT = 0x00000002,
    EPSF_VERTEX_INPUT_BIT = 0x00000004,
    EPSF_VERTEX_SHADER_BIT = 0x00000008,
    EPSF_TESSELLATION_CONTROL_SHADER_BIT = 0x00000010,
    EPSF_TESSELLATION_EVALUATION_SHADER_BIT = 0x00000020,
    EPSF_GEOMETRY_SHADER_BIT = 0x00000040,
    EPSF_FRAGMENT_SHADER_BIT = 0x00000080,
    EPSF_EARLY_FRAGMENT_TESTS_BIT = 0x00000100,
    EPSF_LATE_FRAGMENT_TESTS_BIT = 0x00000200,
    EPSF_COLOR_ATTACHMENT_OUTPUT_BIT = 0x00000400,
    EPSF_COMPUTE_SHADER_BIT = 0x00000800,
    EPSF_TRANSFER_BIT = 0x00001000,
    EPSF_BOTTOM_OF_PIPE_BIT = 0x00002000,
    EPSF_HOST_BIT = 0x00004000,
    EPSF_ALL_GRAPHICS_BIT = 0x00008000,
    EPSF_ALL_COMMANDS_BIT = 0x00010000,
    EPSF_TRANSFORM_FEEDBACK_BIT_EXT = 0x01000000,
    EPSF_CONDITIONAL_RENDERING_BIT_EXT = 0x00040000,
    EPSF_RAY_TRACING_SHADER_BIT_KHR = 0x00200000,
    EPSF_ACCELERATION_STRUCTURE_BUILD_BIT_KHR = 0x02000000,
    EPSF_SHADING_RATE_IMAGE_BIT_NV = 0x00400000,
    EPSF_TASK_SHADER_BIT_NV = 0x00080000,
    EPSF_MESH_SHADER_BIT_NV = 0x00100000,
    EPSF_FRAGMENT_DENSITY_PROCESS_BIT_EXT = 0x00800000,
    EPSF_COMMAND_PREPROCESS_BIT_NV = 0x00020000
};
enum E_DEPENDENCY_FLAGS : uint32_t
{
    EDF_BY_REGION_BIT = 0x01,
    EDF_DEVICE_GROUP_BIT = 0x04,
    EDF_VIEW_LOCAL_BIT = 0x02
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

class IGPUQueue;

class IGPUCommandBuffer : public core::IReferenceCounted
{
    friend class IGPUQueue;

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

    IGPUCommandBuffer(uint32_t _familyIx)
        : m_familyIndex(_familyIx)
    {
    }

    virtual E_LEVEL getLevel() const = 0;

    uint32_t getQueueFamilyIndex() const { return m_familyIndex; }

    //! `_flags` takes bits from E_RESET_FLAGS
    virtual void reset(uint32_t _flags)
    {
        assert(m_state != ES_PENDING);
        m_state = ES_INITIAL;
    }

    virtual void end()
    {
        assert(m_state != ES_PENDING);
        m_state = ES_EXECUTABLE;
    }

    virtual void bindIndexBuffer(IGPUBuffer* buffer, size_t offset, asset::E_INDEX_TYPE indexType) = 0;
    virtual void draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) = 0;
    virtual void drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) = 0;
    virtual void drawIndirect(IGPUBuffer* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
    virtual void drawIndexedIndirect(IGPUBuffer* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;

    //virtual void setViewport(uint32_t firstViewport, uint32_t viewportCount, const VkViewport* pViewports) = 0;

    virtual void setLineWidth(float lineWidth) = 0;
    virtual void setDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor) = 0;

    virtual void setBlendConstants(const float blendConstants[4]) = 0;

    virtual void copyBuffer(IGPUBuffer* srcBuffer, IGPUBuffer* dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions) = 0;
    //TODO theres no E_LAYOUT enum!
    //virtual void copyImage(IGPUImage* srcImage, IGPUImage::E_LAYOUT srcImageLayout, IGPUImage* dstImage, IGPUImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions) = 0;
    //virtual void copyBufferToImage(IGPUBuffer* srcBuffer, IGPUImage* dstImage, IGPUImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions) = 0;
    //virtual void copyImageToBuffer(IGPUBuffer* srcImage, IGPUImage::E_LAYOUT srcImageLayout, IGPUBuffer* dstBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions) = 0;
    //virtual void blitImage(IGPUImage* srcImage, IGPUImage::E_LAYOUT srcImageLayout, IGPUImage* dstImage, IGPUImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const VkImageBlit* pRegions, IGPUSampler::E_TEXTURE_FILTER filter) = 0;
    //virtual void resolveImage(IGPUImage* srcImage, IGPUImage::E_LAYOUT srcImageLayout, IGPUImage* dstImage, IGPUImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const VkImageResolve* pRegions) = 0;

    virtual void bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, IGPUBuffer** pBuffers, const size_t pOffsets) = 0;

    virtual void setScissor(uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors) = 0;
    virtual void setDepthBounds(float minDepthBounds, float maxDepthBounds) = 0;
    virtual void setStencilCompareMask(E_STENCIL_FACE_FLAGS faceMask, uint32_t compareMask) = 0;
    virtual void setStencilWriteMask(E_STENCIL_FACE_FLAGS faceMask, uint32_t writeMask) = 0;
    virtual void setStencilReference(E_STENCIL_FACE_FLAGS faceMask, uint32_t reference) = 0;

    virtual void dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) = 0;
    virtual void dispatchIndirect(IGPUBuffer* buffer, size_t offset) = 0;
    virtual void dispatchBase(uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) = 0;

    //virtual void setEvent(VkEvent event, VkPipelineStageFlags stageMask) = 0;
    //virtual void resetEvent(VkEvent event, VkPipelineStageFlags stageMask) = 0;

    //`srcStagemask`, `dstStageMask` take bits from E_PIPELINE_STAGE_FLAGS
    /*virtual void waitEvents(uint32_t eventCount, const VkEvent* pEvents, std::underlying_type_t<E_PIPELINE_STAGE_FLAGS> srcStageMask, std::underlying_type_t<E_PIPELINE_STAGE_FLAGS> dstStageMask,
        uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers
    ) = 0;*/

    /*virtual void pipelineBarrier(uint32_t eventCount, const VkEvent* pEvents, std::underlying_type_t<E_PIPELINE_STAGE_FLAGS> srcStageMask, std::underlying_type_t<E_PIPELINE_STAGE_FLAGS> dstStageMask,
        std::underlying_type_t<E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers) = 0;*/

    //virtual void beginRenderPass(const VkRenderPassBeginInfo* pRenderPassBegin, VkSubpassContents content) = 0;
    //virtual void nextSubpass(VkSubpassContents contents) = 0;
    //virtual void endRenderPass() = 0;

    virtual void setDeviceMask(uint32_t deviceMask) { m_deviceMask = deviceMask; }

    //those two instead of bindPipeline(E_PIPELINE_BIND_POINT, pipeline)
    virtual void bindGraphicsPipeline(IGPURenderpassIndependentPipeline* pipeline) = 0;
    virtual void bindComputePipeline(IGPUComputePipeline* pipeline) = 0;

    //virtual void resetQueryPool(IGPUQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) = 0;
    //virtual void beginQuery(IGPUQueryPool* queryPool, uint32_t entry, std::underlying_type_t<E_QUERY_CONTROL_FLAGS> flags) = 0;
    //virtual void endQuery(IGPUQueryPool* queryPool, uint32_t query) = 0;
    //virtual void copyQueryPoolResults(IGPUQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, IGPUBuffer* dstBuffer, size_t dstOffset, size_t stride, std::underlying_type_t<E_QUERY_RESULT_FLAGS> flags) = 0;
    //virtual void writeTimestamp(std::underlying_type_t<E_PIPELINE_STAGE_FLAGS> pipelineStage, IGPUQueryPool* queryPool, uint32_t query) = 0;

    virtual void bindDescriptorSets(E_PIPELINE_BIND_POINT pipelineBindPoint, IGPUPipelineLayout* layout, uint32_t firstSet, uint32_t descriptorSetCount,
        IGPUDescriptorSet** pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t pDynamicOffsets) = 0;
    virtual void pushConstants(IGPUPipelineLayout* layout, std::underlying_type_t<IGPUSpecializedShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) = 0;

    //virtual void clearColorImage(IGPUImage* image, IGPUImage::E_LAYOUT imageLayout, const VkClearColorValue* pColor, uint32_t rangeCount, const VkImageSubresourceRange* pRanges) = 0;
    //virtual void clearDepthStencilImage(IGPUImage* image, IGPUImage::E_LAYOUT imageLayout, const VkClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const VkImageSubresourceRange* pRanges) = 0;
    //virtual void clearAttachments(uint32_t attachmentCount, const VkClearAttachment* pAttachments, uint32_t rectCount, const VkClearRect* pRects) = 0;
    virtual void fillBuffer(IGPUBuffer* dstBuffer, size_t dstOffset, size_t size, uint32_t data) = 0;
    virtual void updateBuffer(IGPUBuffer* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) = 0;

protected:
    //! `_flags` takes bits from E_USAGE
    virtual void begin(uint32_t _flags)
    {
        assert(m_state != ES_PENDING);
        assert(m_state != ES_RECORDING);

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
    const uint32_t m_familyIndex;
    //future
    uint32_t m_deviceMask = ~0u;
};

}
}

#endif