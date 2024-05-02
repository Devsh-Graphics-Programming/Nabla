#include "nbl/video/CVulkanCommandBuffer.h"

#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanQueryPool.h"


using namespace nbl;
using namespace nbl::video;


const VolkDeviceTable& CVulkanCommandBuffer::getFunctionTable() const
{
    return static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable()->vk;
}


bool CVulkanCommandBuffer::begin_impl(const core::bitflag<USAGE> recordingFlags, const SInheritanceInfo* const inheritanceInfo)
{
    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkDeviceGroupCommandBufferBeginInfo
    beginInfo.flags = static_cast<VkCommandBufferUsageFlags>(recordingFlags.value);

    VkCommandBufferInheritanceInfo vk_inheritanceInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO };
    if (inheritanceInfo)
    {
        vk_inheritanceInfo.renderPass = static_cast<const CVulkanRenderpass*>(inheritanceInfo->renderpass)->getInternalObject();
        vk_inheritanceInfo.subpass = inheritanceInfo->subpass;
        // From the spec:
        // Specifying the exact framebuffer that the secondary command buffer will be
        // executed with may result in better performance at command buffer execution time.
        if (inheritanceInfo->framebuffer)
            vk_inheritanceInfo.framebuffer = static_cast<const CVulkanFramebuffer*>(inheritanceInfo->framebuffer)->getInternalObject();
        vk_inheritanceInfo.occlusionQueryEnable = inheritanceInfo->occlusionQueryEnable;
        vk_inheritanceInfo.queryFlags = static_cast<VkQueryControlFlags>(inheritanceInfo->queryFlags.value);
        vk_inheritanceInfo.pipelineStatistics = static_cast<VkQueryPipelineStatisticFlags>(0u); // must be 0

        beginInfo.pInheritanceInfo = &vk_inheritanceInfo;
    }

    return getFunctionTable().vkBeginCommandBuffer(m_cmdbuf,&beginInfo)==VK_SUCCESS;
}


template<typename vk_barrier_t, typename ResourceBarrier>
void fill(vk_barrier_t& out, const ResourceBarrier& in, uint32_t selfQueueFamilyIndex, const bool concurrentSharing=false)
{
    auto getVkQueueIndexFrom = [](const uint32_t nblIx)->uint32_t
    {
        switch (nblIx)
        {
            case IQueue::FamilyIgnored:
                return VK_QUEUE_FAMILY_IGNORED;
            case IQueue::FamilyExternal:
                return VK_QUEUE_FAMILY_EXTERNAL;
            case IQueue::FamilyForeign:
                return VK_QUEUE_FAMILY_FOREIGN_EXT;
            default:
                break;
        }
        return nblIx;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-buffer-04088
    if (concurrentSharing)
        selfQueueFamilyIndex = getVkQueueIndexFrom(IQueue::FamilyIgnored);
    if constexpr (!std::is_same_v<vk_barrier_t,VkMemoryBarrier2>)
    {
        out.srcQueueFamilyIndex = selfQueueFamilyIndex;
        out.dstQueueFamilyIndex = selfQueueFamilyIndex;
    }
    const asset::SMemoryBarrier* memoryBarrier;
    if constexpr (std::is_same_v<IGPUCommandBuffer::SOwnershipTransferBarrier,ResourceBarrier>)
    {
        memoryBarrier = &in.dep;
        // in.otherQueueFamilyIndex==selfQueueFamilyIndex not resulting in ownership transfer is implicit
        if (!concurrentSharing && in.otherQueueFamilyIndex!=IQueue::FamilyIgnored)
        switch (in.ownershipOp)
        {
            case IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE:
                out.dstQueueFamilyIndex = in.otherQueueFamilyIndex;
                break;
            case IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE:
                out.srcQueueFamilyIndex = in.otherQueueFamilyIndex;
                break;
        }
    }
    else
        memoryBarrier = &in;
    out.srcStageMask = getVkPipelineStageFlagsFromPipelineStageFlags(memoryBarrier->srcStageMask);
    out.srcAccessMask = getVkAccessFlagsFromAccessFlags(memoryBarrier->srcAccessMask);
    out.dstStageMask = getVkPipelineStageFlagsFromPipelineStageFlags(memoryBarrier->dstStageMask);
    out.dstAccessMask = getVkAccessFlagsFromAccessFlags(memoryBarrier->dstAccessMask);
}

template<typename SubresourceRange> requires nbl::is_any_of_v<SubresourceRange,IGPUImage::SSubresourceRange,IGPUImage::SSubresourceLayers>
static inline auto getVkImageSubresourceFrom(const SubresourceRange& range) -> std::conditional_t<std::is_same_v<SubresourceRange,IGPUImage::SSubresourceRange>,VkImageSubresourceRange,VkImageSubresourceLayers>
{
    constexpr bool rangeNotLayers =  std::is_same_v<SubresourceRange,IGPUImage::SSubresourceRange>;

    std::conditional_t<rangeNotLayers,VkImageSubresourceRange,VkImageSubresourceLayers> retval = {};
    retval.aspectMask = static_cast<VkImageAspectFlags>(range.aspectMask.value);
    if constexpr (rangeNotLayers)
    {
        retval.baseMipLevel = range.baseMipLevel;
        retval.levelCount = range.levelCount;
    }
    else
        retval.mipLevel = range.mipLevel;
    retval.baseArrayLayer = range.baseArrayLayer;
    retval.layerCount = range.layerCount;
    return retval;
}

template<typename ResourceBarrier>
VkDependencyInfoKHR fill(
    VkMemoryBarrier2* const memoryBarriers, VkBufferMemoryBarrier2* const bufferBarriers, VkImageMemoryBarrier2* const imageBarriers,
    const IGPUCommandBuffer::SDependencyInfo<ResourceBarrier>& depInfo, const uint32_t selfQueueFamilyIndex=IQueue::FamilyIgnored
) {
    VkDependencyInfoKHR info = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,nullptr };
    auto outMem = memoryBarriers;
    for (const auto& in : depInfo.memBarriers)
    {
        outMem->sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
        outMem->pNext = nullptr;
        fill(*(outMem++),in,selfQueueFamilyIndex);
    }
    auto outBuf = bufferBarriers;
    for (const auto& in : depInfo.bufBarriers)
    {
        outBuf->sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
        outBuf->pNext = nullptr; // VkExternalMemoryAcquireUnmodifiedEXT

        fill(*outBuf,in.barrier,selfQueueFamilyIndex,in.range.buffer->getCachedCreationParams().isConcurrentSharing());
        outBuf->buffer = static_cast<const CVulkanBuffer*>(in.range.buffer.get())->getInternalObject();
        outBuf->offset = in.range.offset;
        outBuf->size = in.range.size;
        outBuf++;
    }
    auto outImg = imageBarriers;
    for (const auto& in : depInfo.imgBarriers)
    {
        outImg->sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR;
        outImg->pNext = nullptr; // VkExternalMemoryAcquireUnmodifiedEXT or VkSampleLocationsInfoEXT

        outImg->oldLayout = getVkImageLayoutFromImageLayout(in.oldLayout);
        outImg->newLayout = getVkImageLayoutFromImageLayout(in.newLayout);
        fill(*outImg,in.barrier,selfQueueFamilyIndex,in.image->getCachedCreationParams().isConcurrentSharing());
        outImg->image = static_cast<const CVulkanImage*>(in.image)->getInternalObject();
        outImg->subresourceRange = getVkImageSubresourceFrom(in.subresourceRange);
        outImg++;
    }
    info.dependencyFlags = 0u;
    info.memoryBarrierCount = depInfo.memBarriers.size();
    info.pMemoryBarriers = memoryBarriers;
    info.bufferMemoryBarrierCount = depInfo.bufBarriers.size();
    info.pBufferMemoryBarriers = bufferBarriers;
    info.imageMemoryBarrierCount = depInfo.imgBarriers.size();
    info.pImageMemoryBarriers = imageBarriers;
    return info;
}

bool CVulkanCommandBuffer::setEvent_impl(IEvent* const _event, const SEventDependencyInfo& depInfo)
{
    IGPUCommandPool::StackAllocation<VkMemoryBarrier2KHR> memoryBarriers(m_cmdpool,depInfo.memBarriers.size());
    IGPUCommandPool::StackAllocation<VkBufferMemoryBarrier2KHR> bufferBarriers(m_cmdpool,depInfo.bufBarriers.size());
    IGPUCommandPool::StackAllocation<VkImageMemoryBarrier2KHR> imageBarriers(m_cmdpool,depInfo.imgBarriers.size());
    if (!memoryBarriers || !bufferBarriers || !imageBarriers)
        return false;

    auto info = fill(memoryBarriers.data(),bufferBarriers.data(),imageBarriers.data(),depInfo);
    getFunctionTable().vkCmdSetEvent2(m_cmdbuf,static_cast<CVulkanEvent*>(_event)->getInternalObject(),&info);
    return true;
}

bool CVulkanCommandBuffer::resetEvent_impl(IEvent* const _event, const core::bitflag<stage_flags_t> stageMask)
{
    getFunctionTable().vkCmdResetEvent2(m_cmdbuf,static_cast<CVulkanEvent*>(_event)->getInternalObject(),getVkPipelineStageFlagsFromPipelineStageFlags(stageMask));
    return true;
}

bool CVulkanCommandBuffer::waitEvents_impl(const std::span<IEvent*> events, const SEventDependencyInfo* depInfos)
{
    const uint32_t eventCount = events.size();
    IGPUCommandPool::StackAllocation<VkEvent> vk_events(m_cmdpool,eventCount);
    IGPUCommandPool::StackAllocation<VkDependencyInfoKHR> infos(m_cmdpool,eventCount);
    if (!vk_events || !infos)
        return false;

    uint32_t memBarrierCount = 0u;
    uint32_t bufBarrierCount = 0u;
    uint32_t imgBarrierCount = 0u;
    for (auto i=0u; i<eventCount; i++)
    {
        memBarrierCount += depInfos[i].memBarriers.size();
        bufBarrierCount += depInfos[i].bufBarriers.size();
        imgBarrierCount += depInfos[i].imgBarriers.size();
    }
    IGPUCommandPool::StackAllocation<VkMemoryBarrier2KHR> memoryBarriers(m_cmdpool,memBarrierCount);
    IGPUCommandPool::StackAllocation<VkBufferMemoryBarrier2KHR> bufferBarriers(m_cmdpool,bufBarrierCount);
    IGPUCommandPool::StackAllocation<VkImageMemoryBarrier2KHR> imageBarriers(m_cmdpool,imgBarrierCount);
    if (!memoryBarriers || !bufferBarriers || !imageBarriers)
        return false;

    memBarrierCount = 0u;
    bufBarrierCount = 0u;
    imgBarrierCount = 0u;
    for (auto i=0u; i<eventCount; i++)
    {
        vk_events[i] = static_cast<CVulkanEvent*>(events[i])->getInternalObject();
        infos[i] = fill(memoryBarriers.data()+memBarrierCount,bufferBarriers.data()+bufBarrierCount,imageBarriers.data()+imgBarrierCount,depInfos[i]);
        memBarrierCount += infos[i].memoryBarrierCount;
        bufBarrierCount += infos[i].bufferMemoryBarrierCount;
        imgBarrierCount += infos[i].imageMemoryBarrierCount;
    }
    getFunctionTable().vkCmdWaitEvents2(m_cmdbuf,eventCount,vk_events.data(),infos.data());
    return true;
}

bool CVulkanCommandBuffer::pipelineBarrier_impl(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SPipelineBarrierDependencyInfo& depInfo)
{
    IGPUCommandPool::StackAllocation<VkMemoryBarrier2KHR> memoryBarriers(m_cmdpool,depInfo.memBarriers.size());
    IGPUCommandPool::StackAllocation<VkBufferMemoryBarrier2KHR> bufferBarriers(m_cmdpool,depInfo.bufBarriers.size());
    IGPUCommandPool::StackAllocation<VkImageMemoryBarrier2KHR> imageBarriers(m_cmdpool,depInfo.imgBarriers.size());
    if (!memoryBarriers || !bufferBarriers || !imageBarriers)
        return false;

    auto info = fill(memoryBarriers.data(),bufferBarriers.data(),imageBarriers.data(),depInfo,m_cmdpool->getQueueFamilyIndex());
    info.dependencyFlags = static_cast<VkDependencyFlagBits>(dependencyFlags.value);
    getFunctionTable().vkCmdPipelineBarrier2(m_cmdbuf,&info);
    return true;
}


bool CVulkanCommandBuffer::fillBuffer_impl(const asset::SBufferRange<IGPUBuffer>& range, const uint32_t data)
{
    getFunctionTable().vkCmdFillBuffer(m_cmdbuf,static_cast<const CVulkanBuffer*>(range.buffer.get())->getInternalObject(),range.offset,range.size,data);
    return true;
}

bool CVulkanCommandBuffer::updateBuffer_impl(const asset::SBufferRange<IGPUBuffer>& range, const void* const pData)
{
    getFunctionTable().vkCmdUpdateBuffer(m_cmdbuf,static_cast<const CVulkanBuffer*>(range.buffer.get())->getInternalObject(),range.offset,range.size,pData);
    return true;
}

bool CVulkanCommandBuffer::copyBuffer_impl(const IGPUBuffer* const srcBuffer, IGPUBuffer* const dstBuffer, const uint32_t regionCount, const video::IGPUCommandBuffer::SBufferCopy* const pRegions)
{
    IGPUCommandPool::StackAllocation<VkBufferCopy> vk_bufferCopyRegions(m_cmdpool,regionCount);
    if (!vk_bufferCopyRegions)
        return false;
    for (uint32_t i=0u; i<regionCount; ++i)
    {
        vk_bufferCopyRegions[i].srcOffset = pRegions[i].srcOffset;
        vk_bufferCopyRegions[i].dstOffset = pRegions[i].dstOffset;
        vk_bufferCopyRegions[i].size = pRegions[i].size;
    }

    const VkBuffer vk_srcBuffer = static_cast<const CVulkanBuffer*>(srcBuffer)->getInternalObject();
    const VkBuffer vk_dstBuffer = static_cast<const CVulkanBuffer*>(dstBuffer)->getInternalObject();
    getFunctionTable().vkCmdCopyBuffer(m_cmdbuf, vk_srcBuffer, vk_dstBuffer, regionCount, vk_bufferCopyRegions.data());
    return true;
}


bool CVulkanCommandBuffer::clearColorImage_impl(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearColorValue* const pColor, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges)
{
    IGPUCommandPool::StackAllocation<VkImageSubresourceRange> vk_ranges(m_cmdpool,rangeCount);
    if (!vk_ranges)
        return false;
    for (uint32_t i=0u; i<rangeCount; ++i)
        vk_ranges[i] = getVkImageSubresourceFrom(pRanges[i]);

    getFunctionTable().vkCmdClearColorImage(
        m_cmdbuf,static_cast<const CVulkanImage*>(image)->getInternalObject(),
        getVkImageLayoutFromImageLayout(imageLayout),reinterpret_cast<const VkClearColorValue*>(pColor),
        rangeCount,vk_ranges.data()
    );
    return true;
}

bool CVulkanCommandBuffer::clearDepthStencilImage_impl(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearDepthStencilValue* const pDepthStencil, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges)
{
    IGPUCommandPool::StackAllocation<VkImageSubresourceRange> vk_ranges(m_cmdpool,rangeCount);
    if (!vk_ranges)
        return false;
    for (uint32_t i=0u; i<rangeCount; ++i)
        vk_ranges[i] = getVkImageSubresourceFrom(pRanges[i]);

    getFunctionTable().vkCmdClearDepthStencilImage(
        m_cmdbuf,static_cast<const CVulkanImage*>(image)->getInternalObject(),
        getVkImageLayoutFromImageLayout(imageLayout),reinterpret_cast<const VkClearDepthStencilValue*>(pDepthStencil),
        rangeCount,vk_ranges.data()
    );
    return true;
}


static inline VkBufferImageCopy getVkBufferImageCopy(const IGPUImage::SBufferCopy& region)
{
    VkBufferImageCopy retval = {};
    retval.bufferOffset = region.bufferOffset;
    retval.bufferRowLength = region.bufferRowLength;
    retval.bufferImageHeight = region.bufferImageHeight;
    retval.imageSubresource = getVkImageSubresourceFrom(region.imageSubresource);
    // TODO: Make the regular old assignment operator work
    retval.imageOffset = { static_cast<int32_t>(region.imageOffset.x), static_cast<int32_t>(region.imageOffset.y), static_cast<int32_t>(region.imageOffset.z) };
    retval.imageExtent = { region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth };
    return retval;
}

bool CVulkanCommandBuffer::copyBufferToImage_impl(const IGPUBuffer* const srcBuffer, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions)
{
    IGPUCommandPool::StackAllocation<VkBufferImageCopy> vk_regions(m_cmdpool,regionCount);
    if (!vk_regions)
        return false;
    for (uint32_t i=0u; i<regionCount; ++i)
        vk_regions[i] = getVkBufferImageCopy(pRegions[i]);

    getFunctionTable().vkCmdCopyBufferToImage(m_cmdbuf,
        static_cast<const video::CVulkanBuffer*>(srcBuffer)->getInternalObject(),
        static_cast<const video::CVulkanImage*>(dstImage)->getInternalObject(),
        getVkImageLayoutFromImageLayout(dstImageLayout), regionCount, vk_regions.data()
    );

    return true;
}

bool CVulkanCommandBuffer::copyImageToBuffer_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, const IGPUBuffer* const dstBuffer, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions)
{
    IGPUCommandPool::StackAllocation<VkBufferImageCopy> vk_regions(m_cmdpool,regionCount);
    if (!vk_regions)
        return false;
    for (uint32_t i=0u; i<regionCount; ++i)
        vk_regions[i] = getVkBufferImageCopy(pRegions[i]);

    getFunctionTable().vkCmdCopyImageToBuffer(m_cmdbuf,
        static_cast<const video::CVulkanImage*>(srcImage)->getInternalObject(), getVkImageLayoutFromImageLayout(srcImageLayout),
        static_cast<const video::CVulkanBuffer*>(dstBuffer)->getInternalObject(),regionCount,vk_regions.data()
    );
    return true;
}

bool CVulkanCommandBuffer::copyImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SImageCopy* const pRegions)
{
    IGPUCommandPool::StackAllocation<VkImageCopy> vk_regions(m_cmdpool,regionCount);
    if (!vk_regions)
        return false;
    for (uint32_t i=0u; i<regionCount; ++i)
    {
        vk_regions[i].srcSubresource = getVkImageSubresourceFrom(pRegions[i].srcSubresource);
        vk_regions[i].srcOffset = { static_cast<int32_t>(pRegions[i].srcOffset.x), static_cast<int32_t>(pRegions[i].srcOffset.y), static_cast<int32_t>(pRegions[i].srcOffset.z) };
        vk_regions[i].dstSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].dstSubresource.aspectMask.value);
        vk_regions[i].dstSubresource = getVkImageSubresourceFrom(pRegions[i].dstSubresource);
        vk_regions[i].dstOffset = { static_cast<int32_t>(pRegions[i].dstOffset.x), static_cast<int32_t>(pRegions[i].dstOffset.y), static_cast<int32_t>(pRegions[i].dstOffset.z) };
        vk_regions[i].extent = { pRegions[i].extent.width, pRegions[i].extent.height, pRegions[i].extent.depth };
    }

    getFunctionTable().vkCmdCopyImage(m_cmdbuf,
        static_cast<const CVulkanImage*>(srcImage)->getInternalObject(),getVkImageLayoutFromImageLayout(srcImageLayout),
        static_cast<const CVulkanImage*>(dstImage)->getInternalObject(),getVkImageLayoutFromImageLayout(dstImageLayout),
        regionCount,vk_regions.data()
    );
    return true;
}


bool CVulkanCommandBuffer::copyAccelerationStructure_impl(const IGPUAccelerationStructure::CopyInfo& copyInfo)
{
    const auto info = getVkCopyAccelerationStructureInfoFrom(copyInfo);
    getFunctionTable().vkCmdCopyAccelerationStructureKHR(m_cmdbuf,&info);
    return true;
}
bool CVulkanCommandBuffer::copyAccelerationStructureToMemory_impl(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo)
{
    const auto info = getVkCopyAccelerationStructureToMemoryInfoFrom(copyInfo);
    getFunctionTable().vkCmdCopyAccelerationStructureToMemoryKHR(m_cmdbuf,&info);
    return true;
}

bool CVulkanCommandBuffer::copyAccelerationStructureFromMemory_impl(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo)
{
    const auto info = getVkCopyMemoryToAccelerationStructureInfoFrom(copyInfo);
    getFunctionTable().vkCmdCopyMemoryToAccelerationStructureKHR(m_cmdbuf,&info);
    return true;
}

bool CVulkanCommandBuffer::bindComputePipeline_impl(const IGPUComputePipeline* const pipeline)
{
    getFunctionTable().vkCmdBindPipeline(m_cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, static_cast<const CVulkanComputePipeline*>(pipeline)->getInternalObject());
    return true;
}

bool CVulkanCommandBuffer::bindGraphicsPipeline_impl(const IGPUGraphicsPipeline* const pipeline)
{
    getFunctionTable().vkCmdBindPipeline(m_cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, static_cast<const CVulkanGraphicsPipeline*>(pipeline)->getInternalObject());
    return true;
}

bool CVulkanCommandBuffer::bindDescriptorSets_impl(const asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const IGPUPipelineLayout* const layout, const uint32_t firstSet, const uint32_t descriptorSetCount, const IGPUDescriptorSet* const* const pDescriptorSets, const uint32_t dynamicOffsetCount, const uint32_t* const dynamicOffsets)
{
    VkDescriptorSet vk_descriptorSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT] = {};
    uint32_t dynamicOffsetCountPerSet[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT] = { 0u };

    // We allow null descriptor sets in our bind function to skip a certain set number we don't use
    // Will bind [first, last) with one call
    for (uint32_t i = 0u; i<descriptorSetCount; ++i)
    if (pDescriptorSets[i])
    {
        vk_descriptorSets[i] = static_cast<const CVulkanDescriptorSet*>(pDescriptorSets[i])->getInternalObject();
        // count dynamic offsets per set, if there are any
        if (dynamicOffsets)
        {
            dynamicOffsetCountPerSet[i] += pDescriptorSets[i]->getLayout()->getDescriptorRedirect(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC).getTotalCount();
            dynamicOffsetCountPerSet[i] += pDescriptorSets[i]->getLayout()->getDescriptorRedirect(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC).getTotalCount();
        }
    }

    uint32_t bindCallsCount = 0u;
    uint32_t dynamicOffsetsBindOffset = 0u;
    uint32_t first = ~0u;
    uint32_t last = ~0u;
    const VkPipelineLayout vk_pipelineLayout = static_cast<const CVulkanPipelineLayout*>(layout)->getInternalObject();
    for (uint32_t i=0u; i<descriptorSetCount; ++i)
    if (pDescriptorSets[i])
    {
        if (first==last)
            last = first = i;
        ++last;

        // Do a look ahead
        const auto next = i+1;
        if (next>=descriptorSetCount || !pDescriptorSets[next])
        {
            uint32_t dynamicOffsetCount = 0u;
            for (auto setIndex=first; setIndex<last; ++setIndex)
                dynamicOffsetCount += dynamicOffsetCountPerSet[setIndex];

            getFunctionTable().vkCmdBindDescriptorSets(
                m_cmdbuf,static_cast<VkPipelineBindPoint>(pipelineBindPoint),vk_pipelineLayout,
                firstSet+first, last-first, vk_descriptorSets+first,
                dynamicOffsetCount, dynamicOffsets+dynamicOffsetsBindOffset
            );
            dynamicOffsetsBindOffset += dynamicOffsetCount;

            first = last;
            ++bindCallsCount;
        }
    }
    // with K slots you need at most (K+1)/2 calls
    assert(bindCallsCount <= (IGPUPipelineLayout::DESCRIPTOR_SET_COUNT+1)/2);
    return true;
}

bool CVulkanCommandBuffer::pushConstants_impl(const IGPUPipelineLayout* const layout, const core::bitflag<IGPUShader::E_SHADER_STAGE> stageFlags, const uint32_t offset, const uint32_t size, const void* const pValues)
{
    getFunctionTable().vkCmdPushConstants(m_cmdbuf,static_cast<const CVulkanPipelineLayout*>(layout)->getInternalObject(),getVkShaderStageFlagsFromShaderStage(stageFlags),offset,size,pValues);
    return true;
}

bool CVulkanCommandBuffer::bindVertexBuffers_impl(const uint32_t firstBinding, const uint32_t bindingCount, const asset::SBufferBinding<const IGPUBuffer>* const pBindings)
{    
    VkBuffer vk_buffers[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
    VkDeviceSize vk_offsets[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];

    VkBuffer dummyBuffer = VK_NULL_HANDLE;
    for (uint32_t i=0u; i<bindingCount; ++i)
    {
        if (!pBindings[i].buffer)
        {
            vk_buffers[i] = VK_NULL_HANDLE;
            vk_offsets[i] = 0;
        }
        else
        {
            dummyBuffer = vk_buffers[i] = static_cast<const CVulkanBuffer*>(pBindings[i].buffer.get())->getInternalObject();
            vk_offsets[i] = pBindings[i].offset;
        }
    }
    // backfill all invalids
    for (uint32_t i=0u; i<bindingCount; ++i)
    if (vk_buffers[i]==VK_NULL_HANDLE)
        vk_buffers[i] = dummyBuffer;

    getFunctionTable().vkCmdBindVertexBuffers(m_cmdbuf,firstBinding,bindingCount,vk_buffers,vk_offsets);
    return true;
}

bool CVulkanCommandBuffer::bindIndexBuffer_impl(const asset::SBufferBinding<const IGPUBuffer>& binding, const asset::E_INDEX_TYPE indexType)
{
    assert(indexType<asset::EIT_UNKNOWN);
    getFunctionTable().vkCmdBindIndexBuffer(m_cmdbuf,static_cast<const CVulkanBuffer*>(binding.buffer.get())->getInternalObject(),binding.offset,static_cast<VkIndexType>(indexType));
    return true;
}


bool CVulkanCommandBuffer::setScissor_impl(const uint32_t first, const uint32_t count, const VkRect2D* const pScissors)
{
    getFunctionTable().vkCmdSetScissor(m_cmdbuf,first,count,pScissors);
    return true;
}

bool CVulkanCommandBuffer::setViewport_impl(const uint32_t first, const uint32_t count, const asset::SViewport* const pViewports)
{
    getFunctionTable().vkCmdSetViewport(m_cmdbuf,first,count,reinterpret_cast<const VkViewport*>(pViewports));
    return true;
}

bool CVulkanCommandBuffer::setLineWidth_impl(const float width)
{
    getFunctionTable().vkCmdSetLineWidth(m_cmdbuf,width);
    return true;
}

bool CVulkanCommandBuffer::setDepthBias_impl(const float depthBiasConstantFactor, const float depthBiasClamp, const float depthBiasSlopeFactor)
{
    getFunctionTable().vkCmdSetDepthBias(m_cmdbuf,depthBiasConstantFactor,depthBiasClamp,depthBiasSlopeFactor);
    return true;
}

bool CVulkanCommandBuffer::setBlendConstants_impl(const hlsl::float32_t4& constants)
{
    getFunctionTable().vkCmdSetBlendConstants(m_cmdbuf,reinterpret_cast<const float*>(&constants));
    return true;
}

bool CVulkanCommandBuffer::setDepthBounds_impl(const float minDepthBounds, const float maxDepthBounds)
{
    getFunctionTable().vkCmdSetDepthBounds(m_cmdbuf,minDepthBounds,maxDepthBounds);
    return true;
}

bool CVulkanCommandBuffer::setStencilCompareMask_impl(const asset::E_FACE_CULL_MODE faces, const uint8_t compareMask)
{
    getFunctionTable().vkCmdSetStencilCompareMask(m_cmdbuf,static_cast<VkStencilFaceFlags>(faces),compareMask);
    return true;
}

bool CVulkanCommandBuffer::setStencilWriteMask_impl(const asset::E_FACE_CULL_MODE faces, const uint8_t writeMask)
{
    getFunctionTable().vkCmdSetStencilWriteMask(m_cmdbuf,static_cast<VkStencilFaceFlags>(faces),writeMask);
    return true;
}

bool CVulkanCommandBuffer::setStencilReference_impl(const asset::E_FACE_CULL_MODE faces, const uint8_t reference)
{
    getFunctionTable().vkCmdSetStencilReference(m_cmdbuf,static_cast<VkStencilFaceFlags>(faces),reference);
    return true;
}


bool CVulkanCommandBuffer::resetQueryPool_impl(IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount)
{
    getFunctionTable().vkCmdResetQueryPool(m_cmdbuf, static_cast<CVulkanQueryPool*>(queryPool)->getInternalObject(), firstQuery, queryCount);
    return true;
}

bool CVulkanCommandBuffer::beginQuery_impl(IQueryPool* const queryPool, const uint32_t query, const core::bitflag<QUERY_CONTROL_FLAGS> flags)
{
    const auto vk_flags = CVulkanQueryPool::getVkQueryControlFlagsFrom(flags.value);
    getFunctionTable().vkCmdBeginQuery(m_cmdbuf, static_cast<CVulkanQueryPool*>(queryPool)->getInternalObject(), query, vk_flags);
    return true;
}

bool CVulkanCommandBuffer::endQuery_impl(IQueryPool* const queryPool, const uint32_t query)
{
    getFunctionTable().vkCmdEndQuery(m_cmdbuf, static_cast<CVulkanQueryPool*>(queryPool)->getInternalObject(), query);
    return true;
}

bool CVulkanCommandBuffer::writeTimestamp_impl(const asset::PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* const queryPool, const uint32_t query)
{
    getFunctionTable().vkCmdWriteTimestamp2(m_cmdbuf, getVkPipelineStageFlagsFromPipelineStageFlags(pipelineStage), static_cast<CVulkanQueryPool*>(queryPool)->getInternalObject(), query);
    return true;
}

bool CVulkanCommandBuffer::writeAccelerationStructureProperties_impl(const std::span<const IGPUAccelerationStructure* const> pAccelerationStructures, const IQueryPool::TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery)
{
    IGPUCommandPool::StackAllocation<VkAccelerationStructureKHR> vk_accelerationStructures(m_cmdpool,pAccelerationStructures.size());
    if (!vk_accelerationStructures)
        return false;
    for (size_t i=0; i<pAccelerationStructures.size(); ++i)
        vk_accelerationStructures[i] = *reinterpret_cast<const VkAccelerationStructureKHR*>(static_cast<const IGPUAccelerationStructure*>(pAccelerationStructures[i])->getNativeHandle());

    getFunctionTable().vkCmdWriteAccelerationStructuresPropertiesKHR(
        m_cmdbuf, vk_accelerationStructures.size(), vk_accelerationStructures.data(),
        CVulkanQueryPool::getVkQueryTypeFrom(queryType), static_cast<CVulkanQueryPool*>(queryPool)->getInternalObject(), firstQuery
    );
    return true;
}

bool CVulkanCommandBuffer::copyQueryPoolResults_impl(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, const asset::SBufferBinding<IGPUBuffer>& dstBuffer, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags)
{
    getFunctionTable().vkCmdCopyQueryPoolResults(
        m_cmdbuf, static_cast<const CVulkanQueryPool*>(queryPool)->getInternalObject(), firstQuery, queryCount,
        static_cast<CVulkanBuffer*>(dstBuffer.buffer.get())->getInternalObject(), dstBuffer.offset, stride,
        CVulkanQueryPool::getVkQueryResultsFlagsFrom(flags.value)
    );
    return true;
}


bool CVulkanCommandBuffer::dispatch_impl(const uint32_t groupCountX, const uint32_t groupCountY, const uint32_t groupCountZ)
{
    getFunctionTable().vkCmdDispatch(m_cmdbuf, groupCountX, groupCountY, groupCountZ);
    return true;
}

bool CVulkanCommandBuffer::dispatchIndirect_impl(const asset::SBufferBinding<const IGPUBuffer>& binding)
{
    getFunctionTable().vkCmdDispatchIndirect(m_cmdbuf,static_cast<const CVulkanBuffer*>(binding.buffer.get())->getInternalObject(),binding.offset);
    return true;
}


bool CVulkanCommandBuffer::beginRenderPass_impl(const SRenderpassBeginInfo& info, const SUBPASS_CONTENTS contents)
{
    const auto depthStencilAttachmentCount = info.renderpass->getDepthStencilAttachmentCount();
    const auto colorAttachmentCount = info.renderpass->getColorAttachmentCount();
    IGPUCommandPool::StackAllocation<VkClearValue> vk_clearValues(m_cmdpool,depthStencilAttachmentCount+colorAttachmentCount);
    if (!vk_clearValues)
        return false;

    // We can just speculatively copy, its probably more performant in most circumstances.
    // We just check the pointers so you can use nullptr if your renderpass won't clear any attachment
    if (info.depthStencilClearValues)
    for (auto i=0u; i<depthStencilAttachmentCount; i++)
    //if (renderpass->getCreationParameters().depthStencilAttachments[i].loadOp.stencil==IGPURenderpass::LOAD_OP::CLEAR) or depth
    {
        vk_clearValues[i].depthStencil.depth = info.depthStencilClearValues[i].depth;
        vk_clearValues[i].depthStencil.stencil = info.depthStencilClearValues[i].stencil;
    }
    if (info.colorClearValues)
    for (auto i=0u; i<colorAttachmentCount; i++)
    //if (renderpass->getCreationParameters().colorAttachments[i].loadOp==IGPURenderpass::LOAD_OP::CLEAR)
        std::copy_n(info.colorClearValues[i].uint32,4,vk_clearValues[i+depthStencilAttachmentCount].color.uint32);

    const VkRenderPassBeginInfo vk_beginInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr, // DeviceGroupRpassBeginInfo, SampleLocations?
        .renderPass = static_cast<const CVulkanRenderpass*>(info.renderpass)->getInternalObject(),
        .framebuffer = static_cast<const CVulkanFramebuffer*>(info.framebuffer)->getInternalObject(),
        .renderArea = info.renderArea,
        // Implicitly but could be optimizedif needed
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRenderPassBeginInfo.html#VUID-VkRenderPassBeginInfo-clearValueCount-00902
        .clearValueCount = vk_clearValues.size(),
        // Implicit
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRenderPassBeginInfo.html#VUID-VkRenderPassBeginInfo-clearValueCount-04962
        .pClearValues = vk_clearValues.data()
    };

    VkSubpassBeginInfo vk_subpassBeginInfo = {VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO,nullptr};
    vk_subpassBeginInfo.contents = static_cast<VkSubpassContents>(contents);

    // Implicitly satisfied by our sane API:
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRenderPassBeginInfo.html#VUID-VkRenderPassBeginInfo-renderPass-00904
    getFunctionTable().vkCmdBeginRenderPass2(m_cmdbuf,&vk_beginInfo,&vk_subpassBeginInfo);
    return true;
}

bool CVulkanCommandBuffer::nextSubpass_impl(const SUBPASS_CONTENTS contents)
{
    VkSubpassBeginInfo vk_beginInfo = {VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO,nullptr};
    vk_beginInfo.contents = static_cast<VkSubpassContents>(contents);
    const VkSubpassEndInfo vk_endInfo = {VK_STRUCTURE_TYPE_SUBPASS_END_INFO,nullptr};
    getFunctionTable().vkCmdNextSubpass2(m_cmdbuf,&vk_beginInfo,&vk_endInfo);
    return true;
}

bool CVulkanCommandBuffer::endRenderPass_impl()
{
    const VkSubpassEndInfo vk_subpassEndInfo = {VK_STRUCTURE_TYPE_SUBPASS_END_INFO,nullptr};
    getFunctionTable().vkCmdEndRenderPass2(m_cmdbuf,&vk_subpassEndInfo);
    return true;
}

bool CVulkanCommandBuffer::clearAttachments_impl(const SClearAttachments& info)
{
    constexpr auto MaxClears = 1u+IGPURenderpass::SCreationParams::SSubpassDescription::MaxColorAttachments;
    VkClearAttachment vk_clearAttachments[MaxClears];

    auto outAttachment = vk_clearAttachments;
    if (info.clearDepth || info.clearStencil)
    {
        outAttachment->aspectMask = (info.clearDepth ? VK_IMAGE_ASPECT_DEPTH_BIT:0)|(info.clearStencil ? VK_IMAGE_ASPECT_STENCIL_BIT:0);
        outAttachment->colorAttachment = 0;
        outAttachment->clearValue.depthStencil.depth = info.depthStencilValue.depth;
        outAttachment->clearValue.depthStencil.stencil = info.depthStencilValue.stencil;
        outAttachment++;
    }
    for (auto i=0; i<IGPURenderpass::SCreationParams::SSubpassDescription::MaxColorAttachments; i++)
    if (info.clearColor(i))
    {
        outAttachment->aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        outAttachment->colorAttachment = i;
        std::copy_n(info.colorValues[i].uint32,4,outAttachment->clearValue.color.uint32);
        outAttachment++;
    }
    const auto attachmentCount = std::distance(vk_clearAttachments,outAttachment);

    getFunctionTable().vkCmdClearAttachments(m_cmdbuf,attachmentCount,vk_clearAttachments,info.regions.size(),reinterpret_cast<const VkClearRect*>(info.regions.data()));
    return true;
}

bool CVulkanCommandBuffer::draw_impl(const uint32_t vertexCount, const uint32_t instanceCount, const uint32_t firstVertex, const uint32_t firstInstance)
{
    getFunctionTable().vkCmdDraw(m_cmdbuf, vertexCount, instanceCount, firstVertex, firstInstance);
    return true;
}

bool CVulkanCommandBuffer::drawIndexed_impl(const uint32_t indexCount, const uint32_t instanceCount, const uint32_t firstIndex, const int32_t vertexOffset, const uint32_t firstInstance)
{
    getFunctionTable().vkCmdDrawIndexed(m_cmdbuf, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    return true;
}

bool CVulkanCommandBuffer::drawIndirect_impl(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride)
{
    getFunctionTable().vkCmdDrawIndirect(m_cmdbuf,static_cast<const CVulkanBuffer*>(binding.buffer.get())->getInternalObject(),binding.offset,drawCount,stride);
    return true;
}

bool CVulkanCommandBuffer::drawIndexedIndirect_impl(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride)
{
    getFunctionTable().vkCmdDrawIndexedIndirect(m_cmdbuf,static_cast<const CVulkanBuffer*>(binding.buffer.get())->getInternalObject(),binding.offset,drawCount,stride);
    return true;
}

bool CVulkanCommandBuffer::drawIndirectCount_impl(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride)
{
    getFunctionTable().vkCmdDrawIndirectCount(
        m_cmdbuf,static_cast<const CVulkanBuffer*>(indirectBinding.buffer.get())->getInternalObject(),indirectBinding.offset,
        static_cast<const CVulkanBuffer*>(countBinding.buffer.get())->getInternalObject(),countBinding.offset,maxDrawCount,stride
    );
    return true;
}

bool CVulkanCommandBuffer::drawIndexedIndirectCount_impl(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride)
{
    getFunctionTable().vkCmdDrawIndexedIndirectCount(
        m_cmdbuf,static_cast<const CVulkanBuffer*>(indirectBinding.buffer.get())->getInternalObject(),indirectBinding.offset,
        static_cast<const CVulkanBuffer*>(countBinding.buffer.get())->getInternalObject(),countBinding.offset,maxDrawCount,stride
    );
    return true;
}

bool CVulkanCommandBuffer::blitImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const std::span<const SImageBlit> regions, const IGPUSampler::E_TEXTURE_FILTER filter)
{
    VkImage vk_srcImage = static_cast<const CVulkanImage*>(srcImage)->getInternalObject();
    VkImage vk_dstImage = static_cast<const CVulkanImage*>(dstImage)->getInternalObject();

    core::vector<VkImageBlit> vk_blitRegions(regions.size());
    auto outRegionIt = vk_blitRegions.data();
    for (auto region : regions)
    {
        outRegionIt->srcSubresource.aspectMask = static_cast<VkImageAspectFlags>(region.aspectMask);
        outRegionIt->srcSubresource.mipLevel = region.srcMipLevel;
        outRegionIt->srcSubresource.baseArrayLayer = region.srcBaseLayer;
        outRegionIt->srcSubresource.layerCount = region.layerCount;

        memcpy(outRegionIt->srcOffsets,&region.srcMinCoord,sizeof(VkOffset3D)*2);

        outRegionIt->dstSubresource.aspectMask = static_cast<VkImageAspectFlags>(region.aspectMask);
        outRegionIt->dstSubresource.mipLevel = region.dstMipLevel;
        outRegionIt->dstSubresource.baseArrayLayer = region.dstBaseLayer;
        outRegionIt->dstSubresource.layerCount = region.layerCount;
        
        memcpy(outRegionIt->dstOffsets,&region.dstMinCoord,sizeof(VkOffset3D)*2);
        outRegionIt++;
    }

    getFunctionTable().vkCmdBlitImage(m_cmdbuf,vk_srcImage,getVkImageLayoutFromImageLayout(srcImageLayout),vk_dstImage,getVkImageLayoutFromImageLayout(dstImageLayout),regions.size(),vk_blitRegions.data(),static_cast<VkFilter>(filter));

    return true;
}

bool CVulkanCommandBuffer::resolveImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageResolve* pRegions)
{
    constexpr uint32_t MAX_COUNT = (1u << 12) / sizeof(VkImageResolve);
    assert(regionCount <= MAX_COUNT);

    VkImageResolve vk_regions[MAX_COUNT];
    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        vk_regions[i].srcSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].srcSubresource.aspectMask.value);
        vk_regions[i].srcSubresource.baseArrayLayer = pRegions[i].srcSubresource.baseArrayLayer;
        vk_regions[i].srcSubresource.layerCount = pRegions[i].srcSubresource.layerCount;
        vk_regions[i].srcSubresource.mipLevel = pRegions[i].srcSubresource.mipLevel;

        vk_regions[i].srcOffset = { static_cast<int32_t>(pRegions[i].srcOffset.x), static_cast<int32_t>(pRegions[i].srcOffset.y), static_cast<int32_t>(pRegions[i].srcOffset.z) };

        vk_regions[i].dstSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].dstSubresource.aspectMask.value);
        vk_regions[i].dstSubresource.baseArrayLayer = pRegions[i].dstSubresource.baseArrayLayer;
        vk_regions[i].dstSubresource.layerCount = pRegions[i].dstSubresource.layerCount;
        vk_regions[i].dstSubresource.mipLevel = pRegions[i].dstSubresource.mipLevel;

        vk_regions[i].dstOffset = { static_cast<int32_t>(pRegions[i].dstOffset.x), static_cast<int32_t>(pRegions[i].dstOffset.y), static_cast<int32_t>(pRegions[i].dstOffset.z) };

        vk_regions[i].extent = { pRegions[i].extent.width, pRegions[i].extent.height, pRegions[i].extent.depth };
    }

    getFunctionTable().vkCmdResolveImage(
        m_cmdbuf,
        static_cast<const CVulkanImage*>(srcImage)->getInternalObject(),
        getVkImageLayoutFromImageLayout(srcImageLayout),
        static_cast<const CVulkanImage*>(dstImage)->getInternalObject(),
        getVkImageLayoutFromImageLayout(dstImageLayout),
        regionCount,
        vk_regions);

    return true;
}

bool CVulkanCommandBuffer::executeCommands_impl(const uint32_t count, IGPUCommandBuffer* const* const cmdbufs)
{
    IGPUCommandPool::StackAllocation<VkCommandBuffer> vk_commandBuffers(m_cmdpool,count);
    if (!vk_commandBuffers)
        return false;
    for (uint32_t i=0u; i<count; ++i)
        vk_commandBuffers[i] = static_cast<const CVulkanCommandBuffer*>(cmdbufs[i])->getInternalObject();

    getFunctionTable().vkCmdExecuteCommands(m_cmdbuf, count, vk_commandBuffers.data());
    return true;
}