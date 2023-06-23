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
        vk_inheritanceInfo.renderPass = static_cast<const CVulkanRenderpass*>(inheritanceInfo->renderpass.get())->getInternalObject();
        vk_inheritanceInfo.subpass = inheritanceInfo->subpass;
        // From the spec:
        // Specifying the exact framebuffer that the secondary command buffer will be
        // executed with may result in better performance at command buffer execution time.
        if (inheritanceInfo->framebuffer)
            vk_inheritanceInfo.framebuffer = static_cast<const CVulkanFramebuffer*>(inheritanceInfo->framebuffer.get())->getInternalObject();
        vk_inheritanceInfo.occlusionQueryEnable = inheritanceInfo->occlusionQueryEnable;
        vk_inheritanceInfo.queryFlags = static_cast<VkQueryControlFlags>(inheritanceInfo->queryFlags.value);
        vk_inheritanceInfo.pipelineStatistics = static_cast<VkQueryPipelineStatisticFlags>(0u); // must be 0

        beginInfo.pInheritanceInfo = &vk_inheritanceInfo;
    }

    return getFunctionTable().vkBeginCommandBuffer(m_cmdbuf,&beginInfo)==VK_SUCCESS;
}


template<typename vk_barrier_t, typename ResourceBarrier>
void fill(vk_barrier_t& out, const ResourceBarrier& in, const uint32_t selfQueueFamilyIndex)
{
    out.srcQueueFamilyIndex = selfQueueFamilyIndex;
    out.dstQueueFamilyIndex = selfQueueFamilyIndex;
    const asset::SMemoryBarrier* memoryBarrier;
    if constexpr (std::is_same_v<IGPUCommandBuffer::SOwnershipTransferBarrier,ResourceBarrier>)
    {
        memoryBarrier = &in.dep;
        if (in.otherQueueFamilyIndex!=IQueue::FamilyIgnored)
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
template<typename ResourceBarrier>
VkDependencyInfoKHR fill(
    VkMemoryBarrier2KHR* memoryBarriers, VkBufferMemoryBarrier2KHR* bufferBarriers, VkImageMemoryBarrier2KHR* imageBarriers,
    const IGPUCommandBuffer::SDependencyInfo<ResourceBarrier>& depInfo, const uint32_t selfQueueFamilyIndex=IQueue::FamilyIgnored
) {
    VkDependencyInfoKHR info = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,nullptr };
    for (auto i=0; i<depInfo.memBarrierCount; i++)
    {
        auto& out = memoryBarriers[i];
        out.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
        out.pNext = nullptr;
        fill(out,depInfo.memBarriers[i],selfQueueFamilyIndex);
    }
    for (auto i=0; i<depInfo.bufBarrierCount; i++)
    {
        auto& out = bufferBarriers[i];
        out.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
        out.pNext = nullptr;

        const auto& in = depInfo.bufBarriers[i];
        fill(out,in.barrier,selfQueueFamilyIndex);
        out.buffer = static_cast<const CVulkanBuffer*>(in.range.buffer.get())->getInternalObject();
        out.offset = in.range.offset;
        out.size = in.range.size;
    }
    for (auto i=0; i<depInfo.imgBarrierCount; i++)
    {
        auto& out = imageBarriers[i];
        out.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR;
        out.pNext = nullptr;

        const auto& in = depInfo.imgBarriers[i];
        out.oldLayout = getVkImageLayoutFromImageLayout(in.oldLayout);
        out.newLayout = getVkImageLayoutFromImageLayout(in.newLayout);
        fill(out,in.barrier,selfQueueFamilyIndex);
        out.image = static_cast<const CVulkanImage*>(in.image)->getInternalObject();
        out.subresourceRange = in.subresourceRange;
    }
    info.dependencyFlags = 0u;
    info.memoryBarrierCount = depInfo.memBarrierCount;
    info.pMemoryBarriers = memoryBarriers;
    info.bufferMemoryBarrierCount = depInfo.bufBarrierCount;
    info.pBufferMemoryBarriers = bufferBarriers;
    info.imageMemoryBarrierCount = depInfo.imgBarrierCount;
    info.pImageMemoryBarriers = imageBarriers;
    return info;
}

bool CVulkanCommandBuffer::setEvent_impl(IEvent* const _event, const SEventDependencyInfo& depInfo)
{
    IGPUCommandPool::StackAllocation<VkMemoryBarrier2KHR> memoryBarriers(m_cmdpool,depInfo.memBarrierCount);
    IGPUCommandPool::StackAllocation<VkBufferMemoryBarrier2KHR> bufferBarriers(m_cmdpool,depInfo.bufBarrierCount);
    IGPUCommandPool::StackAllocation<VkImageMemoryBarrier2KHR> imageBarriers(m_cmdpool,depInfo.imgBarrierCount);
    if (!memoryBarriers || !bufferBarriers || !imageBarriers)
        return false;

    auto info = fill(memoryBarriers.data(),bufferBarriers.data(),imageBarriers.data(),depInfo);
    getFunctionTable().vkCmdSetEvent2KHR(m_cmdbuf,static_cast<CVulkanEvent*>(_event)->getInternalObject(),&info);
    return true;
}

bool CVulkanCommandBuffer::resetEvent_impl(IEvent* const _event, const core::bitflag<stage_flags_t> stageMask)
{
    getFunctionTable().vkCmdResetEvent2KHR(m_cmdbuf,static_cast<CVulkanEvent*>(_event)->getInternalObject(),getVkPipelineStageFlagsFromPipelineStageFlags(stageMask));
    return true;
}

bool CVulkanCommandBuffer::waitEvents_impl(const uint32_t eventCount, IEvent* const* const pEvents, const SEventDependencyInfo* depInfos)
{
    IGPUCommandPool::StackAllocation<VkEvent> events(m_cmdpool,eventCount);
    IGPUCommandPool::StackAllocation<VkDependencyInfoKHR> infos(m_cmdpool,eventCount);
    if (!events || !infos)
        return false;

    uint32_t memBarrierCount = 0u;
    uint32_t bufBarrierCount = 0u;
    uint32_t imgBarrierCount = 0u;
    for (auto i=0u; i<eventCount; i++)
    {
        memBarrierCount += depInfos[i].memBarrierCount;
        bufBarrierCount += depInfos[i].bufBarrierCount;
        imgBarrierCount += depInfos[i].imgBarrierCount;
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
        events[i] = static_cast<CVulkanEvent*>(pEvents[i])->getInternalObject();
        infos[i] = fill(memoryBarriers.data()+memBarrierCount,bufferBarriers.data()+bufBarrierCount,imageBarriers.data()+imgBarrierCount,depInfos[i]);
        memBarrierCount += infos[i].memoryBarrierCount;
        bufBarrierCount += infos[i].bufferMemoryBarrierCount;
        imgBarrierCount += infos[i].imageMemoryBarrierCount;
    }
    getFunctionTable().vkCmdWaitEvents2KHR(m_cmdbuf,eventCount,events.data(),infos.data());
    return true;
}

bool CVulkanCommandBuffer::pipelineBarrier_impl(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SPipelineBarrierDependencyInfo& depInfo)
{
    IGPUCommandPool::StackAllocation<VkMemoryBarrier2KHR> memoryBarriers(m_cmdpool,depInfo.memBarrierCount);
    IGPUCommandPool::StackAllocation<VkBufferMemoryBarrier2KHR> bufferBarriers(m_cmdpool,depInfo.bufBarrierCount);
    IGPUCommandPool::StackAllocation<VkImageMemoryBarrier2KHR> imageBarriers(m_cmdpool,depInfo.imgBarrierCount);
    if (!memoryBarriers || !bufferBarriers || !imageBarriers)
        return false;

    auto info = fill(memoryBarriers.data(),bufferBarriers.data(),imageBarriers.data(),depInfo,m_cmdpool->getQueueFamilyIndex());
    info.dependencyFlags = static_cast<VkDependencyFlagBits>(dependencyFlags.value);
    getFunctionTable().vkCmdPipelineBarrier2KHR(m_cmdbuf,&info);
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


template<typename SubresourceRange>
static inline auto getVkImageSubresourceFrom(const SubresourceRange& range) -> std::conditional_t<std::is_same_v<SubresourceRange,IGPUImage::SSubresourceRange>,VkImageSubresourceRange,VkImageSubresourceLayers>
{
    constexpr bool rangeNotLayers =  std::is_same_v<SubresourceRange,IGPUImage::SSubresourceRange>;

    std::conditional_t<rangeNotLayers,VkImageSubresourceRange,VkImageSubresourceLayers> retval = {};
    retval.aspectMask = static_cast<VkImageAspectFlags>(range.aspectMask.value);
    if constexpr
        retval.baseMipLevel = range.baseMipLevel;
    retval.levelCount = range.layerCount;
    retval.baseArrayLayer = range.baseArrayLayer;
    retval.layerCount = range.layerCount;
    return retval;
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
    VkCopyAccelerationStructureInfoKHR info = { VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR, nullptr };
    info.mode = CVulkanAccelerationStructure::getVkCopyAccelerationStructureModeFrom(copyInfo.copyMode);
    info.src = static_cast<const CVulkanAccelerationStructure*>(copyInfo.src)->getInternalObject();
    info.dst = static_cast<CVulkanAccelerationStructure*>(copyInfo.dst)->getInternalObject();
    getFunctionTable().vkCmdCopyAccelerationStructureKHR(m_cmdbuf,&info);
    return true;
}
#if 0
bool CVulkanCommandBuffer::copyAccelerationStructureToMemory_impl(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo)
{
    VkCopyAccelerationStructureToMemoryInfoKHR info = CVulkanAccelerationStructure::getVkASCopyToMemoryInfo(vk_device,copyInfo);
    getFunctionTable().vkCmdCopyAccelerationStructureToMemoryKHR(m_cmdbuf, &info);
    return true;
}

bool CVulkanCommandBuffer::copyAccelerationStructureFromMemory_impl(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo)
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    VkDevice vk_device = vulkanDevice->getInternalObject();
    auto* vk = vulkanDevice->getFunctionTable();

    VkCopyMemoryToAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyFromMemoryInfo(vk_device, vk, copyInfo);
    getFunctionTable().vkCmdCopyMemoryToAccelerationStructureKHR(m_cmdbuf, &info);
    return true;
}


bool CVulkanCommandBuffer::buildAccelerationStructures_impl(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const video::IGPUAccelerationStructure::BuildRangeInfo* const* const ppBuildRangeInfos)
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    VkDevice vk_device = vulkanDevice->getInternalObject();
    auto* vk = vulkanDevice->getFunctionTable();

    static constexpr size_t MaxGeometryPerBuildInfoCount = 64;
    static constexpr size_t MaxBuildInfoCount = 128;
    size_t infoCount = pInfos.size();
    assert(infoCount <= MaxBuildInfoCount);

    // TODO: Use better container when ready for these stack allocated memories.
    VkAccelerationStructureBuildGeometryInfoKHR vk_buildGeomsInfos[MaxBuildInfoCount] = {};

    uint32_t geometryArrayOffset = 0u;
    VkAccelerationStructureGeometryKHR vk_geometries[MaxGeometryPerBuildInfoCount * MaxBuildInfoCount] = {};

    auto* infos = pInfos.begin();
    
    for (uint32_t i = 0; i < infoCount; ++i)
    {
        uint32_t geomCount = infos[i].geometries.size();
        vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(vk_device, vk, infos[i], &vk_geometries[geometryArrayOffset]);
        geometryArrayOffset += geomCount;
    }

    static_assert(sizeof(IGPUAccelerationStructure::BuildRangeInfo) == sizeof(VkAccelerationStructureBuildRangeInfoKHR));
    auto buildRangeInfos = reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR* const*>(ppBuildRangeInfos);
    getFunctionTable().vkCmdBuildAccelerationStructuresKHR(m_cmdbuf, infoCount, vk_buildGeomsInfos, buildRangeInfos);

    return true;
}

bool CVulkanCommandBuffer::buildAccelerationStructuresIndirect_impl(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<const IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* const pIndirectStrides, const uint32_t* const* const ppMaxPrimitiveCounts)
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    VkDevice vk_device = vulkanDevice->getInternalObject();
    auto* vk = vulkanDevice->getFunctionTable();

    static constexpr size_t MaxGeometryPerBuildInfoCount = 64;
    static constexpr size_t MaxBuildInfoCount = 128;

    size_t infoCount = pInfos.size();
    size_t indirectDeviceAddressesCount = pIndirectDeviceAddresses.size();
    assert(infoCount <= MaxBuildInfoCount);
    assert(infoCount == indirectDeviceAddressesCount);

    // TODO: Use better container when ready for these stack allocated memories.
    VkAccelerationStructureBuildGeometryInfoKHR vk_buildGeomsInfos[MaxBuildInfoCount] = {};
    VkDeviceSize vk_indirectDeviceAddresses[MaxBuildInfoCount] = {};

    uint32_t geometryArrayOffset = 0u;
    VkAccelerationStructureGeometryKHR vk_geometries[MaxGeometryPerBuildInfoCount * MaxBuildInfoCount] = {};

    auto* infos = pInfos.begin();
    auto* indirectDeviceAddresses = pIndirectDeviceAddresses.begin();

    for (uint32_t i = 0; i < infoCount; ++i)
    {
        uint32_t geomCount = infos[i].geometries.size();

        vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(vk_device, vk, infos[i], &vk_geometries[geometryArrayOffset]);
        geometryArrayOffset += geomCount;

        auto addr = CVulkanAccelerationStructure::getVkDeviceOrHostAddress<IGPUAccelerationStructure::DeviceAddressType>(vk_device, vk, indirectDeviceAddresses[i]);
        vk_indirectDeviceAddresses[i] = addr.deviceAddress;
    }

    getFunctionTable().vkCmdBuildAccelerationStructuresIndirectKHR(m_cmdbuf, infoCount, vk_buildGeomsInfos, vk_indirectDeviceAddresses, pIndirectStrides, ppMaxPrimitiveCounts);

    return true;
}
#endif

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

bool CVulkanCommandBuffer::bindDescriptorSets_impl(const asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const IGPUPipelineLayout* const layout, const uint32_t firstSet, const uint32_t descriptorSetCount, const IGPUDescriptorSet* const* const pDescriptorSets, const uint32_t dynamicOffsetCount = 0u, const uint32_t* const dynamicOffsets)
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
    assert(bindCallsCount < (IGPUPipelineLayout::DESCRIPTOR_SET_COUNT-1)/2);
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

bool CVulkanCommandBuffer::resetQueryPool_impl(IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount)
{
    getFunctionTable().vkCmdResetQueryPool(m_cmdbuf, static_cast<CVulkanQueryPool*>(queryPool)->getInternalObject(), firstQuery, queryCount);
    return true;
}

bool CVulkanCommandBuffer::beginQuery_impl(IQueryPool* const queryPool, const uint32_t query, const core::bitflag<QUERY_CONTROL_FLAGS> flags = QUERY_CONTROL_FLAGS::NONE)
{
    const auto vk_flags = CVulkanQueryPool::getVkQueryControlFlagsFromQueryControlFlags(flags.value);
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
    getFunctionTable().vkCmdWriteTimestamp2KHR(m_cmdbuf, getVkPipelineStageFlagsFromPipelineStageFlags(pipelineStage), static_cast<CVulkanQueryPool*>(queryPool)->getInternalObject(), query);
    return true;
}

bool CVulkanCommandBuffer::writeAccelerationStructureProperties_impl(const core::SRange<const IGPUAccelerationStructure*>& pAccelerationStructures, const IQueryPool::TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery)
{
    IGPUCommandPool::StackAllocation<VkAccelerationStructureKHR> vk_accelerationStructures(m_cmdpool,pAccelerationStructures.size());
    if (!vk_accelerationStructures)
        return false;
    for (size_t i=0; i<pAccelerationStructures.size(); ++i)
        vk_accelerationStructures[i] = static_cast<const CVulkanAccelerationStructure*>(pAccelerationStructures[i])->getInternalObject();

    getFunctionTable().vkCmdWriteAccelerationStructuresPropertiesKHR(
        m_cmdbuf, vk_accelerationStructures.size(), vk_accelerationStructures.data(),
        CVulkanQueryPool::getVkQueryTypeFromQueryType(queryType), static_cast<CVulkanQueryPool*>(queryPool)->getInternalObject(), firstQuery
    );
    return true;
}

bool CVulkanCommandBuffer::copyQueryPoolResults_impl(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, const asset::SBufferBinding<IGPUBuffer>& dstBuffer, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags)
{
    getFunctionTable().vkCmdCopyQueryPoolResults(
        m_cmdbuf, static_cast<const CVulkanQueryPool*>(queryPool)->getInternalObject(), firstQuery, queryCount,
        static_cast<CVulkanBuffer*>(dstBuffer.buffer.get())->getInternalObject(), dstBuffer.offset, stride,
        CVulkanQueryPool::getVkQueryResultsFlagsFromQueryResultsFlags(flags.value)
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

#if 0
bool CVulkanCommandBuffer::beginRenderPass_impl(const SRenderpassBeginInfo& info, SUBPASS_CONTENTS contents)
{
    constexpr uint32_t MAX_CLEAR_VALUE_COUNT = (1 << 12ull) / sizeof(VkClearValue);
    VkClearValue vk_clearValues[MAX_CLEAR_VALUE_COUNT];

    assert(pRenderPassBegin.clearValueCount <= MAX_CLEAR_VALUE_COUNT);

    for (uint32_t i = 0u; i < pRenderPassBegin->clearValueCount; ++i)
    {
        for (uint32_t k = 0u; k < 4u; ++k)
            vk_clearValues[i].color.uint32[k] = pRenderPassBegin->clearValues[i].color.uint32[k];

        vk_clearValues[i].depthStencil.depth = pRenderPassBegin->clearValues[i].depthStencil.depth;
        vk_clearValues[i].depthStencil.stencil = pRenderPassBegin->clearValues[i].depthStencil.stencil;
    }

    VkRenderPassBeginInfo vk_beginInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    vk_beginInfo.pNext = nullptr;
    vk_beginInfo.renderPass = static_cast<const CVulkanRenderpass*>(pRenderPassBegin->renderpass.get())->getInternalObject();
    vk_beginInfo.framebuffer = static_cast<const CVulkanFramebuffer*>(pRenderPassBegin->framebuffer.get())->getInternalObject();
    vk_beginInfo.renderArea = pRenderPassBegin->renderArea;
    vk_beginInfo.clearValueCount = pRenderPassBegin->clearValueCount;
    vk_beginInfo.pClearValues = vk_clearValues;

    VkSubpassBeginInfo vk_subpassBeginInfo = { VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO, nullptr };
    vk_subpassBeginInfo.contents = static_cast<VkSubpassContents>(contents);

    getFunctionTable().vkCmdBeginRenderPass2(m_cmdbuf, &vk_beginInfo, &vk_subpassBeginInfo);

    return true;
}

bool CVulkanCommandBuffer::nextSubpass_impl(const SUBPASS_CONTENTS contents)
{

}

bool CVulkanCommandBuffer::endRenderPass_impl()
{

}

bool CVulkanCommandBuffer::clearAttachments_impl(const SClearAttachments& info)
{
    /*
    constexpr uint32_t MAX_ATTACHMENT_COUNT = 8u;
    assert(attachmentCount <= MAX_ATTACHMENT_COUNT);
    VkClearAttachment vk_clearAttachments[MAX_ATTACHMENT_COUNT];

    constexpr uint32_t MAX_REGION_PER_ATTACHMENT_COUNT = ((1u << 12) - sizeof(vk_clearAttachments)) / sizeof(VkClearRect);
    assert(rectCount <= MAX_REGION_PER_ATTACHMENT_COUNT);
    VkClearRect vk_clearRects[MAX_REGION_PER_ATTACHMENT_COUNT];

    for (uint32_t i = 0u; i < attachmentCount; ++i)
    {
        vk_clearAttachments[i].aspectMask = static_cast<VkImageAspectFlags>(pAttachments[i].aspectMask);
        vk_clearAttachments[i].colorAttachment = pAttachments[i].colorAttachment;

        auto& vk_clearValue = vk_clearAttachments[i].clearValue;
        const auto& clearValue = pAttachments[i].clearValue;

        for (uint32_t k = 0u; k < 4u; ++k)
            vk_clearValue.color.uint32[k] = clearValue.color.uint32[k];

        vk_clearValue.depthStencil.depth = clearValue.depthStencil.depth;
        vk_clearValue.depthStencil.stencil = clearValue.depthStencil.stencil;
    }

    for (uint32_t i = 0u; i < rectCount; ++i)
    {
        vk_clearRects[i].rect = pRects[i].rect;
        vk_clearRects[i].baseArrayLayer = pRects[i].baseArrayLayer;
        vk_clearRects[i].layerCount = pRects[i].layerCount;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    getFunctionTable().vkCmdClearAttachments(
        m_cmdbuf,
        attachmentCount,
        vk_clearAttachments,
        rectCount,
        vk_clearRects);

    return true;
    */
}
#endif

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

bool CVulkanCommandBuffer::blitImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageBlit* pRegions, const IGPUSampler::E_TEXTURE_FILTER filter)
{
    VkImage vk_srcImage = static_cast<const CVulkanImage*>(srcImage)->getInternalObject();
    VkImage vk_dstImage = static_cast<const CVulkanImage*>(dstImage)->getInternalObject();

    constexpr uint32_t MAX_BLIT_REGION_COUNT = 100u;
    VkImageBlit vk_blitRegions[MAX_BLIT_REGION_COUNT];
    assert(regionCount <= MAX_BLIT_REGION_COUNT);

    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        vk_blitRegions[i].srcSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].srcSubresource.aspectMask.value);
        vk_blitRegions[i].srcSubresource.mipLevel = pRegions[i].srcSubresource.mipLevel;
        vk_blitRegions[i].srcSubresource.baseArrayLayer = pRegions[i].srcSubresource.baseArrayLayer;
        vk_blitRegions[i].srcSubresource.layerCount = pRegions[i].srcSubresource.layerCount;

        // Todo(achal): Remove `static_cast`s
        vk_blitRegions[i].srcOffsets[0] = { static_cast<int32_t>(pRegions[i].srcOffsets[0].x), static_cast<int32_t>(pRegions[i].srcOffsets[0].y), static_cast<int32_t>(pRegions[i].srcOffsets[0].z) };
        vk_blitRegions[i].srcOffsets[1] = { static_cast<int32_t>(pRegions[i].srcOffsets[1].x), static_cast<int32_t>(pRegions[i].srcOffsets[1].y), static_cast<int32_t>(pRegions[i].srcOffsets[1].z) };

        vk_blitRegions[i].dstSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].dstSubresource.aspectMask.value);
        vk_blitRegions[i].dstSubresource.mipLevel = pRegions[i].dstSubresource.mipLevel;
        vk_blitRegions[i].dstSubresource.baseArrayLayer = pRegions[i].dstSubresource.baseArrayLayer;
        vk_blitRegions[i].dstSubresource.layerCount = pRegions[i].dstSubresource.layerCount;

        // Todo(achal): Remove `static_cast`s
        vk_blitRegions[i].dstOffsets[0] = { static_cast<int32_t>(pRegions[i].dstOffsets[0].x), static_cast<int32_t>(pRegions[i].dstOffsets[0].y), static_cast<int32_t>(pRegions[i].dstOffsets[0].z) };
        vk_blitRegions[i].dstOffsets[1] = { static_cast<int32_t>(pRegions[i].dstOffsets[1].x), static_cast<int32_t>(pRegions[i].dstOffsets[1].y), static_cast<int32_t>(pRegions[i].dstOffsets[1].z) };
    }

    getFunctionTable().vkCmdBlitImage(m_cmdbuf, vk_srcImage, getVkImageLayoutFromImageLayout(srcImageLayout),
        vk_dstImage, getVkImageLayoutFromImageLayout(dstImageLayout), regionCount, vk_blitRegions,
        static_cast<VkFilter>(filter));

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

#if 0
bool CVulkanCommandBuffer::setViewport(uint32_t firstViewport, uint32_t viewportCount, const asset::SViewport* pViewports)
{
    constexpr uint32_t MAX_VIEWPORT_COUNT = (1u << 12) / sizeof(VkViewport);
    assert(viewportCount <= MAX_VIEWPORT_COUNT);

    VkViewport vk_viewports[MAX_VIEWPORT_COUNT];
    for (uint32_t i = 0u; i < viewportCount; ++i)
    {
        vk_viewports[i].x = pViewports[i].x;
        vk_viewports[i].y = pViewports[i].y;
        vk_viewports[i].width = pViewports[i].width;
        vk_viewports[i].height = pViewports[i].height;
        vk_viewports[i].minDepth = pViewports[i].minDepth;
        vk_viewports[i].maxDepth = pViewports[i].maxDepth;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    getFunctionTable().vkCmdSetViewport(m_cmdbuf, firstViewport, viewportCount, vk_viewports);
    return true;
}

static std::vector<core::smart_refctd_ptr<const core::IReferenceCounted>> getBuildGeometryInfoReferences(const IGPUAccelerationStructure::DeviceBuildGeometryInfo& info)
{   
    // TODO: Use Better Container than Vector
    std::vector<core::smart_refctd_ptr<const core::IReferenceCounted>> ret;
        
    static constexpr size_t MaxGeometryPerBuildInfoCount = 64;
    // + 3 because of info.srcAS + info.dstAS + info.scratchAddr.buffer
    // * 3 because of worst-case all triangle data ( vertexData + indexData + transformData+
    ret.resize(MaxGeometryPerBuildInfoCount * 3 + 3); 

    ret.push_back(core::smart_refctd_ptr<const IGPUAccelerationStructure>(info.srcAS));
    ret.push_back(core::smart_refctd_ptr<const IGPUAccelerationStructure>(info.dstAS));
    ret.push_back(info.scratchAddr.buffer);
                
    if(!info.geometries.empty())
    {
        IGPUAccelerationStructure::Geometry<IGPUAccelerationStructure::DeviceAddressType>* geoms = info.geometries.begin();
        for(uint32_t g = 0; g < info.geometries.size(); ++g)
        {
            auto const & geometry = geoms[g];
            if(IGPUAccelerationStructure::E_GEOM_TYPE::EGT_TRIANGLES == geometry.type)
            {
                auto const & triangles = geometry.data.triangles;
                if (triangles.vertexData.isValid())
                    ret.push_back(triangles.vertexData.buffer);
                if (triangles.indexData.isValid())
                    ret.push_back(triangles.indexData.buffer);
                if (triangles.transformData.isValid())
                    ret.push_back(triangles.transformData.buffer);
            }
            else if(IGPUAccelerationStructure::E_GEOM_TYPE::EGT_AABBS == geometry.type)
            {
                const auto & aabbs = geometry.data.aabbs;
                if (aabbs.data.isValid())
                    ret.push_back(aabbs.data.buffer);
            }
            else if(IGPUAccelerationStructure::E_GEOM_TYPE::EGT_INSTANCES == geometry.type)
            {
                const auto & instances = geometry.data.instances;
                if (instances.data.isValid())
                    ret.push_back(instances.data.buffer);
            }
        }
    }
    return ret;
}
#endif