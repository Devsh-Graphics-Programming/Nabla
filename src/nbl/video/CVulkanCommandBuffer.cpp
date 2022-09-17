#include "nbl/video/CVulkanCommandBuffer.h"

#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanQueryPool.h"

namespace nbl::video
{

bool CVulkanCommandBuffer::copyImageToBuffer_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions)
{
    VkImage vk_srcImage = IBackendObject::compatibility_cast<const CVulkanImage*>(srcImage, this)->getInternalObject();
    VkBuffer vk_dstBuffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(dstBuffer, this)->getInternalObject();

    constexpr uint32_t MAX_REGION_COUNT = (1u << 12) / sizeof(VkBufferImageCopy);
    VkBufferImageCopy vk_copyRegions[MAX_REGION_COUNT];
    assert(regionCount <= MAX_REGION_COUNT);

    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        vk_copyRegions[i].bufferOffset = static_cast<VkDeviceSize>(pRegions[i].bufferOffset);
        vk_copyRegions[i].bufferRowLength = pRegions[i].bufferRowLength;
        vk_copyRegions[i].bufferImageHeight = pRegions[i].bufferImageHeight;
        vk_copyRegions[i].imageSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].imageSubresource.aspectMask);
        vk_copyRegions[i].imageSubresource.baseArrayLayer = pRegions[i].imageSubresource.baseArrayLayer;
        vk_copyRegions[i].imageSubresource.layerCount = pRegions[i].imageSubresource.layerCount;
        vk_copyRegions[i].imageSubresource.mipLevel = pRegions[i].imageSubresource.mipLevel;
        vk_copyRegions[i].imageOffset = { static_cast<int32_t>(pRegions[i].imageOffset.x), static_cast<int32_t>(pRegions[i].imageOffset.y), static_cast<int32_t>(pRegions[i].imageOffset.z) };
        vk_copyRegions[i].imageExtent = { pRegions[i].imageExtent.width, pRegions[i].imageExtent.height, pRegions[i].imageExtent.depth };
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdCopyImageToBuffer(
        m_cmdbuf,
        vk_srcImage,
        static_cast<VkImageLayout>(srcImageLayout),
        vk_dstBuffer,
        regionCount,
        vk_copyRegions);

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

bool CVulkanCommandBuffer::buildAccelerationStructures_impl(const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos)
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

    IGPUAccelerationStructure::DeviceBuildGeometryInfo* infos = pInfos.begin();

    for(uint32_t i = 0; i < infoCount; ++i)
    {
        uint32_t geomCount = infos[i].geometries.size();
        vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(vk_device, vk, infos[i], &vk_geometries[geometryArrayOffset]);
        geometryArrayOffset += geomCount;
    }

    static_assert(sizeof(IGPUAccelerationStructure::BuildRangeInfo) == sizeof(VkAccelerationStructureBuildRangeInfoKHR));
    auto buildRangeInfos = reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR* const*>(ppBuildRangeInfos);
    vk->vk.vkCmdBuildAccelerationStructuresKHR(m_cmdbuf, infoCount, vk_buildGeomsInfos, buildRangeInfos);
    
    return true;
}
    
bool CVulkanCommandBuffer::buildAccelerationStructuresIndirect_impl(
    const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, 
    const core::SRange<IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses,
    const uint32_t* pIndirectStrides,
    const uint32_t* const* ppMaxPrimitiveCounts)
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

    IGPUAccelerationStructure::DeviceBuildGeometryInfo* infos = pInfos.begin();
    IGPUAccelerationStructure::DeviceAddressType* indirectDeviceAddresses = pIndirectDeviceAddresses.begin();
    for(uint32_t i = 0; i < infoCount; ++i)
    {
        uint32_t geomCount = infos[i].geometries.size();

        vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(vk_device, vk, infos[i], &vk_geometries[geometryArrayOffset]);
        geometryArrayOffset += geomCount;

        auto addr = CVulkanAccelerationStructure::getVkDeviceOrHostAddress<IGPUAccelerationStructure::DeviceAddressType>(vk_device, vk, indirectDeviceAddresses[i]);
        vk_indirectDeviceAddresses[i] = addr.deviceAddress;
    }
                
    vk->vk.vkCmdBuildAccelerationStructuresIndirectKHR(m_cmdbuf, infoCount, vk_buildGeomsInfos, vk_indirectDeviceAddresses, pIndirectStrides, ppMaxPrimitiveCounts);
    return true;
}

bool CVulkanCommandBuffer::copyAccelerationStructure(const IGPUAccelerationStructure::CopyInfo& copyInfo)
{
    bool ret = false;
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(originDevice);
        VkDevice vk_device = vulkanDevice->getInternalObject();
        auto* vk = vulkanDevice->getFunctionTable();

        if(copyInfo.dst == nullptr || copyInfo.src == nullptr) 
        {
            assert(false && "invalid src or dst");
            return false;
        }
            
        // Add Ref to CmdPool
        core::smart_refctd_ptr<const core::IReferenceCounted> tmpRefCntd[2] = 
        {
            core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.src),
            core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.dst),
        };
        CVulkanCommandPool* vulkanCommandPool = IBackendObject::compatibility_cast<CVulkanCommandPool*>(m_cmdpool.get(), this);
        vulkanCommandPool->emplace_n(m_argListTail, tmpRefCntd, tmpRefCntd + 2);


        VkCopyAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyInfo(vk_device, vk, copyInfo);
        vk->vk.vkCmdCopyAccelerationStructureKHR(m_cmdbuf, &info);
        ret = true;
    }
    return ret;
}
    
bool CVulkanCommandBuffer::copyAccelerationStructureToMemory(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo)
{
    bool ret = false;
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(originDevice);
        VkDevice vk_device = vulkanDevice->getInternalObject();
        auto* vk = vulkanDevice->getFunctionTable();

        if(copyInfo.dst.isValid() == false || copyInfo.src == nullptr) 
        {
            assert(false && "invalid src or dst");
            return false;
        }
            
        // Add Ref to CmdPool
        core::smart_refctd_ptr<const core::IReferenceCounted> tmpRefCntd[2] = 
        {
            copyInfo.dst.buffer,
            core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.src),
        };
        CVulkanCommandPool* vulkanCommandPool = IBackendObject::compatibility_cast<CVulkanCommandPool*>(m_cmdpool.get(), this);
        vulkanCommandPool->emplace_n(m_argListTail, tmpRefCntd, tmpRefCntd + 2);
            
        VkCopyAccelerationStructureToMemoryInfoKHR info = CVulkanAccelerationStructure::getVkASCopyToMemoryInfo(vk_device, vk, copyInfo);
        vk->vk.vkCmdCopyAccelerationStructureToMemoryKHR(m_cmdbuf, &info);
        ret = true;
    }
    return ret;
}

bool CVulkanCommandBuffer::copyAccelerationStructureFromMemory(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo)
{
    bool ret = false;
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(originDevice);
        VkDevice vk_device = vulkanDevice->getInternalObject();
        auto* vk = vulkanDevice->getFunctionTable();

        if(copyInfo.dst == nullptr || copyInfo.src.isValid() == false) 
        {
            assert(false && "invalid src or dst");
            return false;
        }
            
        // Add Ref to CmdPool
        core::smart_refctd_ptr<const core::IReferenceCounted> tmpRefCntd[2] = 
        {
            copyInfo.src.buffer,
            core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.dst),
        };
        CVulkanCommandPool* vulkanCommandPool = IBackendObject::compatibility_cast<CVulkanCommandPool*>(m_cmdpool.get(), this);
        vulkanCommandPool->emplace_n(m_argListTail, tmpRefCntd, tmpRefCntd + 2);
            
        VkCopyMemoryToAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyFromMemoryInfo(vk_device, vk, copyInfo);
        vk->vk.vkCmdCopyMemoryToAccelerationStructureKHR(m_cmdbuf, &info);
        ret = true;
    }
    return ret;
}
    
bool CVulkanCommandBuffer::resetQueryPool_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    vk->vk.vkCmdResetQueryPool(m_cmdbuf, vk_queryPool, firstQuery, queryCount);

    return true;
}

bool CVulkanCommandBuffer::beginQuery_impl(IQueryPool* queryPool, uint32_t query, core::bitflag<IQueryPool::E_QUERY_CONTROL_FLAGS> flags)
{
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    auto vk_flags = CVulkanQueryPool::getVkQueryControlFlagsFromQueryControlFlags(flags.value);
    vk->vk.vkCmdBeginQuery(m_cmdbuf, vk_queryPool, query, vk_flags);

    return true;
}

bool CVulkanCommandBuffer::endQuery_impl(IQueryPool* queryPool, uint32_t query)
{
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    vk->vk.vkCmdEndQuery(m_cmdbuf, vk_queryPool, query);

    return true;
}

bool CVulkanCommandBuffer::copyQueryPoolResults_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags)
{
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    auto vk_dstBuffer = IBackendObject::compatibility_cast<CVulkanBuffer*>(dstBuffer, this)->getInternalObject();
    auto vk_queryResultsFlags = CVulkanQueryPool::getVkQueryResultsFlagsFromQueryResultsFlags(flags.value); 
    vk->vk.vkCmdCopyQueryPoolResults(m_cmdbuf, vk_queryPool, firstQuery, queryCount, vk_dstBuffer, dstOffset, static_cast<VkDeviceSize>(stride), vk_queryResultsFlags);
        
    return true;
}

bool CVulkanCommandBuffer::writeTimestamp_impl(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* queryPool, uint32_t query)
{
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    auto vk_pipelineStageFlagBit = static_cast<VkPipelineStageFlagBits>(getVkPipelineStageFlagsFromPipelineStageFlags(pipelineStage));
    vk->vk.vkCmdWriteTimestamp(m_cmdbuf, vk_pipelineStageFlagBit, vk_queryPool, query);

    return true;
}

bool CVulkanCommandBuffer::writeAccelerationStructureProperties_impl(const core::SRange<IGPUAccelerationStructure>& pAccelerationStructures, IQueryPool::E_QUERY_TYPE queryType, IQueryPool* queryPool, uint32_t firstQuery) 
{
    // TODO: Use Better Containers
    static constexpr size_t MaxAccelerationStructureCount = 128;
    uint32_t asCount = static_cast<uint32_t>(pAccelerationStructures.size());
    assert(asCount <= MaxAccelerationStructureCount);
    auto accelerationStructures = pAccelerationStructures.begin();

    VkAccelerationStructureKHR vk_accelerationStructures[MaxAccelerationStructureCount] = {};
    for(size_t i = 0; i < asCount; ++i) 
        vk_accelerationStructures[i] = IBackendObject::compatibility_cast<CVulkanAccelerationStructure*>(&accelerationStructures[i], this)->getInternalObject();
            
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();

    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    auto vk_queryType = CVulkanQueryPool::getVkQueryTypeFromQueryType(queryType);
    vk->vk.vkCmdWriteAccelerationStructuresPropertiesKHR(m_cmdbuf, asCount, vk_accelerationStructures, vk_queryType, vk_queryPool, firstQuery);

    return true;
}

}