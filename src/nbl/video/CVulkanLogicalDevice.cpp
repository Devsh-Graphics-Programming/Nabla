#include "CVulkanLogicalDevice.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanQueryPool.h"

namespace nbl::video
{

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateDeviceLocalMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getDeviceLocalGPUMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(additionalReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateSpilloverMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (additionalReqs.memoryHeapLocation == IDriverMemoryAllocation::ESMT_DEVICE_LOCAL)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getSpilloverGPUMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateUpStreamingMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (getUpStreamingMemoryReqs().memoryHeapLocation != additionalReqs.memoryHeapLocation)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getUpStreamingMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateDownStreamingMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (getDownStreamingMemoryReqs().memoryHeapLocation != additionalReqs.memoryHeapLocation)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getDownStreamingMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateCPUSideGPUVisibleMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (additionalReqs.memoryHeapLocation != IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getCPUSideGPUVisibleGPUMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateGPUMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& reqs)
{
    VkMemoryPropertyFlags desiredMemoryProperties = static_cast<VkMemoryPropertyFlags>(0u);

    if (reqs.memoryHeapLocation == IDriverMemoryAllocation::ESMT_DEVICE_LOCAL)
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if ((reqs.mappingCapability & IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ) ||
        (reqs.mappingCapability & IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE))
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

    if (reqs.mappingCapability & IDriverMemoryAllocation::EMCF_COHERENT)
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    if (reqs.mappingCapability & IDriverMemoryAllocation::EMCF_CACHED)
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    const IPhysicalDevice::SMemoryProperties& memoryProperties = m_physicalDevice->getMemoryProperties();

    uint32_t compatibleMemoryTypeCount = 0u;
    uint32_t compatibleMemoryTypeIndices[VK_MAX_MEMORY_TYPES];

    for (uint32_t i = 0u; i < memoryProperties.memoryTypeCount; ++i)
    {
        const bool memoryTypeSupportedForResource = (reqs.vulkanReqs.memoryTypeBits & (1 << i));

        const bool memoryHasDesirableProperties = (memoryProperties.memoryTypes[i].propertyFlags
            & desiredMemoryProperties) == desiredMemoryProperties;

        if (memoryTypeSupportedForResource && memoryHasDesirableProperties)
            compatibleMemoryTypeIndices[compatibleMemoryTypeCount++] = i;
    }

    for (uint32_t i = 0u; i < compatibleMemoryTypeCount; ++i)
    {
        // Todo(achal): Make use of requiresDedicatedAllocation and prefersDedicatedAllocation

        VkMemoryAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        vk_allocateInfo.pNext = nullptr; // No extensions for now
        vk_allocateInfo.allocationSize = reqs.vulkanReqs.size;
        vk_allocateInfo.memoryTypeIndex = compatibleMemoryTypeIndices[i];

        VkDeviceMemory vk_deviceMemory;
        if (vkAllocateMemory(m_vkdev, &vk_allocateInfo, nullptr, &vk_deviceMemory) == VK_SUCCESS)
        {
            // Todo(achal): Change dedicate to not always be false
            return core::make_smart_refctd_ptr<CVulkanMemoryAllocation>(this, reqs.vulkanReqs.size, false, vk_deviceMemory);
        }
    }

    return nullptr;
}

core::smart_refctd_ptr<IGPUAccelerationStructure> CVulkanLogicalDevice::createGPUAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) 
{
    auto physicalDevice = static_cast<const CVulkanPhysicalDevice*>(getPhysicalDevice());
    auto features = physicalDevice->getFeatures();
    
    if(!features.accelerationStructure)
    {
        assert(false && "device accelerationStructures is not enabled.");
        return nullptr;
    }

    VkAccelerationStructureKHR vk_as = VK_NULL_HANDLE;
    VkAccelerationStructureCreateInfoKHR vasci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR, nullptr};
    vasci.createFlags = CVulkanAccelerationStructure::getVkASCreateFlagsFromASCreateFlags(params.flags);
    vasci.type = CVulkanAccelerationStructure::getVkASTypeFromASType(params.type);
    vasci.buffer = static_cast<const CVulkanBuffer*>(params.bufferRange.buffer.get())->getInternalObject();
    vasci.offset = static_cast<VkDeviceSize>(params.bufferRange.offset);
    vasci.size = static_cast<VkDeviceSize>(params.bufferRange.size);
    auto vk_res = vkCreateAccelerationStructureKHR(m_vkdev, &vasci, nullptr, &vk_as);
    if(VK_SUCCESS != vk_res)
        return nullptr;
    return core::make_smart_refctd_ptr<CVulkanAccelerationStructure>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(params), vk_as);
}

bool CVulkanLogicalDevice::buildAccelerationStructures(
    core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation,
    const core::SRange<IGPUAccelerationStructure::HostBuildGeometryInfo>& pInfos,
    IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos)
{
    auto physicalDevice = static_cast<const CVulkanPhysicalDevice*>(getPhysicalDevice());
    auto features = physicalDevice->getFeatures();
    if(!features.accelerationStructure)
    {
        assert(false && "device acceleration structures is not enabled.");
        return false;
    }


    bool ret = false;
    if(!pInfos.empty() && deferredOperation.get() != nullptr)
    {
        VkDeferredOperationKHR vk_deferredOp = static_cast<CVulkanDeferredOperation *>(deferredOperation.get())->getInternalObject();
        static constexpr size_t MaxGeometryPerBuildInfoCount = 64;
        static constexpr size_t MaxBuildInfoCount = 128;
        size_t infoCount = pInfos.size();
        assert(infoCount <= MaxBuildInfoCount);
                
        // TODO: Use better container when ready for these stack allocated memories.
        VkAccelerationStructureBuildGeometryInfoKHR vk_buildGeomsInfos[MaxBuildInfoCount] = {};

        uint32_t geometryArrayOffset = 0u;
        VkAccelerationStructureGeometryKHR vk_geometries[MaxGeometryPerBuildInfoCount * MaxBuildInfoCount] = {};

        IGPUAccelerationStructure::HostBuildGeometryInfo* infos = pInfos.begin();
        for(uint32_t i = 0; i < infoCount; ++i)
        {
            uint32_t geomCount = infos[i].geometries.size();

            assert(geomCount > 0);
            assert(geomCount <= MaxGeometryPerBuildInfoCount);

            vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(m_vkdev, infos[i], &vk_geometries[geometryArrayOffset]);
            geometryArrayOffset += geomCount; 
        }
                
        static_assert(sizeof(IGPUAccelerationStructure::BuildRangeInfo) == sizeof(VkAccelerationStructureBuildRangeInfoKHR));
        auto buildRangeInfos = reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR* const*>(ppBuildRangeInfos);
        VkResult vk_res = vkBuildAccelerationStructuresKHR(m_vkdev, vk_deferredOp, infoCount, vk_buildGeomsInfos, buildRangeInfos);
        if(VK_SUCCESS == vk_res)
        {
            ret = true;
        }
    }
    return ret;
}

bool CVulkanLogicalDevice::copyAccelerationStructure(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo)
{
    auto physicalDevice = static_cast<const CVulkanPhysicalDevice*>(getPhysicalDevice());
    auto features = physicalDevice->getFeatures();
    if(!features.accelerationStructureHostCommands || !features.accelerationStructure)
    {
        assert(false && "device accelerationStructuresHostCommands is not enabled.");
        return false;
    }

    bool ret = false;
    if(deferredOperation.get() != nullptr)
    {
        VkDeferredOperationKHR vk_deferredOp = static_cast<CVulkanDeferredOperation *>(deferredOperation.get())->getInternalObject();
        if(copyInfo.dst == nullptr || copyInfo.src == nullptr) 
        {
            assert(false && "invalid src or dst");
            return false;
        }

        VkCopyAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyInfo(m_vkdev, copyInfo);
        VkResult res = vkCopyAccelerationStructureKHR(m_vkdev, vk_deferredOp, &info);
        if(VK_SUCCESS == res)
        {
            ret = true;
        }
    }
    return ret;
}
    
bool CVulkanLogicalDevice::copyAccelerationStructureToMemory(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::HostCopyToMemoryInfo& copyInfo)
{
    auto physicalDevice = static_cast<const CVulkanPhysicalDevice*>(getPhysicalDevice());
    auto features = physicalDevice->getFeatures();
    if(!features.accelerationStructureHostCommands || !features.accelerationStructure)
    {
        assert(false && "device accelerationStructuresHostCommands is not enabled.");
        return false;
    }

    bool ret = false;
    if(deferredOperation.get() != nullptr)
    {
        VkDeferredOperationKHR vk_deferredOp = static_cast<CVulkanDeferredOperation *>(deferredOperation.get())->getInternalObject();

        if(copyInfo.dst.isValid() == false || copyInfo.src == nullptr) 
        {
            assert(false && "invalid src or dst");
            return false;
        }

        VkCopyAccelerationStructureToMemoryInfoKHR info = CVulkanAccelerationStructure::getVkASCopyToMemoryInfo(m_vkdev, copyInfo);
        VkResult res = vkCopyAccelerationStructureToMemoryKHR(m_vkdev, vk_deferredOp, &info);
        if(VK_SUCCESS == res)
        {
            ret = true;
        }
    }
    return ret;
}

bool CVulkanLogicalDevice::copyAccelerationStructureFromMemory(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::HostCopyFromMemoryInfo& copyInfo)
{
    auto physicalDevice = static_cast<const CVulkanPhysicalDevice*>(getPhysicalDevice());
    auto features = physicalDevice->getFeatures();
    if(!features.accelerationStructureHostCommands || !features.accelerationStructure)
    {
        assert(false && "device accelerationStructuresHostCommands is not enabled.");
        return false;
    }

    bool ret = false;
    if(deferredOperation.get() != nullptr)
    {
        VkDeferredOperationKHR vk_deferredOp = static_cast<CVulkanDeferredOperation *>(deferredOperation.get())->getInternalObject();
        if(copyInfo.dst == nullptr || copyInfo.src.isValid() == false) 
        {
            assert(false && "invalid src or dst");
            return false;
        }

        VkCopyMemoryToAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyFromMemoryInfo(m_vkdev, copyInfo);
        VkResult res = vkCopyMemoryToAccelerationStructureKHR(m_vkdev, vk_deferredOp, &info);
        if(VK_SUCCESS == res)
        {
            ret = true;
        }
    }
    return ret;
}

IGPUAccelerationStructure::BuildSizes CVulkanLogicalDevice::getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::HostBuildGeometryInfo& pBuildInfo, const uint32_t* pMaxPrimitiveCounts)
{
    // TODO(Validation): Rayquery or RayTracing Pipeline must be enabled
    return getAccelerationStructureBuildSizes_impl(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR, pBuildInfo, pMaxPrimitiveCounts);
}

IGPUAccelerationStructure::BuildSizes CVulkanLogicalDevice::getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::DeviceBuildGeometryInfo& pBuildInfo, const uint32_t* pMaxPrimitiveCounts)
{
    // TODO(Validation): Rayquery or RayTracing Pipeline must be enabled
    return getAccelerationStructureBuildSizes_impl(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, pBuildInfo, pMaxPrimitiveCounts);
}

core::smart_refctd_ptr<IQueryPool> CVulkanLogicalDevice::createQueryPool(IQueryPool::SCreationParams&& params)
{
    VkQueryPool vk_queryPool = VK_NULL_HANDLE;
    VkQueryPoolCreateInfo vk_qpci = CVulkanQueryPool::getVkCreateInfoFromCreationParams(std::move(params));
    auto vk_res = vkCreateQueryPool(m_vkdev, &vk_qpci, nullptr, &vk_queryPool);
    if(VK_SUCCESS != vk_res)
        return nullptr;
    return core::make_smart_refctd_ptr<CVulkanQueryPool>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(params), vk_queryPool);
}

bool CVulkanLogicalDevice::getQueryPoolResults(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void * pData, uint64_t stride, IQueryPool::E_QUERY_RESULTS_FLAGS flags)
{
    bool ret = false;
    if(queryPool != nullptr)
    {
        auto vk_queryPool = static_cast<CVulkanQueryPool*>(queryPool)->getInternalObject();
        auto vk_queryResultsflags = CVulkanQueryPool::getVkQueryResultsFlagsFromQueryResultsFlags(flags);
        auto vk_res = vkGetQueryPoolResults(m_vkdev, vk_queryPool, firstQuery, queryCount, dataSize, pData, static_cast<VkDeviceSize>(stride), vk_queryResultsflags);
        if(VK_SUCCESS == vk_res)
            ret = true;
    }
    return ret;
}

}