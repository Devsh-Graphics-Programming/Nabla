#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanPhysicalDevice.h"
namespace nbl::video
{
core::smart_refctd_ptr<IGPUAccelerationStructure> CVulkanLogicalDevice::createGPUAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) 
{
    auto physicalDevice = static_cast<const CVulkanPhysicalDevice*>(getPhysicalDevice());
    auto features = physicalDevice->getFeatures();
    
    // TODO(Validation): accelerationStructure feature must be enabled
    if(!features.accelerationStructure) // this is not "enabled" feature :(
    {
        assert(false && "device acceleration structures is not enabled.");
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
    // TODO(Validation): accelerationStructureHostCommands feature must be enabled
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
    // TODO(Validation): accelerationStructureHostCommands feature must be enabled
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
    // TODO(Validation): accelerationStructureHostCommands feature must be enabled
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
    // TODO(Validation): accelerationStructureHostCommands feature must be enabled
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

IGPUAccelerationStructure::BuildSizes CVulkanLogicalDevice::getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::HostBuildGeometryInfo& pPartialInfos, const uint32_t* pMaxPrimitiveCounts)
{
    // TODO(Validation): Rayquery or RayTracing Pipeline must be enabled
    return getAccelerationStructureBuildSizes_impl(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR, pPartialInfos, pMaxPrimitiveCounts);
}

IGPUAccelerationStructure::BuildSizes CVulkanLogicalDevice::getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::DeviceBuildGeometryInfo& pPartialInfos, const uint32_t* pMaxPrimitiveCounts)
{
    // TODO(Validation): Rayquery or RayTracing Pipeline must be enabled
    return getAccelerationStructureBuildSizes_impl(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, pPartialInfos, pMaxPrimitiveCounts);
}

}