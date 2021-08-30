#include "CVulkanCommandBuffer.h"

#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{
    bool CVulkanCommandBuffer::buildAccelerationStructures(const core::SRange<accstruct_t::DeviceBuildGeometryInfo>& pInfos, accstruct_t::BuildRangeInfo* const* ppBuildRangeInfos)
    {
        bool ret = false;
        const auto originDevice = getOriginDevice();
        if (originDevice->getAPIType() == EAT_VULKAN)
        {
            VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
            if(!pInfos.empty())
            {
                static constexpr size_t MaxGeometryPerBuildInfoCount = 64;
                static constexpr size_t MaxBuildInfoCount = 128;
                size_t infoCount = pInfos.size();
                assert(infoCount <= MaxBuildInfoCount);
                
                // TODO: Use better container when ready for these stack allocated memories.
                VkAccelerationStructureBuildGeometryInfoKHR vk_buildGeomsInfos[MaxBuildInfoCount] = {};

                uint32_t geometryArrayOffset = 0u;
                VkAccelerationStructureGeometryKHR vk_geometries[MaxGeometryPerBuildInfoCount * MaxBuildInfoCount] = {};

                accstruct_t::DeviceBuildGeometryInfo* infos = pInfos.begin();
                for(uint32_t i = 0; i < infoCount; ++i)
                {
                    uint32_t geomCount = infos[i].geometries.size();

                    assert(geomCount > 0);
                    assert(geomCount <= MaxGeometryPerBuildInfoCount);

                    vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(vk_device, infos[i], &vk_geometries[geometryArrayOffset]);
                    geometryArrayOffset += geomCount; 
                }
                
                static_assert(sizeof(accstruct_t::BuildRangeInfo) == sizeof(VkAccelerationStructureBuildRangeInfoKHR));
                auto buildRangeInfos = reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR* const*>(ppBuildRangeInfos);
                vkCmdBuildAccelerationStructuresKHR(m_cmdbuf, infoCount, vk_buildGeomsInfos, buildRangeInfos);
                ret = true;
            }
        }
        return ret;
    }
    
    bool CVulkanCommandBuffer::buildAccelerationStructuresIndirect(
        const core::SRange<accstruct_t::DeviceBuildGeometryInfo>& pInfos, 
        const core::SRange<accstruct_t::DeviceAddressType>& pIndirectDeviceAddresses,
        const uint32_t* pIndirectStrides,
        const uint32_t* const* ppMaxPrimitiveCounts)
    {
        bool ret = false;
        const auto originDevice = getOriginDevice();
        if (originDevice->getAPIType() == EAT_VULKAN)
        {
            VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
            if(!pInfos.empty())
            {
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

                accstruct_t::DeviceBuildGeometryInfo* infos = pInfos.begin();
                accstruct_t::DeviceAddressType* indirectDeviceAddresses = pIndirectDeviceAddresses.begin();
                for(uint32_t i = 0; i < infoCount; ++i)
                {
                    uint32_t geomCount = infos[i].geometries.size();

                    assert(geomCount > 0);
                    assert(geomCount <= MaxGeometryPerBuildInfoCount);

                    vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(vk_device, infos[i], &vk_geometries[geometryArrayOffset]);
                    geometryArrayOffset += geomCount;

                    auto addr = CVulkanAccelerationStructure::getVkDeviceOrHostAddress<accstruct_t::DeviceAddressType>(vk_device, indirectDeviceAddresses[i]);
                    vk_indirectDeviceAddresses[i] = addr.deviceAddress;
                }
                
                vkCmdBuildAccelerationStructuresIndirectKHR(m_cmdbuf, infoCount, vk_buildGeomsInfos, vk_indirectDeviceAddresses, pIndirectStrides, ppMaxPrimitiveCounts);
                ret = true;
            }
        }
        return ret;
    }

    bool CVulkanCommandBuffer::copyAccelerationStructure(const accstruct_t::CopyInfo& copyInfo)
    {
        bool ret = false;
        const auto originDevice = getOriginDevice();
        if (originDevice->getAPIType() == EAT_VULKAN)
        {
            VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
            if(copyInfo.dst == nullptr || copyInfo.src == nullptr) 
            {
                assert(false && "invalid src or dst");
                return false;
            }

            VkCopyAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyInfo(vk_device, copyInfo);
            vkCmdCopyAccelerationStructureKHR(m_cmdbuf, &info);
            ret = true;
        }
        return ret;
    }
    
    bool CVulkanCommandBuffer::copyAccelerationStructureToMemory(const accstruct_t::DeviceCopyToMemoryInfo& copyInfo)
    {
        bool ret = false;
        const auto originDevice = getOriginDevice();
        if (originDevice->getAPIType() == EAT_VULKAN)
        {
            VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
            if(copyInfo.dst.isValid() == false || copyInfo.src == nullptr) 
            {
                assert(false && "invalid src or dst");
                return false;
            }

            VkCopyAccelerationStructureToMemoryInfoKHR info = CVulkanAccelerationStructure::getVkASCopyToMemoryInfo(vk_device, copyInfo);
            vkCmdCopyAccelerationStructureToMemoryKHR(m_cmdbuf, &info);
            ret = true;
        }
        return ret;
    }

    bool CVulkanCommandBuffer::copyAccelerationStructureFromMemory(const accstruct_t::DeviceCopyFromMemoryInfo& copyInfo)
    {
        bool ret = false;
        const auto originDevice = getOriginDevice();
        if (originDevice->getAPIType() == EAT_VULKAN)
        {
            VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
            if(copyInfo.dst == nullptr || copyInfo.src.isValid() == false) 
            {
                assert(false && "invalid src or dst");
                return false;
            }

            VkCopyMemoryToAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyFromMemoryInfo(vk_device, copyInfo);
            vkCmdCopyMemoryToAccelerationStructureKHR(m_cmdbuf, &info);
            ret = true;
        }
        return ret;
    }
}