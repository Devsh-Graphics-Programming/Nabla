#include "CVulkanCommandBuffer.h"

#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanQueryPool.h"

namespace nbl::video
{
    bool CVulkanCommandBuffer::buildAccelerationStructures(const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos)
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

                IGPUAccelerationStructure::DeviceBuildGeometryInfo* infos = pInfos.begin();
                for(uint32_t i = 0; i < infoCount; ++i)
                {
                    uint32_t geomCount = infos[i].geometries.size();

                    assert(geomCount > 0);
                    assert(geomCount <= MaxGeometryPerBuildInfoCount);

                    vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(vk_device, infos[i], &vk_geometries[geometryArrayOffset]);
                    geometryArrayOffset += geomCount; 
                }
                
                static_assert(sizeof(IGPUAccelerationStructure::BuildRangeInfo) == sizeof(VkAccelerationStructureBuildRangeInfoKHR));
                auto buildRangeInfos = reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR* const*>(ppBuildRangeInfos);
                vkCmdBuildAccelerationStructuresKHR(m_cmdbuf, infoCount, vk_buildGeomsInfos, buildRangeInfos);
                ret = true;
            }
        }
        return ret;
    }
    
    bool CVulkanCommandBuffer::buildAccelerationStructuresIndirect(
        const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, 
        const core::SRange<IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses,
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

                IGPUAccelerationStructure::DeviceBuildGeometryInfo* infos = pInfos.begin();
                IGPUAccelerationStructure::DeviceAddressType* indirectDeviceAddresses = pIndirectDeviceAddresses.begin();
                for(uint32_t i = 0; i < infoCount; ++i)
                {
                    uint32_t geomCount = infos[i].geometries.size();

                    assert(geomCount > 0);
                    assert(geomCount <= MaxGeometryPerBuildInfoCount);

                    vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(vk_device, infos[i], &vk_geometries[geometryArrayOffset]);
                    geometryArrayOffset += geomCount;

                    auto addr = CVulkanAccelerationStructure::getVkDeviceOrHostAddress<IGPUAccelerationStructure::DeviceAddressType>(vk_device, indirectDeviceAddresses[i]);
                    vk_indirectDeviceAddresses[i] = addr.deviceAddress;
                }
                
                vkCmdBuildAccelerationStructuresIndirectKHR(m_cmdbuf, infoCount, vk_buildGeomsInfos, vk_indirectDeviceAddresses, pIndirectStrides, ppMaxPrimitiveCounts);
                ret = true;
            }
        }
        return ret;
    }

    bool CVulkanCommandBuffer::copyAccelerationStructure(const IGPUAccelerationStructure::CopyInfo& copyInfo)
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
    
    bool CVulkanCommandBuffer::copyAccelerationStructureToMemory(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo)
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

    bool CVulkanCommandBuffer::copyAccelerationStructureFromMemory(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo)
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
    
    bool CVulkanCommandBuffer::resetQueryPool(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount)
    {
        bool ret = false;
        if(queryPool != nullptr)
        {
            auto vk_queryPool = static_cast<video::CVulkanQueryPool*>(queryPool)->getInternalObject();
            vkCmdResetQueryPool(m_cmdbuf, vk_queryPool, firstQuery, queryCount);
            ret = true;
        }
        return ret;
    }

    bool CVulkanCommandBuffer::beginQuery(video::IQueryPool* queryPool, uint32_t query, video::IQueryPool::E_QUERY_CONTROL_FLAGS flags)
    {
        bool ret = false;
        if(queryPool != nullptr)
        {
            auto vk_queryPool = static_cast<video::CVulkanQueryPool*>(queryPool)->getInternalObject();
            auto vk_flags = CVulkanQueryPool::getVkQueryControlFlagsFromQueryControlFlags(flags);
            vkCmdBeginQuery(m_cmdbuf, vk_queryPool, query, vk_flags);
            ret = true;
        }
        return ret;
    }

    bool CVulkanCommandBuffer::endQuery(video::IQueryPool* queryPool, uint32_t query)
    {
        bool ret = false;
        if(queryPool != nullptr)
        {
            auto vk_queryPool = static_cast<video::CVulkanQueryPool*>(queryPool)->getInternalObject();
            vkCmdEndQuery(m_cmdbuf, vk_queryPool, query);
            ret = true;
        }
        return ret;
    }

    bool CVulkanCommandBuffer::copyQueryPoolResults(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, video::IQueryPool::E_QUERY_RESULTS_FLAGS flags)
    {
        bool ret = false;
        if(queryPool != nullptr && dstBuffer != nullptr)
        {
            auto vk_queryPool = static_cast<video::CVulkanQueryPool*>(queryPool)->getInternalObject();
            auto vk_dstBuffer = static_cast<video::CVulkanBuffer*>(dstBuffer)->getInternalObject();
            auto vk_queryResultsFlags = CVulkanQueryPool::getVkQueryResultsFlagsFromQueryResultsFlags(flags); 
            vkCmdCopyQueryPoolResults(m_cmdbuf, vk_queryPool, firstQuery, queryCount, vk_dstBuffer, dstOffset, static_cast<VkDeviceSize>(stride), vk_queryResultsFlags);
            ret = true;
        }
        return ret;
    }

    bool CVulkanCommandBuffer::writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, video::IQueryPool* queryPool, uint32_t query)
    {
        bool ret = false;
        if(queryPool != nullptr)
        {
            auto vk_queryPool = static_cast<video::CVulkanQueryPool*>(queryPool)->getInternalObject();
            auto vk_pipelineStage = static_cast<VkPipelineStageFlagBits>(pipelineStage); // am I doing this right?

            vkCmdWriteTimestamp(m_cmdbuf, vk_pipelineStage, vk_queryPool, query);
            ret = true;
        }
        return ret;
    }
    // TRANSFORM_FEEDBACK_STREAM
    bool CVulkanCommandBuffer::beginQueryIndexed(video::IQueryPool* queryPool, uint32_t query, uint32_t index, video::IQueryPool::E_QUERY_CONTROL_FLAGS flags)
    {
        bool ret = false;
         // TODO: Check  for PhysicalDevice Availability and Extension
        if(queryPool != nullptr)
        {
            auto vk_queryPool = static_cast<video::CVulkanQueryPool*>(queryPool)->getInternalObject();
            auto vk_flags = CVulkanQueryPool::getVkQueryControlFlagsFromQueryControlFlags(flags);
            vkCmdBeginQueryIndexedEXT(m_cmdbuf, vk_queryPool, query, vk_flags, index);
            ret = true;
        }
        return ret;
    }

    bool CVulkanCommandBuffer::endQueryIndexed(video::IQueryPool* queryPool, uint32_t query, uint32_t index)
    {
        bool ret = false;
        if(queryPool != nullptr)
        {
            auto vk_queryPool = static_cast<video::CVulkanQueryPool*>(queryPool)->getInternalObject();
            vkCmdEndQueryIndexedEXT(m_cmdbuf, vk_queryPool, query, index);
            ret = true;
        }
        return ret;
    }

    // Acceleration Structure Properties (Only available on Vulkan)
    bool CVulkanCommandBuffer::writeAccelerationStructureProperties(const core::SRange<video::IGPUAccelerationStructure>& pAccelerationStructures, video::IQueryPool::E_QUERY_TYPE queryType, video::IQueryPool* queryPool, uint32_t firstQuery) 
    {
        bool ret = false;
        if(queryPool != nullptr && pAccelerationStructures.empty() == false)
        {
            // TODO: Use Better Containers
            static constexpr size_t MaAccelerationStructureCount = 128;
            uint32_t asCount = static_cast<uint32_t>(pAccelerationStructures.size());
            assert(asCount <= MaAccelerationStructureCount);
            auto accelerationStructures = pAccelerationStructures.begin();
            VkAccelerationStructureKHR vk_accelerationStructures[MaAccelerationStructureCount] = {};

            for(size_t i = 0; i < asCount; ++i) 
            {
                vk_accelerationStructures[i] = static_cast<CVulkanAccelerationStructure*>(&accelerationStructures[i])->getInternalObject();
            }

            auto vk_queryPool = static_cast<video::CVulkanQueryPool*>(queryPool)->getInternalObject();
            auto vk_queryType = CVulkanQueryPool::getVkQueryTypeFromQueryType(queryType);
            vkCmdWriteAccelerationStructuresPropertiesKHR(m_cmdbuf, asCount, vk_accelerationStructures, vk_queryType, vk_queryPool, firstQuery);
            ret = true;
        }
        return ret;
    }

}