#ifndef _NBL_C_VULKAN_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_C_VULKAN_ACCELERATION_STRUCTURE_H_INCLUDED_

#include "nbl/video/IGPUAccelerationStructure.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanAccelerationStructure final : public IGPUAccelerationStructure
{
public:
    CVulkanAccelerationStructure(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice,
        SCreationParams&& _params, VkAccelerationStructureKHR accelerationStructure)
        : IGPUAccelerationStructure(std::move(logicalDevice), std::move(_params)), m_vkAccelerationStructure(accelerationStructure)
    {}

    ~CVulkanAccelerationStructure();

    inline VkAccelerationStructureKHR getInternalObject() const { return m_vkAccelerationStructure; }

private:
    VkAccelerationStructureKHR m_vkAccelerationStructure;
};

}

#endif
