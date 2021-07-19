#ifndef __NBL_C_VULKAN_PRIMARY_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_VULKAN_PRIMARY_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUPrimaryCommandBuffer.h"
#include "nbl/video/CVulkanCommandBuffer.h"

namespace nbl
{
namespace video
{

class CVKLogicalDevice;

class CVulkanPrimaryCommandBuffer final : public IGPUPrimaryCommandBuffer, public CVulkanCommandBuffer
{
public:
    CVulkanPrimaryCommandBuffer(CVKLogicalDevice* logicalDevice, IGPUCommandPool* _cmdpool, VkCommandBuffer commandBuffer) : 
        IGPUCommandBuffer(logicalDevice, EL_PRIMARY, _cmdpool), // init virtual base
        IGPUPrimaryCommandBuffer(logicalDevice, EL_PRIMARY, _cmdpool),
        CVulkanCommandBuffer(logicalDevice, EL_PRIMARY, commandBuffer, _cmdpool)
    {}
};

}
}

#endif
