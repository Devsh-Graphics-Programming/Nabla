#ifndef __NBL_C_VULKAN_PRIMARY_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_VULKAN_PRIMARY_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUPrimaryCommandBuffer.h"
#include "nbl/video/CVulkanCommandBuffer.h"

namespace nbl {
namespace video
{

class CVulkanPrimaryCommandBuffer final : public IGPUPrimaryCommandBuffer, public CVulkanCommandBuffer
{
public:
    CVulkanPrimaryCommandBuffer(const IGPUCommandPool* _cmdpool, VkCommandBuffer _vkcmdbuf) : 
        IGPUCommandBuffer(_cmdpool), // init virtual base
        IGPUPrimaryCommandBuffer(),
        CVulkanCommandBuffer(_vkcmdbuf)
    {}
};

}}

#endif
