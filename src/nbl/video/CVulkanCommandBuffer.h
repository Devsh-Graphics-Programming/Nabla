#ifndef __NBL_C_VULKAN_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_VULKAN_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUCommandBuffer.h"

#include <volk.h>

namespace nbl
{
namespace video
{

class CVulkanCommandBuffer : public virtual IGPUCommandBuffer
{
public:
    VkCommandBuffer getInternalObject() const { return m_cmdbuf; }

    // TODO impl member functions

protected:
    explicit CVulkanCommandBuffer(CVKLogicalDevice* logicalDevice, E_LEVEL level, VkCommandBuffer _vkcmdbuf, IGPUCommandPool* commandPool)
        : IGPUCommandBuffer(logicalDevice, level, commandPool), m_cmdbuf(_vkcmdbuf) {}

    VkCommandBuffer m_cmdbuf;
};

}
}

#endif
