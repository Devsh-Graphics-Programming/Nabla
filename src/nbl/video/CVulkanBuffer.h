#ifndef __NBL_C_VULKAN_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUBuffer.h"

#include <volk.h>

namespace nbl::video
{
class ILogicalDevice;

class CVulkanBuffer : public IGPUBuffer
{
public:
    CVulkanBuffer(ILogicalDevice* dev, const IDriverMemoryBacked::SDriverMemoryRequirements& reqs, 
        VkBuffer buffer) : IGPUBuffer(dev, reqs), m_buffer(buffer)
    {}

    ~CVulkanBuffer();

    inline VkBuffer getInternalObject() const { return m_buffer; };

private:
    VkBuffer m_buffer;

};

}

#define __NBL_C_VULKAN_BUFFER_H_INCLUDED__
#endif
