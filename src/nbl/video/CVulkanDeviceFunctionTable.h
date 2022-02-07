#ifndef __C_VULKAN_FUNCTION_TABLE_H_INCLUDED__
#define __C_VULKAN_FUNCTION_TABLE_H_INCLUDED__

#include <volk/volk.h>

namespace nbl::video
{
class CVulkanDeviceFunctionTable
{
public:
    CVulkanDeviceFunctionTable(VkDevice dev)
    {
        volkLoadDeviceTable(&vk, dev);
    }

    VolkDeviceTable vk;
};

}

#endif