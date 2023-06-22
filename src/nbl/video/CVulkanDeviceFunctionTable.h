#ifndef _C_VULKAN_FUNCTION_TABLE_H_INCLUDED_
#define _C_VULKAN_FUNCTION_TABLE_H_INCLUDED_

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