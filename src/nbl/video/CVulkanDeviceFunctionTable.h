#ifndef _C_VULKAN_FUNCTION_TABLE_H_INCLUDED_
#define _C_VULKAN_FUNCTION_TABLE_H_INCLUDED_

NBL_PUSH_DISABLE_WARNINGS
#include <volk/volk.h>
NBL_POP_DISABLE_WARNINGS

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