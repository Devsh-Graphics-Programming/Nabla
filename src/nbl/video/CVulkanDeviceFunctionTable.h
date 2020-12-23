#ifndef __C_VULKAN_FUNCTION_TABLE_H_INCLUDED__
#define __C_VULKAN_FUNCTION_TABLE_H_INCLUDED__

#include <volk.h>

namespace nbl {
namespace video
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
}

#endif