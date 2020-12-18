#ifndef __C_VULKAN_FUNCTION_TABLE_H_INCLUDED__
#define __C_VULKAN_FUNCTION_TABLE_H_INCLUDED__

#include <volk.h>

#include "nbl/core/IReferenceCounted.h"

namespace nbl {
namespace video
{

class CVulkanDeviceFunctionTable : public core::IReferenceCounted
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