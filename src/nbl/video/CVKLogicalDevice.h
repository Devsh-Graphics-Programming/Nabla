#ifndef __NBL_C_VK_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_VK_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/CVulkanDeviceFunctionTable.h"

namespace nbl {
namespace video
{

class CVKLogicalDevice final : public ILogicalDevice
{
public:
    CVKLogicalDevice(VkDevice vkdev) :
        m_vkdev(vkdev),
        m_devf(core::make_smart_refctd_ptr<CVulkanDeviceFunctionTable>(vkdev))
    {
        
    }

private:
    VkDevice m_vkdev;
    core::smart_refctd_ptr<CVulkanDeviceFunctionTable> m_devf;
};

}
}

#endif