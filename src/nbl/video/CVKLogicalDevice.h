#ifndef __NBL_C_VK_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_VK_LOGICAL_DEVICE_H_INCLUDED__

#include <algorithm>

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/CVulkanDeviceFunctionTable.h"
#include "nbl/video/CVKSwapchain.h"
#include "nbl/video/CVulkanQueue.h"

namespace nbl {
namespace video
{

class CVKLogicalDevice final : public ILogicalDevice
{
public:
    CVKLogicalDevice(VkDevice vkdev, const SCreationParams& params, core::smart_refctd_ptr<io::IFileSystem>&& fs, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc) :
        ILogicalDevice(EAT_VULKAN, params, std::move(fs), std::move(glslc)),
        m_vkdev(vkdev),
        m_devf(vkdev)
    {
        // create actual queue objects
        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
        {
            const auto& qci = params.queueCreateInfos[i];
            const uint32_t famIx = qci.familyIndex;
            const uint32_t offset = (*m_offsets)[famIx];
            const auto flags = qci.flags;

            for (uint32_t j = 0u; j < qci.count; ++j)
            {
                const float priority = qci.priorities[j];

                VkQueue q;
                m_devf.vk.vkGetDeviceQueue(m_vkdev, famIx, j, &q);

                const uint32_t ix = offset + j;
                (*m_queues)[ix] = core::make_smart_refctd_ptr<CVulkanQueue>(this, q, famIx, flags, priority);
            }
        }
    }

    ~CVKLogicalDevice()
    {
        m_devf.vk.vkDestroyDevice(m_vkdev, nullptr);
    }

    core::smart_refctd_ptr<ISwapchain> createSwapchain(ISwapchain::SCreationParams&& params) override
    {
        return core::make_smart_refctd_ptr<CVKSwapchain>(std::move(params), this);
    }

    CVulkanDeviceFunctionTable* getFunctionTable() { return &m_devf; }
    VkDevice getInternalObject() const { return m_vkdev; }

private:
    VkDevice m_vkdev;
    CVulkanDeviceFunctionTable m_devf;
};

}
}

#endif