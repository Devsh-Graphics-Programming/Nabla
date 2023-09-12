#ifndef _NBL_VIDEO_C_VULKAN_SWAPCHAIN_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_SWAPCHAIN_H_INCLUDED_

#include "nbl/video/ISwapchain.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanSwapchain final : public ISwapchain
{
    public:
        NBL_API2 static core::smart_refctd_ptr<CVulkanSwapchain> create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);

        core::smart_refctd_ptr<IGPUImage> createImage(const uint32_t imageIndex) override;

        void setObjectDebugName(const char* label) const override;

        inline const void* getNativeHandle() const {return &m_vkSwapchainKHR;}
        inline VkSwapchainKHR getInternalObject() const {return m_vkSwapchainKHR;}

    private:
        CVulkanSwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& logicalDevice, SCreationParams&& params, const uint32_t imageCount, const VkSwapchainKHR swapchain, const VkSemaphore* const _adaptorSemaphores);
        ~CVulkanSwapchain();

        ACQUIRE_IMAGE_RESULT acquireNextImage_impl(const SAcquireInfo& info, uint32_t* const out_imgIx) override;

        PRESENT_RESULT present_impl(const SPresentInfo& info) override;

        inline VkSemaphoreSubmitInfoKHR getAdaptorSemaphore(const bool notNull)
        {
            VkSemaphoreSubmitInfoKHR info = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,nullptr,VK_NULL_HANDLE};
            if (notNull)
            {
                info.semaphore = m_adaptorSemaphores[(m_internalCounter++)%(2*m_imageCount)];
                // value is ignored because the adaptors are binary
                info.stageMask = VK_PIPELINE_STAGE_2_NONE;
                info.deviceIndex = 0u; // TODO: later obtain from swapchain
            }
            return info;
        }

        IDeviceMemoryBacked::SDeviceMemoryRequirements m_imgMemRequirements;
        VkSwapchainKHR m_vkSwapchainKHR;
        VkImage m_images[ISwapchain::MaxImages];
        VkSemaphore m_adaptorSemaphores[2*ISwapchain::MaxImages];
        uint8_t m_internalCounter = 0u;
};

}

#endif