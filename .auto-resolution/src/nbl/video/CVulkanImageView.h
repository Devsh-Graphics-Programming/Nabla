#ifndef _NBL_C_VULKAN_IMAGE_VIEW_H_INCLUDED_
#define _NBL_C_VULKAN_IMAGE_VIEW_H_INCLUDED_

#include "nbl/video/IGPUImageView.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanImageView final : public IGPUImageView
{
    public:
        CVulkanImageView(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, SCreationParams&& _params, VkImageView imageView)
            : IGPUImageView(std::move(logicalDevice), std::move(_params)), m_vkImageView(imageView)
        {}

        ~CVulkanImageView();
    
	    inline const void* getNativeHandle() const override {return &m_vkImageView;}
        inline VkImageView getInternalObject() const { return m_vkImageView; }

        void setObjectDebugName(const char* label) const override;

    private:
        VkImageView m_vkImageView;
};

}

#endif
