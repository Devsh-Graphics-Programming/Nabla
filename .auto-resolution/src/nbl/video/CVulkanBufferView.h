#ifndef _NBL_VIDEO_C_VULKAN_BUFFER_VIEW_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_BUFFER_VIEW_H_INCLUDED_

#include "nbl/video/IGPUBufferView.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanBufferView : public IGPUBufferView
{
    public:
        CVulkanBufferView(
            const ILogicalDevice* dev, const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT format,
            const VkBufferView handle
        ) : IGPUBufferView(core::smart_refctd_ptr<const ILogicalDevice>(dev),underlying,format), m_vkBufferView(handle) {}
    
        inline const void* getNativeHandle() const override {return &m_vkBufferView;}
        inline VkBufferView getInternalObject() const {return m_vkBufferView;}

        ~CVulkanBufferView();
	
        void setObjectDebugName(const char* label) const override;

    private:
        VkBufferView m_vkBufferView;
};

}
#endif
