#ifndef __NBL_VIDEO_C_VULKAN_BUFFER_VIEW_H_INCLUDED__

#include "nbl/video/IGPUBufferView.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanBufferView : public IGPUBufferView
{
public:
    CVulkanBufferView(core::smart_refctd_ptr<const ILogicalDevice>&& dev,
        core::smart_refctd_ptr<IGPUBuffer> buffer, asset::E_FORMAT format, size_t offset, size_t size,
        VkBufferView handle)
        : IGPUBufferView(std::move(dev), buffer, format, offset, size), m_vkBufferView(handle)
    {}

    inline VkBufferView getInternalObject() const { return m_vkBufferView; }

    ~CVulkanBufferView();
	
    void setObjectDebugName(const char* label) const override;

private:
    VkBufferView m_vkBufferView;
};

}

#define __NBL_VIDEO_C_VULKAN_BUFFER_VIEW_H_INCLUDED__
#endif
