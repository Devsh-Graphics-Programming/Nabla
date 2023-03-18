#ifndef __NBL_VIDEO_C_VULKAN_DESCRIPTOR_SET_H_INCLUDED__

#include "nbl/video/IGPUDescriptorSet.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;
class CVulkanDescriptorPool;

class CVulkanDescriptorSet : public IGPUDescriptorSet
{
public:
    CVulkanDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout, core::smart_refctd_ptr<IDescriptorPool>&& pool, IDescriptorPool::SStorageOffsets offsets, VkDescriptorSet descriptorSet)
        : IGPUDescriptorSet(std::move(layout), std::move(pool), std::move(offsets)), m_descriptorSet(descriptorSet)
    {}

    ~CVulkanDescriptorSet();

    inline VkDescriptorSet getInternalObject() const { return m_descriptorSet; }

private:
    VkDescriptorSet m_descriptorSet;
};

}

#define __NBL_VIDEO_C_VULKAN_DESCRIPTOR_SET_H_INCLUDED__
#endif
