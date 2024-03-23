#ifndef _NBL_C_VULKAN_DESCRIPTOR_POOL_H_INCLUDED_
#define _NBL_C_VULKAN_DESCRIPTOR_POOL_H_INCLUDED_


#include "nbl/video/IDescriptorPool.h"

#include <volk.h>


namespace nbl::video
{

class ILogicalDevice;

class CVulkanDescriptorPool : public IDescriptorPool
{
    public:
        CVulkanDescriptorPool(const ILogicalDevice* dev, const IDescriptorPool::SCreateInfo& createInfo, const VkDescriptorPool descriptorPool)
            : IDescriptorPool(core::smart_refctd_ptr<const ILogicalDevice>(dev),createInfo), m_descriptorPool(descriptorPool) {}

        ~CVulkanDescriptorPool();

        inline VkDescriptorPool getInternalObject() const { return m_descriptorPool; }

        void setObjectDebugName(const char* label) const override;

    private:
        bool createDescriptorSets_impl(uint32_t count, const IGPUDescriptorSetLayout* const* layouts, SStorageOffsets *const offsets, core::smart_refctd_ptr<IGPUDescriptorSet>* output) override;
        bool reset_impl() override;

        const VkDescriptorPool m_descriptorPool;
};

}
#endif
