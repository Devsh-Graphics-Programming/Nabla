#ifndef _NBL_VIDEO_C_VULKAN_SEMAPHORE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_SEMAPHORE_H_INCLUDED_


#include "nbl/video/ISemaphore.h"

#include <volk.h>


namespace nbl::video
{

class ILogicalDevice;

class CVulkanSemaphore final : public ISemaphore
{
    public:
        inline CVulkanSemaphore(core::smart_refctd_ptr<const ILogicalDevice>&& _vkdev, SCreationParams&& creationParams, const VkSemaphore semaphore, std::unique_ptr<system::external_handle_t[]> externalHandles)
            : ISemaphore(std::move(_vkdev), std::move(creationParams)), m_semaphore(semaphore), m_externalHandles(std::move(externalHandles)) {}
        ~CVulkanSemaphore();

        uint64_t getCounterValue() const override;
        void signal(const uint64_t value) override;
    
	    inline const void* getNativeHandle() const override {return &m_semaphore;}
        VkSemaphore getInternalObject() const {return m_semaphore;}

        system::external_handle_t getExportHandle(E_EXTERNAL_HANDLE_TYPE handleType) const override;

        void setObjectDebugName(const char* label) const override;

    private:
        const VkSemaphore m_semaphore;

        // Can store either duplicated importHandle or exportHandle.
        // For now, it only store exportHandle, since we haven't support importing external semaphore yet
        std::unique_ptr<system::external_handle_t[]> m_externalHandles;
};

}

#endif