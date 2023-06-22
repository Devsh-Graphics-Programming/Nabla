#ifndef _NBL_VIDEO_C_VULKAN_QUEUE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_QUEUE_H_INCLUDED_


#include "nbl/video/IQueue.h"

#include <volk.h>


namespace nbl::video
{

class ILogicalDevice;

class CVulkanQueue final : public IQueue
{
    public:
        inline CVulkanQueue(ILogicalDevice* logicalDevice, renderdoc_api_t* rdoc, VkInstance vkinst, VkQueue vkq, uint32_t _famIx, IQueue::CREATE_FLAGS _flags, float _priority)
            : IQueue(logicalDevice, _famIx, _flags, _priority), m_vkQueue(vkq), m_rdoc_api(rdoc), m_vkInstance(vkinst) {}

        inline const void* getNativeHandle() const override {return &m_vkQueue;}
        inline VkQueue getInternalObject() const {return m_vkQueue;}

        bool startCapture() override;
        bool endCapture() override;

    private:
        bool submit_impl(const uint32_t _count, const SSubmitInfo* _submits) override;

        renderdoc_api_t* m_rdoc_api;
	    VkInstance m_vkInstance;
        VkQueue m_vkQueue;
};

}

#endif
