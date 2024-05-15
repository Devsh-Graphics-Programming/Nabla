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
        inline CVulkanQueue(ILogicalDevice* logicalDevice, renderdoc_api_t* rdoc, VkInstance vkinst, VkQueue vkq, const uint32_t _famIx, const core::bitflag<IQueue::CREATE_FLAGS> _flags, const float _priority)
            : IQueue(logicalDevice, _famIx, _flags, _priority), m_vkQueue(vkq), m_rdoc_api(rdoc), m_vkInstance(vkinst) {}

        static inline RESULT getResultFrom(const VkResult result)
        {
            switch (result)
            {
                case VK_SUCCESS:
                    return RESULT::SUCCESS;
                    break;
                case VK_ERROR_DEVICE_LOST:
                    return RESULT::DEVICE_LOST;
                    break;
                default:
                    break;
            }
            return RESULT::OTHER_ERROR;
        }

        bool startCapture() override;
        bool endCapture() override;
        bool insertDebugMarker(const char* name, const core::vector4df_SIMD& color) override;
        bool beginDebugMarker(const char* name, const core::vector4df_SIMD& color) override;
        bool endDebugMarker() override;
        
        inline const void* getNativeHandle() const override {return &m_vkQueue;}
        inline VkQueue getInternalObject() const {return m_vkQueue;}

    private:
        RESULT submit_impl(const std::span<const SSubmitInfo> _submits) override;
        RESULT waitIdle_impl() const override;

        renderdoc_api_t* m_rdoc_api;
	    VkInstance m_vkInstance;
        VkQueue m_vkQueue;
};

}

#endif
