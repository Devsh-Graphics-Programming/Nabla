#ifndef _NBL_C_VULKAN_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_C_VULKAN_GRAPHICS_PIPELINE_H_INCLUDED_


#include "nbl/video/IGPUGraphicsPipeline.h"


namespace nbl::video
{

class CVulkanGraphicsPipeline final : public IGPUGraphicsPipeline
{
    public:
        CVulkanGraphicsPipeline(const SCreationParams& params, const VkPipeline vk_pipeline) :
            IGPUGraphicsPipeline(params), m_vkPipeline(vk_pipeline) {}

        inline const void* getNativeHandle() const override {return &m_vkPipeline;}

        inline VkPipeline getInternalObject() const {return m_vkPipeline;}

    private:
        ~CVulkanGraphicsPipeline();

        const VkPipeline m_vkPipeline;
};

}

#endif
