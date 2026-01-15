#ifndef _NBL_C_VULKAN_MESH_PIPELINE_H_INCLUDED_
#define _NBL_C_VULKAN_MESH_PIPELINE_H_INCLUDED_


#include "nbl/video/IGPUMeshPipeline.h"

#include <volk.h>

namespace nbl::video
{

    //potentially collapse this so Mesh just uses CVulkanGraphicsPipeline
    //if thats done, BindMesh can go away
class CVulkanMeshPipeline final : public IGPUMeshPipeline
{
    public:
        CVulkanMeshPipeline(const SCreationParams& params, const VkPipeline vk_pipeline) :
            IGPUMeshPipeline(params), m_vkPipeline(vk_pipeline) {}

        inline const void* getNativeHandle() const override {return &m_vkPipeline;}

        inline VkPipeline getInternalObject() const {return m_vkPipeline;}

        void setObjectDebugName(const char* label) const override; //exists in compute but not in graphics
    private:
        ~CVulkanMeshPipeline();

        const VkPipeline m_vkPipeline;
};

}

#endif