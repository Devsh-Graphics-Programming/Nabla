#ifndef _NBL_C_VULKAN_RAY_TRACING_PIPELINE_H_INCLUDED_
#define _NBL_C_VULKAN_RAY_TRACING_PIPELINE_H_INCLUDED_


#include "nbl/video/IGPURayTracingPipeline.h"

#include "nbl/video/CVulkanShader.h"


namespace nbl::video
{

class CVulkanRayTracingPipeline final : public IGPURayTracingPipeline
{
    using ShaderRef = core::smart_refctd_ptr<const CVulkanShader>;
    using ShaderContainer = core::smart_refctd_dynamic_array<ShaderRef>;

  public:

    using ShaderGroupHandleContainer = core::smart_refctd_dynamic_array<SShaderGroupHandle>;

    CVulkanRayTracingPipeline(
      const SCreationParams& params, 
      const VkPipeline vk_pipeline, 
      ShaderGroupHandleContainer&& shaderGroupHandles);

    inline const void* getNativeHandle() const override { return &m_vkPipeline; }

    inline VkPipeline getInternalObject() const { return m_vkPipeline; }

    virtual const SShaderGroupHandle& getRaygen() const override;
    virtual const SShaderGroupHandle& getMiss(uint32_t index) const override;
    virtual const SShaderGroupHandle& getHit(uint32_t index) const override;
    virtual const SShaderGroupHandle& getCallable(uint32_t index) const override;

  private:
    ~CVulkanRayTracingPipeline() override;

    const VkPipeline m_vkPipeline;
    ShaderContainer m_shaders;
    ShaderGroupHandleContainer m_shaderGroupHandles;
};

}

#endif
