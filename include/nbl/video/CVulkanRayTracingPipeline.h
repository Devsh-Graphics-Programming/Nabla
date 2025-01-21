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
    using ShaderHandleContainer = core::smart_refctd_dynamic_array<uint8_t>;
  public:

    CVulkanRayTracingPipeline(const SCreationParams& params, const VkPipeline vk_pipeline);

    inline const void* getNativeHandle() const override { return &m_vkPipeline; }

    inline VkPipeline getInternalObject() const { return m_vkPipeline; }

    std::span<uint8_t> getRaygenGroupShaderHandle() const override;
    std::span<uint8_t> getHitGroupShaderHandle(uint32_t index) const override;
    std::span<uint8_t> getMissGroupShaderHandle(uint32_t index) const override;
    std::span<uint8_t> getCallableGroupShaderHandle(uint32_t index) const override;

  private:
    ~CVulkanRayTracingPipeline();

    const VkPipeline m_vkPipeline;
    ShaderContainer m_shaders;
    ShaderHandleContainer m_shaderGroupHandles;
  };

}

#endif
