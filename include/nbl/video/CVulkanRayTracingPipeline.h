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
    using GeneralGroupStackSizeContainer = core::smart_refctd_dynamic_array<uint16_t>;
    using HitGroupStackSizeContainer = core::smart_refctd_dynamic_array<SHitGroupStackSize>;

  public:

    using ShaderGroupHandleContainer = core::smart_refctd_dynamic_array<SShaderGroupHandle>;

    CVulkanRayTracingPipeline(
      const SCreationParams& params, 
      const VkPipeline vk_pipeline, 
      ShaderGroupHandleContainer&& shaderGroupHandles);

    inline const void* getNativeHandle() const override { return &m_vkPipeline; }

    inline VkPipeline getInternalObject() const { return m_vkPipeline; }

    virtual const SShaderGroupHandle& getRaygen() const override;
    virtual std::span<const SShaderGroupHandle> getMissHandles() const override;
    virtual std::span<const SShaderGroupHandle> getHitHandles() const override;
    virtual std::span<const SShaderGroupHandle> getCallableHandles() const override;

    virtual uint16_t getRaygenStackSize() const override;
    virtual std::span<const uint16_t> getMissStackSizes() const override;
    virtual std::span<const SHitGroupStackSize> getHitStackSizes() const override;
    virtual std::span<const uint16_t> getCallableStackSizes() const override;
    virtual uint16_t getDefaultStackSize() const override;

  private:
    ~CVulkanRayTracingPipeline() override;

    const VkPipeline m_vkPipeline;
    ShaderContainer m_shaders;
    ShaderGroupHandleContainer m_shaderGroupHandles;
    uint16_t m_raygenStackSize;
    core::smart_refctd_dynamic_array<uint16_t> m_missStackSizes;
    core::smart_refctd_dynamic_array<SHitGroupStackSize> m_hitGroupStackSizes;
    core::smart_refctd_dynamic_array<uint16_t> m_callableStackSizes;

    uint32_t getRaygenIndex() const;
    uint32_t getMissBaseIndex() const;
    uint32_t getHitBaseIndex() const;
    uint32_t getCallableBaseIndex() const;
};

}

#endif
