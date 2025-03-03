#include "nbl/video/CVulkanRayTracingPipeline.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

  CVulkanRayTracingPipeline::CVulkanRayTracingPipeline(
    const SCreationParams& params, 
    const VkPipeline vk_pipeline, 
    ShaderGroupHandleContainer&& shaderGroupHandles) :
    IGPURayTracingPipeline(params),
    m_vkPipeline(vk_pipeline),
    m_shaders(core::make_refctd_dynamic_array<ShaderContainer>(params.shaders.size())),
    m_shaderGroupHandles(std::move(shaderGroupHandles))
  {
    for (size_t shaderIx = 0; shaderIx < params.shaders.size(); shaderIx++)
      m_shaders->operator[](shaderIx) = ShaderRef(static_cast<const CVulkanShader*>(params.shaders[shaderIx].shader));

  }

  CVulkanRayTracingPipeline::~CVulkanRayTracingPipeline()
  {
    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyPipeline(vulkanDevice->getInternalObject(), m_vkPipeline, nullptr);
  }


  const IGPURayTracingPipeline::SShaderGroupHandle& CVulkanRayTracingPipeline::getRaygen() const
  {
    return m_shaderGroupHandles->operator[](0);
  }

  const IGPURayTracingPipeline::SShaderGroupHandle& CVulkanRayTracingPipeline::getMiss(uint32_t index) const
  {
    const auto baseIndex = 1; // one raygen group before this groups
    return m_shaderGroupHandles->operator[](baseIndex + index);
  }

  const IGPURayTracingPipeline::SShaderGroupHandle& CVulkanRayTracingPipeline::getHit(uint32_t index) const
  {
    const auto baseIndex = 1 + getMissGroupCount(); // one raygen group + miss gropus before this groups
    return m_shaderGroupHandles->operator[](baseIndex + index);
  }

  const IGPURayTracingPipeline::SShaderGroupHandle& CVulkanRayTracingPipeline::getCallable(uint32_t index) const
  {
    const auto baseIndex = 1 + getMissGroupCount() + getHitGroupCount(); // one raygen group + miss groups + hit gropus before this groups
    return m_shaderGroupHandles->operator[](baseIndex + index);
  }
}
