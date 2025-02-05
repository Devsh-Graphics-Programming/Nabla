#include "nbl/video/CVulkanRayTracingPipeline.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

  CVulkanRayTracingPipeline::CVulkanRayTracingPipeline(const SCreationParams& params, const VkPipeline vk_pipeline) :
    IGPURayTracingPipeline(params),
    m_vkPipeline(vk_pipeline),
    m_shaders(core::make_refctd_dynamic_array<ShaderContainer>(params.shaders.size()))
  {
    for (size_t shaderIx = 0; shaderIx < params.shaders.size(); shaderIx++)
      m_shaders->operator[](shaderIx) = ShaderRef(static_cast<const CVulkanShader*>(params.shaders[shaderIx].shader));

    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    const auto handleCount = params.cached.shaderGroups.getShaderGroupCount();
    const auto handleSize = SPhysicalDeviceLimits::ShaderGroupHandleSize;
    const auto dataSize = handleCount * handleSize;
    auto* vk = vulkanDevice->getFunctionTable();
    m_shaderGroupHandles = core::make_refctd_dynamic_array<ShaderHandleContainer>(dataSize);
    vk->vk.vkGetRayTracingShaderGroupHandlesKHR(vulkanDevice->getInternalObject(), m_vkPipeline, 0, handleCount, dataSize, m_shaderGroupHandles->data());
  }

  CVulkanRayTracingPipeline::~CVulkanRayTracingPipeline()
  {
    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyPipeline(vulkanDevice->getInternalObject(), m_vkPipeline, nullptr);
  }

  std::span<uint8_t> CVulkanRayTracingPipeline::getRaygenGroupShaderHandle() const
  {
    const auto handleSize = SPhysicalDeviceLimits::ShaderGroupHandleSize;
    return {m_shaderGroupHandles->data(), handleSize};
  }

  std::span<uint8_t> CVulkanRayTracingPipeline::getMissGroupShaderHandle(uint32_t index) const
  {
    const auto handleSize = SPhysicalDeviceLimits::ShaderGroupHandleSize;
    const auto baseOffset = handleSize; // one raygen this group
    return {m_shaderGroupHandles->data() + baseOffset + index * handleSize, handleSize};
  }

  std::span<uint8_t> CVulkanRayTracingPipeline::getHitGroupShaderHandle(uint32_t index) const
  {
    const auto handleSize = SPhysicalDeviceLimits::ShaderGroupHandleSize;
    const auto baseOffset = handleSize + getMissGroupCount() * handleSize; // one raygen + miss groups handle before this group
    return {m_shaderGroupHandles->data() + baseOffset + index * handleSize, handleSize};
  }

  std::span<uint8_t> CVulkanRayTracingPipeline::getCallableGroupShaderHandle(uint32_t index) const
  {
    const auto handleSize = SPhysicalDeviceLimits::ShaderGroupHandleSize;

    // one raygen + hit groups  + miss groups handle before this group
    const auto baseOffset = handleSize + getMissGroupCount() * handleSize + getHitGroupCount() * handleSize;

    return {m_shaderGroupHandles->data() + baseOffset + index * handleSize, handleSize};
  }
}
