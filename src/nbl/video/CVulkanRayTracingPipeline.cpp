#include "nbl/asset/IRayTracingPipeline.h"

#include "nbl/video/CVulkanRayTracingPipeline.h"
#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/IGPURayTracingPipeline.h"

#include <span>

namespace nbl::video
{

  CVulkanRayTracingPipeline::CVulkanRayTracingPipeline(
    const SCreationParams& params, 
    const VkPipeline vk_pipeline, 
    ShaderGroupHandleContainer&& shaderGroupHandles) :
    IGPURayTracingPipeline(params),
    m_vkPipeline(vk_pipeline),
    m_shaders(core::make_refctd_dynamic_array<ShaderContainer>(params.shaders.size())),
    m_missStackSizes(core::make_refctd_dynamic_array<GeneralGroupStackSizeContainer>(params.shaderGroups.misses.size())),
    m_hitGroupStackSizes(core::make_refctd_dynamic_array<HitGroupStackSizeContainer>(params.shaderGroups.hits.size())),
    m_callableStackSizes(core::make_refctd_dynamic_array<GeneralGroupStackSizeContainer>(params.shaderGroups.hits.size())),
    m_shaderGroupHandles(std::move(shaderGroupHandles))
  {
    for (size_t shaderIx = 0; shaderIx < params.shaders.size(); shaderIx++)
      m_shaders->operator[](shaderIx) = ShaderRef(static_cast<const CVulkanShader*>(params.shaders[shaderIx].shader));

    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();

    auto getVkShaderGroupStackSize = [&](uint32_t baseGroupIx, uint32_t shaderGroupIx, uint32_t shaderIx, VkShaderGroupShaderKHR shaderType) -> uint16_t
    {
      if (shaderIx == SShaderGroupsParams::SIndex::Unused)
        return 0;

      return vk->vk.vkGetRayTracingShaderGroupStackSizeKHR(
        vulkanDevice->getInternalObject(),
        m_vkPipeline,
        baseGroupIx + shaderGroupIx,
        shaderType
      );
    };

    m_raygenStackSize = getVkShaderGroupStackSize(getRaygenIndex(), 0, params.shaderGroups.raygen.index, VK_SHADER_GROUP_SHADER_GENERAL_KHR);

    for (size_t shaderGroupIx = 0; shaderGroupIx < params.shaderGroups.misses.size(); shaderGroupIx++)
    {
      m_missStackSizes->operator[](shaderGroupIx) = getVkShaderGroupStackSize(
        getMissBaseIndex(), 
        shaderGroupIx, 
        params.shaderGroups.misses[shaderGroupIx].index,
        VK_SHADER_GROUP_SHADER_GENERAL_KHR);
    }

    for (size_t shaderGroupIx = 0; shaderGroupIx < params.shaderGroups.hits.size(); shaderGroupIx++)
    {
      const auto& hitGroup = params.shaderGroups.hits[shaderGroupIx];
      const auto baseIndex = getHitBaseIndex();
      m_hitGroupStackSizes->operator[](shaderGroupIx) = SHitGroupStackSize{
        .closestHit = getVkShaderGroupStackSize(baseIndex,shaderGroupIx, hitGroup.closestHit, VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR),
        .anyHit = getVkShaderGroupStackSize(baseIndex, shaderGroupIx, hitGroup.anyHit,VK_SHADER_GROUP_SHADER_ANY_HIT_KHR),
        .intersection = getVkShaderGroupStackSize(baseIndex, shaderGroupIx, hitGroup.intersection, VK_SHADER_GROUP_SHADER_INTERSECTION_KHR),
      };
    }

    for (size_t shaderGroupIx = 0; shaderGroupIx < params.shaderGroups.callables.size(); shaderGroupIx++)
    {
      m_callableStackSizes->operator[](shaderGroupIx) = getVkShaderGroupStackSize(
        getCallableBaseIndex(), 
        shaderGroupIx, 
        params.shaderGroups.callables[shaderGroupIx].index,
        VK_SHADER_GROUP_SHADER_GENERAL_KHR);
    }
  }

  CVulkanRayTracingPipeline::~CVulkanRayTracingPipeline()
  {
    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyPipeline(vulkanDevice->getInternalObject(), m_vkPipeline, nullptr);
  }

  const IGPURayTracingPipeline::SShaderGroupHandle& CVulkanRayTracingPipeline::getRaygen() const
  {
  return m_shaderGroupHandles->operator[](getRaygenIndex());
  }

  std::span<const IGPURayTracingPipeline::SShaderGroupHandle> CVulkanRayTracingPipeline::getMissHandles() const
  {
    const auto baseIndex = getMissBaseIndex();
    return std::span(m_shaderGroupHandles->begin() + baseIndex, m_missShaderGroups->size());
  }

  std::span<const IGPURayTracingPipeline::SShaderGroupHandle> CVulkanRayTracingPipeline::getHitHandles() const
  {
    const auto baseIndex = getHitBaseIndex();
    return std::span(m_shaderGroupHandles->begin() + baseIndex, m_hitShaderGroups->size());
  }

  std::span<const IGPURayTracingPipeline::SShaderGroupHandle> CVulkanRayTracingPipeline::getCallableHandles() const
  {
    const auto baseIndex = getCallableBaseIndex();
    return std::span(m_shaderGroupHandles->begin() + baseIndex, m_callableShaderGroups->size());
  }

  uint16_t CVulkanRayTracingPipeline::getRaygenStackSize() const
  {
    return m_raygenStackSize;
  }

  std::span<const uint16_t> CVulkanRayTracingPipeline::getMissStackSizes() const
  {
    return std::span(m_missStackSizes->begin(), m_missStackSizes->end());
  }

  std::span<const IGPURayTracingPipeline::SHitGroupStackSize> CVulkanRayTracingPipeline::getHitStackSizes() const
  {
    return std::span(m_hitGroupStackSizes->begin(), m_hitGroupStackSizes->end());
  }

  std::span<const uint16_t> CVulkanRayTracingPipeline::getCallableStackSizes() const
  {
    return std::span(m_callableStackSizes->begin(), m_callableStackSizes->end());
  }

  uint16_t CVulkanRayTracingPipeline::getDefaultStackSize() const
  {
    // calculation follow the formula from
    // https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#ray-tracing-pipeline-stack
    const auto raygenStackMax = m_raygenStackSize;

    auto getMaxSize = [&](auto ranges, auto valProj) -> uint16_t
      {
        auto maxValue = 0;
        for (const auto& val : ranges)
        {
          maxValue = std::max<uint16_t>(maxValue, std::invoke(valProj, val));
        }
        return maxValue;
      };

    const auto closestHitStackMax = getMaxSize(getHitStackSizes(), &SHitGroupStackSize::closestHit);
    const auto anyHitStackMax = getMaxSize(getHitStackSizes(), &SHitGroupStackSize::anyHit);
    const auto intersectionStackMax = getMaxSize(getHitStackSizes(), &SHitGroupStackSize::intersection);
    const auto missStackMax = getMaxSize(getMissStackSizes(), std::identity{});
    const auto callableStackMax = getMaxSize(getCallableStackSizes(), std::identity{});
    return raygenStackMax + std::min<uint16_t>(1, m_params.maxRecursionDepth) *
      std::max(closestHitStackMax, std::max<uint16_t>(missStackMax, intersectionStackMax + anyHitStackMax)) +
      std::max<uint16_t>(0, m_params.maxRecursionDepth - 1) * std::max(closestHitStackMax, missStackMax) + 2 *
      callableStackMax;
  }

  uint32_t CVulkanRayTracingPipeline::getRaygenIndex() const
  {
    return 0;
  }

  uint32_t CVulkanRayTracingPipeline::getMissBaseIndex() const
  {
   // one raygen group before this groups
    return 1;
  }

  uint32_t CVulkanRayTracingPipeline::getHitBaseIndex() const
  {
    // one raygen group + miss groups before this groups
    return 1 + m_missShaderGroups->size();
  }

  uint32_t CVulkanRayTracingPipeline::getCallableBaseIndex() const
  {
    // one raygen group + miss groups + hit groups before this groups
    return 1 + m_missShaderGroups->size() + m_hitShaderGroups->size();
  }

}
