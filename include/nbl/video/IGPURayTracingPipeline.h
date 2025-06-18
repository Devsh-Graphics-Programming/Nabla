#ifndef _NBL_I_GPU_RAY_TRACING_PIPELINE_H_INCLUDED_
#define _NBL_I_GPU_RAY_TRACING_PIPELINE_H_INCLUDED_

#include "nbl/asset/IPipeline.h"
#include "nbl/asset/IRayTracingPipeline.h"

#include "nbl/video/SPipelineCreationParams.h"


namespace nbl::video
{

class IGPURayTracingPipeline :  public IGPUPipeline<asset::IRayTracingPipeline<const IGPUPipelineLayout>>
{
        using pipeline_t = asset::IRayTracingPipeline<const IGPUPipelineLayout>;

    public:
        struct SHitGroup
        {
            SShaderSpecInfo closestHit;
            SShaderSpecInfo anyHit;
            SShaderSpecInfo intersection;
        };

        struct SCreationParams : public SPipelineCreationParams<const IGPURayTracingPipeline>
        {
            using FLAGS = pipeline_t::FLAGS;

            struct SShaderGroupsParams
            {

                SShaderSpecInfo raygen;
                std::span<SShaderSpecInfo> misses;
                std::span<SHitGroup> hits;
                std::span<SShaderSpecInfo> callables;

                inline uint32_t getShaderGroupCount() const
                {
                    return 1 + hits.size() + misses.size() + callables.size();
                }

            };

            IGPUPipelineLayout* layout = nullptr;
            SShaderGroupsParams shaderGroups;

            SCachedCreationParams cached = {};
            // TODO: Could guess the required flags from SPIR-V introspection of declared caps
            core::bitflag<FLAGS> flags = FLAGS::NONE;

            inline SSpecializationValidationResult valid() const
            {
                if (!layout)
                  return {};

                SSpecializationValidationResult retval = {
                    .count = 0,
                    .dataSize = 0,
                };

                if (!shaderGroups.raygen.accumulateSpecializationValidationResult(&retval))
                    return {};

                for (const auto& shaderGroup : shaderGroups.hits)
                {
                    if (shaderGroup.intersection.shader) 
                    {
                      if (!shaderGroup.intersection.accumulateSpecializationValidationResult(&retval))
                        return {};
                    }

                    if (shaderGroup.closestHit.shader) 
                    {
                      if (!shaderGroup.closestHit.accumulateSpecializationValidationResult(&retval))
                        return {};
                    }

                    // https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-VkRayTracingPipelineCreateInfoKHR-flags-03470
                    if (flags.hasFlags(FLAGS::NO_NULL_ANY_HIT_SHADERS) && !shaderGroup.anyHit.shader)
                        return {};

                    if (shaderGroup.anyHit.shader) 
                    {
                      if (!shaderGroup.anyHit.accumulateSpecializationValidationResult(&retval))
                        return {};
                    }

                    // https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-VkRayTracingPipelineCreateInfoKHR-flags-03471
                    if (flags.hasFlags(FLAGS::NO_NULL_CLOSEST_HIT_SHADERS) && !shaderGroup.intersection.shader)
                        return {};
                }

                for (const auto& miss : shaderGroups.misses)
                {
                  if (miss.shader) 
                  {
                    if (!miss.accumulateSpecializationValidationResult(&retval))
                      return {};
                  }
                }

                for (const auto& callable : shaderGroups.callables)
                {
                  if (callable.shader)
                  {
                    if (!callable.accumulateSpecializationValidationResult(&retval))
                      return {};
                  }
                }

                // https://registry.khronos.org/vulkan/specs/latest/man/html/VkRayTracingPipelineCreateInfoKHR.html#VUID-VkRayTracingPipelineCreateInfoKHR-stage-03425
                if (!shaderGroups.raygen.shader) return {};

                return retval;
            }

            inline core::bitflag<hlsl::ShaderStage> getRequiredSubgroupStages() const
            {
                core::bitflag<hlsl::ShaderStage> stages = {};
                auto processSpecInfo = [&](const SShaderSpecInfo& spec, hlsl::ShaderStage stage)
                {
                    if (spec.shader && spec.requiredSubgroupSize >= SUBGROUP_SIZE::REQUIRE_4) {
                      stages |= stage;
                    }
                };
                processSpecInfo(shaderGroups.raygen, hlsl::ESS_RAYGEN);
                for (const auto& miss : shaderGroups.misses)
                    processSpecInfo(miss, hlsl::ESS_MISS);
                for (const auto& hit : shaderGroups.hits)
                {
                    processSpecInfo(hit.closestHit, hlsl::ESS_CLOSEST_HIT);
                    processSpecInfo(hit.anyHit, hlsl::ESS_ANY_HIT);
                    processSpecInfo(hit.intersection, hlsl::ESS_INTERSECTION);
                }
                for (const auto& callable : shaderGroups.callables)
                    processSpecInfo(callable, hlsl::ESS_CALLABLE);
                return stages;
            }

        };

        struct SShaderGroupHandle
        {
          private:
            uint8_t data[video::SPhysicalDeviceLimits::ShaderGroupHandleSize];
        };
        static_assert(sizeof(SShaderGroupHandle) == video::SPhysicalDeviceLimits::ShaderGroupHandleSize);

        struct SHitGroupStackSize
        {
            uint16_t closestHit;
            uint16_t anyHit;
            uint16_t intersection;
        };

        inline core::bitflag<SCreationParams::FLAGS> getCreationFlags() const { return m_flags; }

        // Vulkan: const VkPipeline*
        virtual const void* getNativeHandle() const = 0;

        virtual const SShaderGroupHandle& getRaygen() const = 0;
        virtual std::span<const SShaderGroupHandle> getMissHandles() const = 0;
        virtual std::span<const SShaderGroupHandle> getHitHandles() const = 0;
        virtual std::span<const SShaderGroupHandle> getCallableHandles() const = 0;

        virtual uint16_t getRaygenStackSize() const = 0;
        virtual std::span<const uint16_t> getMissStackSizes() const = 0;
        virtual std::span<const SHitGroupStackSize> getHitStackSizes() const = 0;
        virtual std::span<const uint16_t> getCallableStackSizes() const = 0;
        virtual uint16_t getDefaultStackSize() const = 0;

    protected:
        IGPURayTracingPipeline(const SCreationParams& params) : IGPUPipeline(core::smart_refctd_ptr<const ILogicalDevice>(params.layout->getOriginDevice()), params.layout, params.cached),
            m_flags(params.flags)
        {}

        virtual ~IGPURayTracingPipeline() = default;

        const core::bitflag<SCreationParams::FLAGS> m_flags;
};

}

#endif
