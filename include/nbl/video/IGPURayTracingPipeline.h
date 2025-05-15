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
        struct SCreationParams
        {
          #define base_flag(F) static_cast<uint64_t>(IPipelineBase::FLAGS::F)
          enum class FLAGS : uint64_t
          {
              NONE = base_flag(NONE),
              DISABLE_OPTIMIZATIONS = base_flag(DISABLE_OPTIMIZATIONS),
              ALLOW_DERIVATIVES = base_flag(ALLOW_DERIVATIVES),
              FAIL_ON_PIPELINE_COMPILE_REQUIRED = base_flag(FAIL_ON_PIPELINE_COMPILE_REQUIRED),
              EARLY_RETURN_ON_FAILURE = base_flag(EARLY_RETURN_ON_FAILURE),
              SKIP_BUILT_IN_PRIMITIVES = 1<<12,
              SKIP_AABBS = 1<<13,
              NO_NULL_ANY_HIT_SHADERS = 1<<14,
              NO_NULL_CLOSEST_HIT_SHADERS = 1<<15,
              NO_NULL_MISS_SHADERS = 1<<16,
              NO_NULL_INTERSECTION_SHADERS = 1<<17,
              ALLOW_MOTION = 1<<20,
          };
          #undef base_flag

          protected:
              template<typename ExtraLambda>
              inline bool impl_valid(ExtraLambda&& extra) const
              {
                  if (!m_layout) return false;

                  for (const auto info : shaders)
                  {
                      if (info.shader)
                      {
                          if (!extra(info))
                            return false;
                          const auto stage = info.stage;
                          if ((stage & ~hlsl::ShaderStage::ESS_ALL_RAY_TRACING) != 0)
                            return false;
                          if (!std::has_single_bit<std::underlying_type_t<hlsl::ShaderStage>>(stage))
                            return false;
                      }
                      else
                      {
                          // every shader must not be null. use SIndex::Unused to represent unused shader.
                          return false;
                      }
                  }

                  auto getShaderStage = [this](size_t index) -> hlsl::ShaderStage
                  {
                      return shaders[index].stage;
                  };

                auto isValidShaderIndex = [this, getShaderStage](size_t index, hlsl::ShaderStage expectedStage, bool is_unused_shader_forbidden) -> bool
                  {
                    if (index == SShaderGroupsParams::SIndex::Unused)
                      return !is_unused_shader_forbidden;
                    if (index >= shaders.size())
                      return false;
                    if (getShaderStage(index) != expectedStage)
                      return false;
                    return true;
                  };

                if (!isValidShaderIndex(shaderGroups.raygen.index, hlsl::ShaderStage::ESS_RAYGEN, true))
                {
                  return false;
                }

                for (const auto& shaderGroup : shaderGroups.hits)
                {
                  // https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-VkRayTracingPipelineCreateInfoKHR-flags-03470
                  if (!isValidShaderIndex(shaderGroup.anyHit, 
                    hlsl::ShaderStage::ESS_ANY_HIT,
                    bool(flags & FLAGS::NO_NULL_ANY_HIT_SHADERS)))
                    return false;

                  // https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-VkRayTracingPipelineCreateInfoKHR-flags-03471
                  if (!isValidShaderIndex(shaderGroup.closestHit, 
                    hlsl::ShaderStage::ESS_CLOSEST_HIT,
                    bool(flags & FLAGS::NO_NULL_CLOSEST_HIT_SHADERS)))
                    return false;

                  if (!isValidShaderIndex(shaderGroup.intersection, 
                    hlsl::ShaderStage::ESS_INTERSECTION,
                    false))
                    return false;
                }

                for (const auto& shaderGroup : shaderGroups.misses)
                {
                  if (!isValidShaderIndex(shaderGroup.index, 
                    hlsl::ShaderStage::ESS_MISS, 
                    false))
                    return false;
                }

                for (const auto& shaderGroup : shaderGroups.callables)
                {
                  if (!isValidShaderIndex(shaderGroup.index, hlsl::ShaderStage::ESS_CALLABLE, false))
                    return false;
                }
                return true;
              }

          public:
            inline bool valid() const
            {
              return impl_valid([](const SShaderSpecInfo& info)->bool
              {
                if (!info.valid())
                  return false;
                return false;
              });
            }

            struct SShaderGroupsParams
            {
                struct SHitGroup
                {
                    SShaderSpecInfo closestHit;
                    SShaderSpecInfo anyHit;
                    SShaderSpecInfo intersection;
                };

                SShaderSpecInfo raygen;
                std::span<SShaderSpecInfo> misses;
                std::span<SHitGroup> hits;
                std::span<SShaderSpecInfo> callables;

                inline uint32_t getShaderGroupCount() const
                {
                    return 1 + hits.size() + misses.size() + callables.size();
                }

            };

            SShaderGroupsParams shaderGroups;

            SCachedCreationParams cached = {};
            // TODO: Could guess the required flags from SPIR-V introspection of declared caps
            core::bitflag<FLAGS> flags = FLAGS::NONE;
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

        using SGeneralShaderGroupContainer = core::smart_refctd_dynamic_array<SGeneralShaderGroup>;
        using SHitShaderGroupContainer = core::smart_refctd_dynamic_array<SHitShaderGroup>;

        struct SCreationParams final : SPipelineCreationParams<const IGPURayTracingPipeline>
        {
            #define base_flag(F) static_cast<uint64_t>(IPipelineBase::CreationFlags::F)
            enum class FLAGS : uint64_t
            {
                NONE = base_flag(NONE),
                DISABLE_OPTIMIZATIONS = base_flag(DISABLE_OPTIMIZATIONS),
                ALLOW_DERIVATIVES = base_flag(ALLOW_DERIVATIVES),
                FAIL_ON_PIPELINE_COMPILE_REQUIRED = base_flag(FAIL_ON_PIPELINE_COMPILE_REQUIRED),
                EARLY_RETURN_ON_FAILURE = base_flag(EARLY_RETURN_ON_FAILURE),
                SKIP_BUILT_IN_PRIMITIVES = 1<<12,
                SKIP_AABBS = 1<<13,
                NO_NULL_ANY_HIT_SHADERS = 1<<14,
                NO_NULL_CLOSEST_HIT_SHADERS = 1<<15,
                NO_NULL_MISS_SHADERS = 1<<16,
                NO_NULL_INTERSECTION_SHADERS = 1<<17,
                ALLOW_MOTION = 1<<20,
            };
            #undef base_flag

            inline SSpecializationValidationResult valid() const
            {
                if (!layout)
                  return {};

                SSpecializationValidationResult retval = {
                    .count=0,
                    .dataSize=0,
                };
                const bool valid = pipeline_t::SCreationParams::impl_valid([&retval](const SShaderSpecInfo& info)->bool
                {
                    const auto dataSize = info.valid();
                    if (dataSize<0)
                        return false;
                    else if (dataSize==0)
                        return true;

                    const size_t count = info.entries ? info.entries->size():0x80000000ull;
                    if (count>0x7fffffff)
                        return {};
                    retval += {.count=dataSize ? static_cast<uint32_t>(count):0,.dataSize=static_cast<uint32_t>(dataSize)};
                    return retval;
                });
                if (!valid)
                    return {};
                return retval;
            }

            inline std::span<const SShaderSpecInfo> getShaders() const { return shaders; }

            IGPUPipelineLayout* layout = nullptr;
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
        IGPURayTracingPipeline(const SCreationParams& params) : IGPUPipeline(core::smart_refctd_ptr<const ILogicalDevice>(params.layout->getOriginDevice()), params),
            m_flags(params.flags)
        {}

        virtual ~IGPURayTracingPipeline() = default;

        const core::bitflag<SCreationParams::FLAGS> m_flags;
};

}

#endif
