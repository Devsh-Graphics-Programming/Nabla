#ifndef _NBL_ASSET_I_RAY_TRACING_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_RAY_TRACING_PIPELINE_H_INCLUDED_

#include "nbl/asset/IShader.h"
#include "nbl/asset/IPipeline.h"

#include <span>
#include <bit>
#include <type_traits>

namespace nbl::asset
{

class IRayTracingPipelineBase : public virtual core::IReferenceCounted
{
  public:
    struct SShaderGroupsParams
    {
      struct SIndex
      {
        constexpr static inline uint32_t Unused = 0xffFFffFFu;
        uint32_t index = Unused;
      };

      struct SHitGroup
      {
        uint32_t closestHit = SIndex::Unused;
        uint32_t anyHit = SIndex::Unused;
        uint32_t intersection = SIndex::Unused;
      };

      SIndex raygen;
      std::span<SIndex> misses;
      std::span<SHitGroup> hits;
      std::span<SIndex> callables;

      inline uint32_t getShaderGroupCount() const
      {
        return 1 + hits.size() + misses.size() + callables.size();
      }

    };
    using SGeneralShaderGroup = SShaderGroupsParams::SIndex;
    using SHitShaderGroup = SShaderGroupsParams::SHitGroup;

    struct SCachedCreationParams final
    {
      uint32_t maxRecursionDepth : 6 = 0;
      uint32_t dynamicStackSize : 1 = false;
    };
};

template<typename PipelineLayoutType, typename ShaderType>
class IRayTracingPipeline : public IPipeline<PipelineLayoutType>, public IRayTracingPipelineBase
{
    using base_creation_params_t = IPipeline<PipelineLayoutType>::SCreationParams;
  public:

    using SGeneralShaderGroupContainer = core::smart_refctd_dynamic_array<SGeneralShaderGroup>;
    using SHitShaderGroupContainer = core::smart_refctd_dynamic_array<SHitShaderGroup>;

    struct SCreationParams : base_creation_params_t
    {
      public:
      #define base_flag(F) static_cast<uint64_t>(base_creation_params_t::FLAGS::F)
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
        using SpecInfo = ShaderType::SSpecInfo;
        template<typename ExtraLambda>
        inline bool impl_valid(ExtraLambda&& extra) const
        {
          if (!IPipeline<PipelineLayoutType>::SCreationParams::layout)
            return false;

          core::bitflag<ICPUShader::E_SHADER_STAGE> stagePresence = {};
          for (const auto info : shaders)
            if (info.shader)
            {
              if (!extra(info))
                return false;
              const auto stage = info.shader->getStage();
              if ((stage & ~ICPUShader::E_SHADER_STAGE::ESS_ALL_RAY_TRACING)!=0)  
                return false;
              if (!std::has_single_bit<std::underlying_type_t<ICPUShader::E_SHADER_STAGE>>(stage))
                return false;
              stagePresence |= stage;
            }

          auto getShaderStage = [this](size_t index) -> ICPUShader::E_SHADER_STAGE
            {
              return shaders[index].shader->getStage();
            };

          auto isValidShaderIndex = [this, getShaderStage](size_t index, ICPUShader::E_SHADER_STAGE expectedStage, bool is_unused_shader_forbidden) -> bool
            {
              if (index == SShaderGroupsParams::SIndex::Unused)
                return !is_unused_shader_forbidden;
              if (index >= shaders.size())
                return false;
              if (getShaderStage(index) != expectedStage)
                return false;
              return true;
            };

          if (!isValidShaderIndex(shaderGroups.raygen.index, ICPUShader::E_SHADER_STAGE::ESS_RAYGEN, true))
          {
            return false;
          }

          for (const auto& shaderGroup : shaderGroups.hits)
          {
            // https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-VkRayTracingPipelineCreateInfoKHR-flags-03470
            if (!isValidShaderIndex(shaderGroup.anyHit, 
              ICPUShader::E_SHADER_STAGE::ESS_ANY_HIT,
              bool(flags & FLAGS::NO_NULL_ANY_HIT_SHADERS)))
              return false;

            // https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-VkRayTracingPipelineCreateInfoKHR-flags-03471
            if (!isValidShaderIndex(shaderGroup.closestHit, 
              ICPUShader::E_SHADER_STAGE::ESS_CLOSEST_HIT,
              bool(flags & FLAGS::NO_NULL_CLOSEST_HIT_SHADERS)))
              return false;

            if (!isValidShaderIndex(shaderGroup.intersection, 
              ICPUShader::E_SHADER_STAGE::ESS_INTERSECTION,
              false))
              return false;
          }

          for (const auto& shaderGroup : shaderGroups.misses)
          {
            if (!isValidShaderIndex(shaderGroup.index, 
              ICPUShader::E_SHADER_STAGE::ESS_MISS, 
              false))
              return false;
          }

          for (const auto& shaderGroup : shaderGroups.callables)
          {
            if (!isValidShaderIndex(shaderGroup.index, ICPUShader::E_SHADER_STAGE::ESS_CALLABLE, false))
              return false;
          }
          return true;
        }

      public:
        inline bool valid() const
        {
          return impl_valid([](const SpecInfo& info)->bool
          {
            if (!info.valid())
              return false;
            return false;
          });
        }

        std::span<const SpecInfo> shaders = {};
        SShaderGroupsParams shaderGroups;
        SCachedCreationParams cached = {};
        // TODO: Could guess the required flags from SPIR-V introspection of declared caps
        core::bitflag<FLAGS> flags = FLAGS::NONE;
    };

    inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }
    inline uint32_t getHitGroupCount() const { return m_hitShaderGroups->size(); }
    inline uint32_t getMissGroupCount() const { return m_missShaderGroups->size(); }
    inline uint32_t getCallableGroupCount() const { return m_callableShaderGroups->size(); }

  protected:
    explicit IRayTracingPipeline(const SCreationParams& _params) :
      IPipeline<PipelineLayoutType>(core::smart_refctd_ptr<const PipelineLayoutType>(_params.layout)),
      m_params(_params.cached),
      m_raygenShaderGroup(_params.shaderGroups.raygen),
      m_missShaderGroups(core::make_refctd_dynamic_array<SGeneralShaderGroupContainer>(_params.shaderGroups.misses)),
      m_hitShaderGroups(core::make_refctd_dynamic_array<SHitShaderGroupContainer>(_params.shaderGroups.hits)),
      m_callableShaderGroups(core::make_refctd_dynamic_array<SGeneralShaderGroupContainer>(_params.shaderGroups.callables))
    {}

    SCachedCreationParams m_params;
    SGeneralShaderGroup m_raygenShaderGroup;
    SGeneralShaderGroupContainer m_missShaderGroups;
    SHitShaderGroupContainer m_hitShaderGroups;
    SGeneralShaderGroupContainer m_callableShaderGroups;

};

}

#endif
