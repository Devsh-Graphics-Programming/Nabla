#ifndef _NBL_ASSET_I_RAY_TRACING_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_RAY_TRACING_PIPELINE_H_INCLUDED_

#include "nbl/asset/IShader.h"
#include "nbl/asset/IPipeline.h"

#include <span>


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
        uint32_t intersectionShader = SIndex::Unused;
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
      uint32_t maxRecursionDepth;
    };
};

template<typename PipelineLayoutType, typename ShaderType>
class IRayTracingPipeline : public IPipeline<PipelineLayoutType>, public IRayTracingPipelineBase
{
  public:

    using SGeneralShaderGroupContainer = core::smart_refctd_dynamic_array<SGeneralShaderGroup>;
    using SHitShaderGroupContainer = core::smart_refctd_dynamic_array<SHitShaderGroup>;

    struct SCreationParams : IPipeline<PipelineLayoutType>::SCreationParams
    {
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
              if (stage > ICPUShader::E_SHADER_STAGE::ESS_CALLABLE || stage < ICPUShader::E_SHADER_STAGE::ESS_RAYGEN)
                return false;
              if (stage == ICPUShader::E_SHADER_STAGE::ESS_RAYGEN && stagePresence.hasFlags(hlsl::ESS_RAYGEN))
                return false;
              stagePresence |= stage;
            }

          auto getShaderStage = [this](size_t index) -> ICPUShader::E_SHADER_STAGE
            {
              return shaders[index].shader->getStage();
            };

          if (shaderGroups.raygen.index >= shaders.size())
            return false;
          if (getShaderStage(shaderGroups.raygen.index) != ICPUShader::E_SHADER_STAGE::ESS_RAYGEN)
            return false;

          auto isValidShaderIndex = [this, getShaderStage](size_t index, ICPUShader::E_SHADER_STAGE expectedStage) -> bool
            {
              if (index == SShaderGroupsParams::SIndex::Unused)
                return true;
              if (index >= shaders.size())
                return false;
              if (getShaderStage(index) != expectedStage)
                return false;
              return true;
            };

          for (const auto& shaderGroup : shaderGroups.hits)
          {
            if (!isValidShaderIndex(shaderGroup.anyHit, ICPUShader::E_SHADER_STAGE::ESS_ANY_HIT))
              return false;

            if (!isValidShaderIndex(shaderGroup.closestHit, ICPUShader::E_SHADER_STAGE::ESS_CLOSEST_HIT))
              return false;

            if (!isValidShaderIndex(shaderGroup.intersectionShader, ICPUShader::E_SHADER_STAGE::ESS_INTERSECTION))
              return false;
          }

          for (const auto& shaderGroup : shaderGroups.misses)
          {
            if (!isValidShaderIndex(shaderGroup.index, ICPUShader::E_SHADER_STAGE::ESS_MISS))
              return false;
          }

          for (const auto& shaderGroup : shaderGroups.callables)
          {
            if (!isValidShaderIndex(shaderGroup.index, ICPUShader::E_SHADER_STAGE::ESS_CALLABLE))
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
    };

    inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }
    size_t getHitGroupCount() const { return m_hitShaderGroups->size(); }
    size_t getMissGroupCount() const { return m_missShaderGroups->size(); }
    size_t getCallableGroupCount() const { return m_callableShaderGroups->size(); }

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
