#ifndef _NBL_ASSET_I_RAY_TRACING_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_RAY_TRACING_PIPELINE_H_INCLUDED_


#include "nbl/asset/IShader.h"
#include "nbl/asset/IPipeline.h"
#include "nbl/asset/IRenderpass.h"

#include <span>


namespace nbl::asset
{
  struct SShaderGroupsParams
  {
    constexpr static inline uint32_t ShaderUnused = 0xffFFffFFu;

    struct SGeneralGroup
    {
      uint32_t shaderIndex = ShaderUnused;
    };

    struct SHitGroup
    {
      uint32_t closestHitShaderIndex = ShaderUnused;
      uint32_t anyHitShaderIndex = ShaderUnused;
      uint32_t intersectionShaderIndex = ShaderUnused;
    };

    SGeneralGroup raygenGroup;
    core::vector<SGeneralGroup> missGroups;
    core::vector<SHitGroup> hitGroups;
    core::vector<SGeneralGroup> callableGroups;

    inline uint32_t getShaderGroupCount() const
    {
      return 1 + hitGroups.size() + missGroups.size() + callableGroups.size();
    }

  };
  using SGeneralShaderGroup = SShaderGroupsParams::SGeneralGroup;
  using SHitShaderGroup = SShaderGroupsParams::SHitGroup;

  class IRayTracingPipelineBase : public virtual core::IReferenceCounted
  {
  public:

    struct SCachedCreationParams final
    {
      SShaderGroupsParams shaderGroups;
      uint32_t maxRecursionDepth;
    };
  };

  template<typename PipelineLayoutType, typename ShaderType>
  class IRayTracingPipeline : public IPipeline<PipelineLayoutType>, public IRayTracingPipelineBase
  {
  public:
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

        if (cached.shaderGroups.raygenGroup.shaderIndex >= shaders.size())
          return false;
        if (getShaderStage(cached.shaderGroups.raygenGroup.shaderIndex) != ICPUShader::E_SHADER_STAGE::ESS_RAYGEN)
          return false;

        auto isValidShaderIndex = [this, getShaderStage](size_t index, ICPUShader::E_SHADER_STAGE expectedStage) -> bool
          {
            if (index == SShaderGroupsParams::ShaderUnused)
              return true;
            if (index >= shaders.size())
              return false;
            if (getShaderStage(index) != expectedStage)
              return false;
            return true;
          };

        for (const auto& shaderGroup : cached.shaderGroups.hitGroups)
        {
          if (!isValidShaderIndex(shaderGroup.anyHitShaderIndex, ICPUShader::E_SHADER_STAGE::ESS_ANY_HIT))
            return false;

          if (!isValidShaderIndex(shaderGroup.closestHitShaderIndex, ICPUShader::E_SHADER_STAGE::ESS_CLOSEST_HIT))
            return false;

          if (!isValidShaderIndex(shaderGroup.intersectionShaderIndex, ICPUShader::E_SHADER_STAGE::ESS_INTERSECTION))
            return false;
        }

        for (const auto& shaderGroup : cached.shaderGroups.missGroups)
        {
          if (!isValidShaderIndex(shaderGroup.shaderIndex, ICPUShader::E_SHADER_STAGE::ESS_MISS))
            return false;
        }

        for (const auto& shaderGroup : cached.shaderGroups.callableGroups)
        {
          if (!isValidShaderIndex(shaderGroup.shaderIndex, ICPUShader::E_SHADER_STAGE::ESS_CALLABLE))
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
      SCachedCreationParams cached = {};
    };

    inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }
    size_t getHitGroupCount() const { return m_params.shaderGroups.hitGroups.size(); }
    size_t getMissGroupCount() const { return m_params.shaderGroups.missGroups.size(); }
    size_t getCallableGroupCount() const { return m_params.shaderGroups.callableGroups.size(); }

  protected:
    explicit IRayTracingPipeline(const SCreationParams& _params) :
      IPipeline<PipelineLayoutType>(core::smart_refctd_ptr<const PipelineLayoutType>(_params.layout)),
      m_params(_params.cached) {
    }

    SCachedCreationParams m_params;
  };

}

#endif
