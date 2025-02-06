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

  struct SGeneralShaderGroup
  {
    uint32_t shaderIndex = ShaderUnused;
  };

  struct SHitShaderGroup
  {
    uint32_t closestHitShaderIndex = ShaderUnused;
    uint32_t anyHitShaderIndex = ShaderUnused;
    uint32_t intersectionShaderIndex = ShaderUnused;
  };

  SGeneralShaderGroup raygenGroup;
  std::span<SGeneralShaderGroup> missGroups;
  std::span<SHitShaderGroup> hitGroups;
  std::span<SGeneralShaderGroup> callableGroups;

  inline uint32_t getShaderGroupCount() const
  {
    return 1 + hitGroups.size() + missGroups.size() + callableGroups.size();
  }

};
using SGeneralShaderGroup = SShaderGroupsParams::SGeneralShaderGroup;
using SHitShaderGroup = SShaderGroupsParams::SHitShaderGroup;

class IRayTracingPipelineBase : public virtual core::IReferenceCounted
{
  public:
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

        if (shaderGroups.raygenGroup.shaderIndex >= shaders.size())
          return false;
        if (getShaderStage(shaderGroups.raygenGroup.shaderIndex) != ICPUShader::E_SHADER_STAGE::ESS_RAYGEN)
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

        for (const auto& shaderGroup : shaderGroups.hitGroups)
        {
          if (!isValidShaderIndex(shaderGroup.anyHitShaderIndex, ICPUShader::E_SHADER_STAGE::ESS_ANY_HIT))
            return false;

          if (!isValidShaderIndex(shaderGroup.closestHitShaderIndex, ICPUShader::E_SHADER_STAGE::ESS_CLOSEST_HIT))
            return false;

          if (!isValidShaderIndex(shaderGroup.intersectionShaderIndex, ICPUShader::E_SHADER_STAGE::ESS_INTERSECTION))
            return false;
        }

        for (const auto& shaderGroup : shaderGroups.missGroups)
        {
          if (!isValidShaderIndex(shaderGroup.shaderIndex, ICPUShader::E_SHADER_STAGE::ESS_MISS))
            return false;
        }

        for (const auto& shaderGroup : shaderGroups.callableGroups)
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
      SShaderGroupsParams shaderGroups;
      SCachedCreationParams cached = {};
    };

    inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }
    size_t getHitGroupCount() const { return m_hitShaderGroups->size(); }
    size_t getMissGroupCount() const { return m_missShaderGroups->size(); }
    size_t getCallableGroupCount() const { return m_callableShaderGroups->size(); }

  protected:
    explicit IRayTracingPipeline(const SCreationParams& _params) :
      IPipeline<PipelineLayoutType>(core::smart_refctd_ptr<const PipelineLayoutType>(_params.layout), EPBP_RAY_TRACING),
      m_params(_params.cached),
      m_raygenShaderGroup(_params.shaderGroups.raygenGroup),
      m_missShaderGroups(core::make_refctd_dynamic_array<SGeneralShaderGroupContainer>(_params.shaderGroups.missGroups)),
      m_hitShaderGroups(core::make_refctd_dynamic_array<SHitShaderGroupContainer>(_params.shaderGroups.hitGroups)),
      m_callableShaderGroups(core::make_refctd_dynamic_array<SGeneralShaderGroupContainer>(_params.shaderGroups.callableGroups))
    {}

    SCachedCreationParams m_params;
    SGeneralShaderGroup m_raygenShaderGroup;
    SGeneralShaderGroupContainer m_missShaderGroups;
    SHitShaderGroupContainer m_hitShaderGroups;
    SGeneralShaderGroupContainer m_callableShaderGroups;

};

}

#endif
