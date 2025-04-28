// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_I_CPU_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_I_CPU_GRAPHICS_PIPELINE_H_INCLUDED_


#include "nbl/asset/IGraphicsPipeline.h"
#include "nbl/asset/ICPURenderpass.h"
#include "nbl/asset/ICPUPipeline.h"


namespace nbl::asset
{

class ICPUGraphicsPipeline final : public ICPUPipeline<IGraphicsPipeline<IPipelineBase::SShaderSpecInfo<true>, ICPUPipelineLayout,ICPURenderpass>>
{
        using pipeline_base_t = IGraphicsPipeline<SShaderSpecInfo<true>,ICPUPipelineLayout, ICPURenderpass>;
        using base_t = ICPUPipeline<pipeline_base_t>;

    public:
		struct SCreationParams final : pipeline_base_t::SCreationParams
		{
			private:
				friend class ICPUGraphicsPipeline;
				template<typename ExtraLambda>
				inline bool impl_valid(ExtraLambda&& extra) const
				{
					return pipeline_base_t::SCreationParams::impl_valid(std::move(extra));
				}
		};

		static core::smart_refctd_ptr<ICPUGraphicsPipeline> create(const SCreationParams& params)
		{
			// we'll validate the specialization info later when attempting to set it
        if (!params.impl_valid([](const SShaderSpecInfo<true>& info)->bool{return true;}))
            return nullptr;
        auto retval = new ICPUGraphicsPipeline(params);
        for (const auto spec : params.shaders)
        {
            if (spec.shader) retval->setSpecInfo(spec);
        }
        return core::smart_refctd_ptr<ICPUGraphicsPipeline>(retval,core::dont_grab);
		}

    inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override final
    {
        core::smart_refctd_ptr<ICPUPipelineLayout> layout;
        if (_depth>0u && m_layout) 
					layout = core::smart_refctd_ptr_static_cast<ICPUPipelineLayout>(m_layout->clone(_depth-1u));

				auto* cp = [&] {
            std::array<SShaderSpecInfo<true>, GRAPHICS_SHADER_STAGE_COUNT> _shaders;
            for (auto i = 0; i < GRAPHICS_SHADER_STAGE_COUNT; i++)
              _shaders[i] = m_specInfos[i];
            const SCreationParams params = { {
              .shaders = _shaders,
              .cached = m_params,
              .renderpass = m_renderpass.get()
            } };
            return new ICPUGraphicsPipeline(params);
				}();
				for (auto specInfo : m_specInfos)
				{
            if (specInfo.shader)
            {
                auto newSpecInfo = specInfo;
                if (_depth>0u)
                {
                    newSpecInfo.shader = core::smart_refctd_ptr_static_cast<IShader>(specInfo.shader->clone(_depth-1u));
                }
                cp->setSpecInfo(newSpecInfo);
            }
				}

        return core::smart_refctd_ptr<ICPUGraphicsPipeline>(cp,core::dont_grab);
    }


		constexpr static inline auto AssetType = ET_GRAPHICS_PIPELINE;
		inline E_TYPE getAssetType() const override { return AssetType; }
		
		inline size_t getDependantCount() const override
		{
			auto stageCount = 2; // the layout and renderpass
			for (const auto& info : m_specInfos)
			{
        if (info.shader)
          stageCount++;
			}
			return stageCount;
		}

		// extras for this class
		inline const SCachedCreationParams& getCachedCreationParams() const {return base_t::getCachedCreationParams();}
        inline SCachedCreationParams& getCachedCreationParams()
        {
            assert(isMutable());
            return m_params;
        }

    protected:
		using base_t::base_t;
        ~ICPUGraphicsPipeline() = default;

		inline IAsset* getDependant_impl(const size_t ix) override
		{
			if (ix==0)
				return const_cast<ICPUPipelineLayout*>(m_layout.get());
			if (ix==1)
				return m_renderpass.get();
			size_t stageCount = 0;
			for (auto& specInfo : m_specInfos)
			{
        if (specInfo.shader)
        {
          if ((stageCount++)==ix-2)
            return specInfo.shader.get();
        }
			}
			return nullptr;
		}

		inline int8_t stageToIndex(const hlsl::ShaderStage stage) const
		{
			const auto stageIx = hlsl::findLSB(stage);
			if (stageIx<0 || stageIx>= GRAPHICS_SHADER_STAGE_COUNT || hlsl::bitCount(stage)!=1)
				return -1;
			return stageIx;
		}

    inline bool setSpecInfo(const SShaderSpecInfo<true>& info)
		{
			assert(isMutable());
      const auto specSize = info.valid();
      if (specSize<0) return false;
			const auto stage = info.stage;
			const auto stageIx = stageToIndex(stage);
			if (stageIx<0) return false;
			m_specInfos[stageIx] = info;
			return true;
		}

		SShaderSpecInfo<true> m_specInfos[GRAPHICS_SHADER_STAGE_COUNT];
};

}

#endif