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

class ICPUGraphicsPipeline final : public ICPUPipeline<IGraphicsPipeline<ICPUPipelineLayout,ICPUShader,ICPURenderpass>,5u>
{
        using pipeline_base_t = IGraphicsPipeline<ICPUPipelineLayout,ICPUShader,ICPURenderpass>;
        using base_t = ICPUPipeline<pipeline_base_t,5u>;

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
            if (!params.impl_valid([](const ICPUShader::SSpecInfo& info)->bool{return true;}))
                return nullptr;
            auto retval = new ICPUGraphicsPipeline(params);
            for (const auto spec : params.shaders)
            if (spec.shader)
				retval->setSpecInfo(spec);
            return core::smart_refctd_ptr<ICPUGraphicsPipeline>(retval,core::dont_grab);
		}

		constexpr static inline auto AssetType = ET_GRAPHICS_PIPELINE;
		inline E_TYPE getAssetType() const override { return AssetType; }

		// extras for this class
		inline const SCachedCreationParams& getCachedCreationParams() const {return base_t::getCachedCreationParams();}
        inline SCachedCreationParams& getCachedCreationParams()
        {
            assert(!isImmutable_debug());
            return m_params;
        }

    protected:
		using base_t::base_t;
        ~ICPUGraphicsPipeline() = default;

		base_t* clone_impl(core::smart_refctd_ptr<ICPUPipelineLayout>&& layout) const override
		{
			std::array<ICPUShader::SSpecInfo,GRAPHICS_SHADER_STAGE_COUNT> _shaders;
			for (auto i=0; i<GRAPHICS_SHADER_STAGE_COUNT; i++)
				_shaders[i] = m_stages[i].info;
			const SCreationParams params = {{
				.shaders = _shaders,
				.cached = m_params,
				.renderpass = m_renderpass.get()
			}};
			return new ICPUGraphicsPipeline(params);
		}
		bool canBeRestoredFrom_impl(const base_t* _other) const override
		{
			auto* other = static_cast<const ICPUGraphicsPipeline*>(_other);
			return memcmp(&m_params,&other->m_params,sizeof(m_params))==0;
		}

		inline int8_t stageToIndex(const ICPUShader::E_SHADER_STAGE stage) const override
		{
			const auto stageIx = hlsl::findLSB(stage);
			if (stageIx<0 || stageIx>=GRAPHICS_SHADER_STAGE_COUNT || hlsl::bitCount(stage)!=1)
				return -1;
			return stageIx;
		}
};

}

#endif