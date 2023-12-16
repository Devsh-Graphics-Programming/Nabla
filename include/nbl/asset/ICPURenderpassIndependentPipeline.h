// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_

#include "nbl/asset/IRenderpassIndependentPipeline.h"
#include "nbl/asset/ICPUPipelineLayout.h"
#include "nbl/asset/ICPUShader.h"

namespace nbl::asset
{

//! CPU Version of Renderpass Independent Pipeline
/*
	@see IRenderpassIndependentPipeline
*/

class ICPURenderpassIndependentPipeline : public IRenderpassIndependentPipeline<ICPUShader>, public IAsset
{
		using base_t = IRenderpassIndependentPipeline<ICPUShader>;

	public:
		//(TODO) it is true however it causes DSs to not be cached when ECF_DONT_CACHE_TOP_LEVEL is set which isnt really intuitive
		constexpr static inline uint32_t DESC_SET_HIERARCHYLEVELS_BELOW = 0u;
		// TODO: @Crisspl HOW ON EARTH DOES THIS MAKE SENSE!?
		constexpr static inline uint32_t IMAGEVIEW_HIERARCHYLEVELS_BELOW = 1u;
		constexpr static inline uint32_t IMAGE_HIERARCHYLEVELS_BELOW = 2u;
		// from here its good
		constexpr static inline uint32_t PIPELINE_LAYOUT_HIERARCHYLEVELS_BELOW = 1u;
		constexpr static inline uint32_t DESC_SET_LAYOUT_HIERARCHYLEVELS_BELOW = 1u+ICPUPipelineLayout::DESC_SET_LAYOUT_HIERARCHYLEVELS_BELOW;
		constexpr static inline uint32_t IMMUTABLE_SAMPLER_HIERARCHYLEVELS_BELOW = 1u+ICPUPipelineLayout::IMMUTABLE_SAMPLER_HIERARCHYLEVELS_BELOW;
		constexpr static inline uint32_t SPECIALIZED_SHADER_HIERARCHYLEVELS_BELOW = 1u;

		using SCreationParams = base_t::SCreationParams;

		// IAsset implementations
		size_t conservativeSizeEstimate() const override { return sizeof(base_t); }
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			if (referenceLevelsBelowToConvert)
			{
                //intentionally parent is not converted
                --referenceLevelsBelowToConvert;
				m_layout->convertToDummyObject(referenceLevelsBelowToConvert);
				for (auto i=0u; i<GRAPHICS_SHADER_STAGE_COUNT; i++)
                if (m_shaders[i])
					m_shaders[i]->convertToDummyObject(referenceLevelsBelowToConvert);
			}
		}

        core::smart_refctd_ptr<IAsset> clone(const uint32_t _depth = ~0u) const override
        {
            core::smart_refctd_ptr<ICPUPipelineLayout> layout;
            if (_depth>0u && m_layout)
				layout = core::smart_refctd_ptr_static_cast<ICPUPipelineLayout>(m_layout->clone(_depth-1u));
			else
				layout = m_layout;

#if 0
            auto pShaders = &m_shaders->get();
			const SCreationParams params = {.shaders={const_cast<ICPUSpecializedShader**>(pShaders),GRAPHICS_SHADER_STAGE_COUNT},.cached=getCachedCreationParams()};
            auto cp = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(layout),params);
            clone_common(cp.get());
            return cp;
#endif
			return nullptr;
        }

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_RENDERPASS_INDEPENDENT_PIPELINE;
		inline E_TYPE getAssetType() const override { return AssetType; }

		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPURenderpassIndependentPipeline*>(_other);
			if (memcmp(&m_cachedParams,&other->m_cachedParams,sizeof(m_cachedParams))!=0)
				return false;

			for (uint32_t i = 0u; i < GRAPHICS_SHADER_STAGE_COUNT; ++i)
			{
				if ((!m_shaders[i]) != (!other->m_shaders[i]))
					return false;
				if (m_shaders[i] && !m_shaders[i]->canBeRestoredFrom(other->m_shaders[i].get()))
					return false;
			}
			if (!m_layout->canBeRestoredFrom(other->m_layout.get()))
				return false;

			return true;
		}

		//
		inline SCachedCreationParams& getCachedCreationParams()
		{
			assert(!isImmutable_debug());
			return m_cachedParams;
		}

		// extras for this class
		inline ICPUPipelineLayout* getLayout()
		{
			assert(!isImmutable_debug());
			return m_layout.get();
		}
		const inline ICPUPipelineLayout* getLayout() const { return m_layout.get(); }
		inline void setLayout(core::smart_refctd_ptr<ICPUPipelineLayout>&& _layout) 
		{
			assert(!isImmutable_debug());
			m_layout = std::move(_layout);
		}

#if 0
		inline const SpecShaderType* getShaderAtStage(IShader::E_SHADER_STAGE _stage) const
		{
			return m_shaders[core::findLSB<uint32_t>(_stage)].get();
		}
		inline const SpecShaderType* getShaderAtIndex(uint32_t _ix) const
		{
			return m_shaders[_ix].get();
		}
		inline ICPUSpecializedShader* getShaderAtStage(IShader::E_SHADER_STAGE _stage)
		{
			assert(!isImmutable_debug());
			return m_shaders[core::findLSB<uint32_t>(_stage)].get();
		}
		inline ICPUSpecializedShader* getShaderAtIndex(uint32_t _ix)
		{
			assert(!isImmutable_debug());
			return m_shaders[_ix].get();
		}

		inline void setShaderAtStage(IShader::E_SHADER_STAGE _stage, ICPUSpecializedShader* _shdr)
		{
			assert(!isImmutable_debug());
			m_shaders[core::findLSB<uint32_t>(_stage)] = core::smart_refctd_ptr<ICPUSpecializedShader>(_shdr);
		}
#endif
	protected:
		constexpr static inline uint32_t GRAPHICS_SHADER_STAGE_COUNT = 5;
		ICPURenderpassIndependentPipeline(core::smart_refctd_ptr<ICPUPipelineLayout>&& _layout, const SCreationParams& params) : base_t(params), m_layout(std::move(_layout))
		{
            for (const auto spec : params.shaders)
            if (spec.shader)
            {
#if 0
                const int32_t ix = core::findLSB<uint32_t>(shdr->getStage());
                assert(ix < static_cast<int32_t>(GRAPHICS_SHADER_STAGE_COUNT));
                assert(!m_shaders[ix]);//must be maximum of 1 for each stage
                m_shaders[ix] = core::smart_refctd_ptr<SpecShaderType>(shdr);
#endif
            }
		}
		virtual ~ICPURenderpassIndependentPipeline() = default;

		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPURenderpassIndependentPipeline*>(_other);

			if (_levelsBelow)
			{
				--_levelsBelow;

				restoreFromDummy_impl_call(m_layout.get(), other->m_layout.get(), _levelsBelow);
				for (uint32_t i = 0u; i < GRAPHICS_SHADER_STAGE_COUNT; ++i)
				if (m_shaders[i])
					restoreFromDummy_impl_call(m_shaders[i].get(), other->m_shaders[i].get(), _levelsBelow);
			}
		}

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			--_levelsBelow;
			if (m_layout->isAnyDependencyDummy(_levelsBelow))
				return true;
			for (auto& shader : m_shaders)
			if (shader && shader->isAnyDependencyDummy(_levelsBelow))
				return true;
			return false;
		}

		core::smart_refctd_ptr<ICPUPipelineLayout> m_layout;
		core::smart_refctd_ptr<ICPUShader> m_shaders[GRAPHICS_SHADER_STAGE_COUNT] = {};
#if 0
		SCreationParams m_params[GRAPHICS_SHADER_STAGE_COUNT] = {};
#endif
};

}
#endif
