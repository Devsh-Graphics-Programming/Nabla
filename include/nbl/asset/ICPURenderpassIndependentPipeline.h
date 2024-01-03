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

		static core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> create(core::smart_refctd_ptr<ICPUPipelineLayout>&& _layout, const SCreationParams& params)
		{
			// we'll validate the specialization info later when attempting to set it
            if (!_layout || params.shaders.empty())
                return nullptr;
            auto retval = new ICPURenderpassIndependentPipeline(std::move(_layout),params.cached);
            for (const auto spec : params.shaders)
            if (spec.shader)
				retval->setSpecInfo(spec);
            return core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>(retval,core::dont_grab);
		}

		// IAsset implementations
		size_t conservativeSizeEstimate() const override { return sizeof(base_t)+sizeof(m_layout)+sizeof(ICPUShader::SSpecInfo)*5; }
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			if (referenceLevelsBelowToConvert)
			{
                //intentionally parent is not converted
                --referenceLevelsBelowToConvert;
				m_layout->convertToDummyObject(referenceLevelsBelowToConvert);
				for (auto& shader : m_shaders)
                if (shader)
					shader->convertToDummyObject(referenceLevelsBelowToConvert);
			}
		}

        core::smart_refctd_ptr<IAsset> clone(const uint32_t _depth = ~0u) const override
        {
            core::smart_refctd_ptr<ICPUPipelineLayout> layout;
            if (_depth>0u && m_layout)
				layout = core::smart_refctd_ptr_static_cast<ICPUPipelineLayout>(m_layout->clone(_depth-1u));
			else
				layout = m_layout;
			
            auto cp = new ICPURenderpassIndependentPipeline(std::move(layout),m_cachedParams);
            for (const auto spec : m_infos)
            if (spec.shader)
				cp->setSpecInfo(spec);
            clone_common(cp);
			
            return core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>(cp,core::dont_grab);
        }

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_RENDERPASS_INDEPENDENT_PIPELINE;
		inline E_TYPE getAssetType() const override { return AssetType; }

		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPURenderpassIndependentPipeline*>(_other);
			if (memcmp(&m_cachedParams,&other->m_cachedParams,sizeof(m_cachedParams))!=0)
				return false;

			for (uint32_t i=0u; i<GRAPHICS_SHADER_STAGE_COUNT; ++i)
			{
				if (m_infos[i].equalAllButShader(other->m_infos[i]))
					return false;
				if (m_shaders[i] && !m_shaders[i]->canBeRestoredFrom(other->m_shaders[i].get()))
					return false;
			}
			if (!m_layout->canBeRestoredFrom(other->m_layout.get()))
				return false;

			return true;
		}

		//
		inline const SCachedCreationParams& getCachedCreationParams() const {return base_t::getCachedCreationParams();}
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

		// The getters are weird because the shader pointer needs patching
		inline IShader::SSpecInfo<ICPUShader> getSpecInfo(const ICPUShader::E_SHADER_STAGE stage)
		{
			assert(!isImmutable_debug());
			const auto stageIx = hlsl::findLSB(stage);
			if (stageIx<0 || stageIx>=GRAPHICS_SHADER_STAGE_COUNT || hlsl::bitCount(stage)!=1)
				return {};
			return m_infos[stageIx];
		}
		inline IShader::SSpecInfo<const ICPUShader> getSpecInfo(const ICPUShader::E_SHADER_STAGE stage) const
		{
			const auto stageIx = hlsl::findLSB(stage);
			if (stageIx<0 || stageIx>=GRAPHICS_SHADER_STAGE_COUNT || hlsl::bitCount(stage)!=1)
				return {};
			return m_infos[stageIx];
		}
		inline bool setSpecInfo(const IShader::SSpecInfo<ICPUShader>& info)
		{
			assert(!isImmutable_debug());
            const int64_t specSize = info.valid();
            if (specSize<0)
                return false;
			const auto stage = info.shader->getStage();
			const auto stageIx = hlsl::findLSB(stage);
			if (stageIx<0 || stageIx>=GRAPHICS_SHADER_STAGE_COUNT || hlsl::bitCount(stage)!=1)
				return false;
			m_infos[stageIx] = info;
			m_shaders[stageIx] = core::smart_refctd_ptr<ICPUShader>(info.shader);
			m_infos[stageIx].shader = m_shaders[stageIx].get();
            if (specSize>0)
            {
                m_entries[stageIx] = std::make_unique<ICPUShader::SSpecInfo::spec_constant_map_t>();
                m_entries[stageIx]->reserve(info.entries->size());
                std::copy(info.entries->begin(),info.entries->end(),std::insert_iterator(*m_entries[stageIx],m_entries[stageIx]->begin()));
            }
            else
                m_entries[stageIx] = nullptr;
			m_infos[stageIx].entries = m_entries[stageIx].get();
			return true;
		}

	protected:
		ICPURenderpassIndependentPipeline(core::smart_refctd_ptr<ICPUPipelineLayout>&& _layout, const base_t::SCachedCreationParams& params)
			: base_t(params), m_layout(std::move(_layout)) {}
		virtual ~ICPURenderpassIndependentPipeline() = default;

		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPURenderpassIndependentPipeline*>(_other);

			if (_levelsBelow)
			{
				--_levelsBelow;

				restoreFromDummy_impl_call(m_layout.get(), other->m_layout.get(), _levelsBelow);
				for (uint32_t i=0u; i<GRAPHICS_SHADER_STAGE_COUNT; ++i)
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
		std::array<core::smart_refctd_ptr<ICPUShader>,GRAPHICS_SHADER_STAGE_COUNT> m_shaders = {};
		std::array<std::unique_ptr<ICPUShader::SSpecInfo::spec_constant_map_t>,GRAPHICS_SHADER_STAGE_COUNT> m_entries = {};
		std::array<ICPUShader::SSpecInfo,GRAPHICS_SHADER_STAGE_COUNT> m_infos = {};
};

}
#endif
