// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_PIPELINE_H_INCLUDED_


#include "nbl/asset/IAsset.h"
#include "nbl/asset/IPipeline.h"
#include "nbl/asset/ICPUPipelineLayout.h"
#include "nbl/asset/ICPUShader.h"


namespace nbl::asset
{

// Common Base class for pipelines
template<typename PipelineNonAssetBase, uint8_t MaxShaderStageCount>
class ICPUPipeline : public IAsset, public PipelineNonAssetBase
{
        using this_t = ICPUPipeline<PipelineNonAssetBase,MaxShaderStageCount>;

    public:
        inline size_t conservativeSizeEstimate() const override final {return sizeof(PipelineNonAssetBase)+sizeof(ICPUShader*)*MaxShaderStageCount;}

        inline void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override final
	    {
            convertToDummyObject_common(referenceLevelsBelowToConvert);

		    if (referenceLevelsBelowToConvert)
		    {
                //intentionally parent is not converted
                --referenceLevelsBelowToConvert;
                PipelineNonAssetBase::m_layout->convertToDummyObject(referenceLevelsBelowToConvert);
                for (auto i=0; i<MaxShaderStageCount; i++)
                if (m_stages[i].shader)
                    m_stages[i].shader->convertToDummyObject(referenceLevelsBelowToConvert);
		    }
	    }

        inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override final
        {
            core::smart_refctd_ptr<ICPUPipelineLayout> layout;
            if (_depth>0u && PipelineNonAssetBase::m_layout)
				layout = core::smart_refctd_ptr_static_cast<ICPUPipelineLayout>(PipelineNonAssetBase::m_layout->clone(_depth-1u));

            auto cp = clone_impl(std::move(layout));
            for (auto i=0; i<MaxShaderStageCount; i++)
            {
                const auto shader = m_stages[i].shader;
                if (shader)
                {
                    auto stageInfo = m_stages[i].info;
                    core::smart_refctd_ptr<ICPUShader> newShader;
                    if (_depth>0u)
                    {
                        newShader = core::smart_refctd_ptr_static_cast<ICPUShader>(shader->clone(_depth-1u));
                        stageInfo.shader = newShader.get();
                    }
                    cp->setSpecInfo(stageInfo);
                }
            }
            clone_common(cp);

            return core::smart_refctd_ptr<this_t>(cp,core::dont_grab);
        }

        bool canBeRestoredFrom(const IAsset* _other) const override final
        {
            auto* other = static_cast<const this_t*>(_other);
            if (!PipelineNonAssetBase::m_layout->canBeRestoredFrom(other->m_layout.get()))
                return false;
            for (auto i=0; i<MaxShaderStageCount; i++)
            {
                const auto& stage = m_stages[i];
                const auto& otherStage = other->m_stages[i];
                if (stage.shader)
                {
                    if (!stage.shader->canBeRestoredFrom(otherStage.shader.get()))
                        return false;
                    if (!stage.info.equalAllButShader(otherStage.info))
                        return false;
                }
                else if (otherStage.shader)
                    return false;
            }
            return canBeRestoredFrom_impl(this);
        }

        // extras for this class
        ICPUPipelineLayout* getLayout() 
        {
            assert(!isImmutable_debug());
            return PipelineNonAssetBase::m_layout.get(); 
        }
        const ICPUPipelineLayout* getLayout() const { return PipelineNonAssetBase::m_layout.get(); }

        inline void setLayout(core::smart_refctd_ptr<ICPUPipelineLayout>&& _layout)
        {
            assert(!isImmutable_debug());
            PipelineNonAssetBase::m_layout = std::move(_layout);
        }

        // The getters are weird because the shader pointer needs patching
		inline IShader::SSpecInfo<ICPUShader> getSpecInfo(const ICPUShader::E_SHADER_STAGE stage)
		{
			assert(!isImmutable_debug());
			const auto stageIx = stageToIndex(stage);
            if (stageIx<0)
                return {};
			return m_stages[stageIx].info;
		}
		inline IShader::SSpecInfo<const ICPUShader> getSpecInfo(const ICPUShader::E_SHADER_STAGE stage) const
		{
			const auto stageIx = stageToIndex(stage);
            if (stageIx<0)
                return {};
			return m_stages[stageIx].info;
		}
		inline bool setSpecInfo(const IShader::SSpecInfo<ICPUShader>& info)
		{
			assert(!isImmutable_debug());
            const int64_t specSize = info.valid();
            if (specSize<0)
                return false;
			const auto stage = info.shader->getStage();
			const auto stageIx = stageToIndex(stage);
			if (stageIx<0)
				return false;
            auto& outStage = m_stages[stageIx];
			outStage.info = info;
			outStage.shader = core::smart_refctd_ptr<ICPUShader>(info.shader);
			outStage.info.shader = outStage.shader.get();
            auto& outEntries = outStage.entries;
            if (specSize>0)
            {
                outEntries = std::make_unique<ICPUShader::SSpecInfo::spec_constant_map_t>();
                outEntries->reserve(info.entries->size());
                std::copy(info.entries->begin(),info.entries->end(),std::insert_iterator(*outEntries,outEntries->begin()));
            }
            else
                outEntries = nullptr;
			outStage.info.entries = outEntries.get();
			return true;
		}
        inline bool clearStage(const ICPUShader::E_SHADER_STAGE stage)
        {
            assert(!isImmutable_debug());
            const auto stageIx = stageToIndex(stage);
            if (stageIx<0)
                return false;
            m_stages[stageIx] = {};
            return true;
        }

    protected:
        using PipelineNonAssetBase::PipelineNonAssetBase;
        virtual ~ICPUPipeline() = default;

        inline void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override final
        {
            auto* other = static_cast<this_t*>(_other);

            if (_levelsBelow)
            {
                --_levelsBelow;
                restoreFromDummy_impl_call(PipelineNonAssetBase::m_layout.get(),other->m_layout.get(),_levelsBelow);
                for (auto i=0; i<MaxShaderStageCount; i++)
                if (m_stages[i].shader)
                    restoreFromDummy_impl_call(m_stages[i].shader.get(),other->m_stages[i].shader.get(),_levelsBelow);
            }
        }

        inline bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override final
        {
            --_levelsBelow;
            for (auto i=0; i<MaxShaderStageCount; i++)
            if (m_stages[i].shader && m_stages[i].shader->isAnyDependencyDummy())
                return true;
            return PipelineNonAssetBase::m_layout->isAnyDependencyDummy(_levelsBelow);
        }

        virtual this_t* clone_impl(core::smart_refctd_ptr<ICPUPipelineLayout>&& layout) const = 0;
        virtual bool canBeRestoredFrom_impl(const this_t* _other) const {return true;}
        virtual int8_t stageToIndex(const ICPUShader::E_SHADER_STAGE stage) const = 0;

        struct ShaderStage {
            core::smart_refctd_ptr<ICPUShader> shader = {};
            std::unique_ptr<ICPUShader::SSpecInfo::spec_constant_map_t> entries = {};
            ICPUShader::SSpecInfo info = {};
        } m_stages[MaxShaderStageCount] = {};
};

}
#endif