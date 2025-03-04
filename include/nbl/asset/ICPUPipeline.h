// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_PIPELINE_H_INCLUDED_


#include "nbl/asset/IAsset.h"
#include "nbl/asset/IPipeline.h"
#include "nbl/asset/ICPUPipelineLayout.h"


namespace nbl::asset
{

// Common Base class for pipelines
template<typename PipelineNonAssetBase, uint8_t MaxShaderStageCount>
class ICPUPipeline : public IAsset, public PipelineNonAssetBase
{
        using this_t = ICPUPipeline<PipelineNonAssetBase,MaxShaderStageCount>;

    public:
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
                    core::smart_refctd_ptr<IShader> newShader;
                    if (_depth>0u)
                    {
                        newShader = core::smart_refctd_ptr_static_cast<IShader>(shader->clone(_depth-1u));
                        stageInfo.shader = newShader.get();
                    }
                    cp->setSpecInfo(stageInfo);
                }
            }

            return core::smart_refctd_ptr<this_t>(cp,core::dont_grab);
        }

        // extras for this class
        ICPUPipelineLayout* getLayout() 
        {
            assert(isMutable());
            return const_cast<ICPUPipelineLayout*>(PipelineNonAssetBase::m_layout.get());
        }
        const ICPUPipelineLayout* getLayout() const { return PipelineNonAssetBase::m_layout.get(); }

        inline void setLayout(core::smart_refctd_ptr<const ICPUPipelineLayout>&& _layout)
        {
            assert(isMutable());
            PipelineNonAssetBase::m_layout = std::move(_layout);
        }

        // The getters are weird because the shader pointer, spec constant map and entry point needs patching
        inline IShader* getShader(const hlsl::ShaderStage stage)
        {
            assert(isMutable());
            return const_cast<IShader*>(getSpecInfo(stage).shader);
        }
		inline std::string* getEntryPoint(const hlsl::ShaderStage stage)
		{
			const auto stageIx = stageToIndex(stage);
            if (stageIx<0)
                return {};
			return &m_stages[stageIx].entryPoint;
		}
        inline IPipelineBase::SShaderSpecInfo::spec_constant_map_t* getSpecConstantMap(const hlsl::ShaderStage stage)
        {
            assert(isMutable());
            return const_cast<IPipelineBase::SShaderSpecInfo::spec_constant_map_t*>(getSpecInfo(stage).entries);
        }
        //
		inline IPipelineBase::SShaderSpecInfo getSpecInfo(const hlsl::ShaderStage stage) const
		{
			const auto stageIx = stageToIndex(stage);
            if (stageIx<0)
                return {};
			return m_stages[stageIx].info;
		}
		inline bool setSpecInfo(const IPipelineBase::SShaderSpecInfo& info)
		{
			assert(isMutable());
            const int64_t specSize = info.valid();
            if (specSize<0)
                return false;
			const auto stageIx = stageToIndex(info.stage);
			if (stageIx<0)
				return false;
            auto& outStage = m_stages[stageIx];
			outStage.info = info;
            outStage.entryPoint = info.entryPoint;
			outStage.shader = core::smart_refctd_ptr<IShader>(const_cast<IShader*>(info.shader));
			outStage.info.shader = outStage.shader.get();
            auto& outEntries = outStage.entries;
            if (specSize>0)
            {
                outEntries = std::make_unique<IPipelineBase::SShaderSpecInfo::spec_constant_map_t>();
                outEntries->reserve(info.entries->size());
                std::copy(info.entries->begin(),info.entries->end(),std::insert_iterator(*outEntries,outEntries->begin()));
            }
            else
                outEntries = nullptr;
			outStage.info.entries = outEntries.get();
			return true;
		}
        inline bool clearStage(const hlsl::ShaderStage stage)
        {
            assert(isMutable());
            const auto stageIx = stageToIndex(stage);
            if (stageIx<0)
                return false;
            m_stages[stageIx] = {};
            return true;
        }

    protected:
        using PipelineNonAssetBase::PipelineNonAssetBase;
        virtual ~ICPUPipeline() = default;

        virtual this_t* clone_impl(core::smart_refctd_ptr<const ICPUPipelineLayout>&& layout) const = 0;
        virtual int8_t stageToIndex(const hlsl::ShaderStage stage) const = 0;

        struct ShaderStage
        {
            std::string entryPoint = {};
            core::smart_refctd_ptr<IShader> shader = {};
            std::unique_ptr<IPipelineBase::SShaderSpecInfo::spec_constant_map_t> entries = {};
            IPipelineBase::SShaderSpecInfo info = {};
        } m_stages[MaxShaderStageCount] = {};
};

}
#endif