// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_COMPUTE_PIPELINE_H_INCLUDED_

#include "nbl/asset/ICPUPipelineLayout.h"
#include "nbl/asset/ICPUShader.h"

namespace nbl::asset
{

//! CPU Version of Compute Pipeline
class ICPUComputePipeline : public IAsset
{
    public:
        static core::smart_refctd_ptr<ICPUComputePipeline> create(core::smart_refctd_ptr<ICPUPipelineLayout>&& _layout, const ICPUShader::SSpecInfo& _cs)
        {
            if (!_layout)
                return nullptr;
            auto retval = new ICPUComputePipeline(std::move(_layout));
            if (!retval->setSpecInfo(_cs))
            {
                retval->drop();
                return nullptr;
            }
            return core::smart_refctd_ptr<ICPUComputePipeline>(retval,core::dont_grab);
        }

        size_t conservativeSizeEstimate() const override {return sizeof(m_layout)+sizeof(m_info);}
        void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
	    {
            convertToDummyObject_common(referenceLevelsBelowToConvert);

		    if (referenceLevelsBelowToConvert)
		    {
                //intentionally parent is not converted
                --referenceLevelsBelowToConvert;
			    m_shader->convertToDummyObject(referenceLevelsBelowToConvert);
			    m_layout->convertToDummyObject(referenceLevelsBelowToConvert);
		    }
	    }

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            core::smart_refctd_ptr<ICPUPipelineLayout> layout = (_depth>0u && m_layout) ? core::smart_refctd_ptr_static_cast<ICPUPipelineLayout>(m_layout->clone(_depth-1u)):m_layout;
            core::smart_refctd_ptr<ICPUShader> shader = (_depth>0u && m_shader) ? core::smart_refctd_ptr_static_cast<ICPUShader>(m_shader->clone(_depth-1u)):m_shader;

            auto cp = new ICPUComputePipeline(std::move(layout));
            cp->setSpecInfo(m_info);
            clone_common(cp);

            return core::smart_refctd_ptr<ICPUComputePipeline>(cp,core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_COMPUTE_PIPELINE;
        inline E_TYPE getAssetType() const override { return AssetType; }

        bool canBeRestoredFrom(const IAsset* _other) const override
        {
            auto* other = static_cast<const ICPUComputePipeline*>(_other);
            if (!m_shader->canBeRestoredFrom(other->m_shader.get()))
                return false;
            if (!m_info.equalAllButShader(other->m_info))
                return false;
            if (!m_layout->canBeRestoredFrom(other->m_layout.get()))
                return false;
            return true;
        }

        //
        ICPUPipelineLayout* getLayout() 
        {
            assert(!isImmutable_debug());
            return m_layout.get(); 
        }
        const ICPUPipelineLayout* getLayout() const { return m_layout.get(); }

        inline void setLayout(core::smart_refctd_ptr<ICPUPipelineLayout>&& _layout)
        {
            assert(!isImmutable_debug());
            m_layout = std::move(_layout);
        }

        // The getters are weird because the shader pointer needs patching
        inline IShader::SSpecInfo<ICPUShader> getSpecInfo()
        {
            assert(!isImmutable_debug());
            return m_info;
        }
        inline IShader::SSpecInfo<const ICPUShader> getSpecInfo() const {return m_info;}
        inline bool setSpecInfo(const IShader::SSpecInfo<ICPUShader>& info)
        {
            assert(!isImmutable_debug());
            const int64_t specSize = info.valid();
            if (specSize<0)
                return false;
            if (info.shader->getStage()!=ICPUShader::ESS_COMPUTE)
                return false;
            m_info = info;
            m_shader = core::smart_refctd_ptr<ICPUShader>(info.shader);
            m_info.shader = m_shader.get();
            if (specSize>0)
            {
                m_entries = std::make_unique<ICPUShader::SSpecInfo::spec_constant_map_t>();
                m_entries->reserve(info.entries->size());
                std::copy(info.entries->begin(),info.entries->end(),std::insert_iterator(*m_entries,m_entries->begin()));
            }
            else
                m_entries = nullptr;
            m_info.entries = m_entries.get();
            return true;
        }

    protected:
        ICPUComputePipeline(core::smart_refctd_ptr<ICPUPipelineLayout>&& _layout) : m_layout(std::move(_layout)) {}
        virtual ~ICPUComputePipeline() = default;

        void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
        {
            auto* other = static_cast<ICPUComputePipeline*>(_other);

            if (_levelsBelow)
            {
                --_levelsBelow;
                restoreFromDummy_impl_call(m_shader.get(), other->m_shader.get(), _levelsBelow);
                restoreFromDummy_impl_call(m_layout.get(), other->m_layout.get(), _levelsBelow);
            }
        }

        bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
        {
            --_levelsBelow;
            return m_shader->isAnyDependencyDummy(_levelsBelow) || m_layout->isAnyDependencyDummy(_levelsBelow);
        }

        core::smart_refctd_ptr<ICPUPipelineLayout> m_layout;
        core::smart_refctd_ptr<ICPUShader> m_shader = {};
        std::unique_ptr<ICPUShader::SSpecInfo::spec_constant_map_t> m_entries = {};
        ICPUShader::SSpecInfo m_info = {};
};

}
#endif