#ifndef __IRR_I_CPU_PIPELINE_LAYOUT_H_INCLUDED__
#define __IRR_I_CPU_PIPELINE_LAYOUT_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUDescriptorSetLayout.h"
#include "irr/asset/IPipelineLayout.h"

namespace irr
{
namespace asset
{

class ICPUPipelineLayout : public IAsset, public IPipelineLayout<ICPUDescriptorSetLayout>
{
	public:
		using IPipelineLayout<ICPUDescriptorSetLayout>::IPipelineLayout;

		ICPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t _set) 
        {
            return m_descSetLayouts[_set].get(); 
        }
		const ICPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t _set) const { return m_descSetLayouts[_set].get(); }

        void setDescriptorSetLayout(uint32_t _set, core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _dslayout) 
        { 
            if (isImmutable_debug())
                return;
            m_descSetLayouts[_set] = std::move(_dslayout); 
        }

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            std::array<core::smart_refctd_ptr<ICPUDescriptorSetLayout>, DESCRIPTOR_SET_COUNT> dsLayouts;
            for (size_t i = 0ull; i < dsLayouts.size(); ++i)
                dsLayouts[i] = (m_descSetLayouts[i] && _depth > 0u) ? core::smart_refctd_ptr_static_cast<ICPUDescriptorSetLayout>(m_descSetLayouts[i]->clone(_depth-1u)) : m_descSetLayouts[i];

            auto cp = core::make_smart_refctd_ptr<ICPUPipelineLayout>(
                nullptr, nullptr, 
                std::move(dsLayouts[0]), std::move(dsLayouts[1]), std::move(dsLayouts[2]), std::move(dsLayouts[3])
            );
            clone_common(cp.get());
            cp->m_pushConstantRanges = m_pushConstantRanges;

            return cp;
        }

		size_t conservativeSizeEstimate() const override { return m_descSetLayouts.size()*sizeof(void*)+m_pushConstantRanges->size()*sizeof(SPushConstantRange); }
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
            convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert)
			    for (auto it=m_descSetLayouts.begin(); it!=m_descSetLayouts.end(); it++)
			        if (it->get())
				        it->get()->convertToDummyObject(referenceLevelsBelowToConvert-1u);

            if (canBeConvertedToDummy())
			    m_pushConstantRanges = nullptr;
		}

        bool canBeRestoredFrom(const IAsset* _other) const override
        {
            if (!IAsset::canBeRestoredFrom(_other))
                return false;

            return true;
        }

        _IRR_STATIC_INLINE_CONSTEXPR auto AssetType = ET_PIPELINE_LAYOUT;
        inline E_TYPE getAssetType() const override { return AssetType; }

private:
    void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
    {
        auto* other = static_cast<ICPUPipelineLayout*>(_other);

        std::swap(m_pushConstantRanges, other->m_pushConstantRanges);
        if (_levelsBelow)
        {
            --_levelsBelow;
            for (uint32_t i = 0u; i < m_descSetLayouts.size(); ++i)
                m_descSetLayouts[i]->restoreFromDummy(other->m_descSetLayouts[i].get(), _levelsBelow);
        }
    }

	protected:
		virtual ~ICPUPipelineLayout() = default;
};

}
}

#endif