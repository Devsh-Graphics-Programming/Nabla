#ifndef __NBL_I_CPU_GRAPHICS_PIPELINE_H_INCLUDED__
#define __NBL_I_CPU_GRAPHICS_PIPELINE_H_INCLUDED__

#include "nbl/asset/IGraphicsPipeline.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
#include "nbl/asset/ICPURenderpass.h"

namespace nbl {
namespace asset
{

class ICPUGraphicsPipeline final : public IAsset, public IGraphicsPipeline<ICPURenderpassIndependentPipeline, ICPURenderpass>
{
    using base_t = IGraphicsPipeline<ICPURenderpassIndependentPipeline, ICPURenderpass>;

public:
    ~ICPUGraphicsPipeline()
    {

    }

    using base_t::base_t;
    _NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_GRAPHICS_PIPELINE;

    renderpass_independent_t* getRenderpassIndependentPipeline() { return m_params.renderpassIndependent.get(); }
    renderpass_t* getRenderpass() { return m_params.renderpass.get(); }
    SCreationParams& getCreationParameters() { return m_params; }

    bool equals(const IAsset* _other) const override
	{
        return compatible(_other);
	}

	size_t hash(std::unordered_map<IAsset*, size_t>* temporary_hash_cache = nullptr) const override
	{
		size_t seed = AssetType;
        core::hash_combine(seed, m_params.createFlags);
        core::hash_combine(seed, m_params.rasterizationSamples);
        core::hash_combine(seed, m_params.renderpass);
        core::hash_combine(seed, m_params.renderpassIndependent);
        core::hash_combine(seed, m_params.subpassIx);
		return seed; // TODO
	}
private:
    bool compatible(const IAsset* _other) const override {
        auto* other = static_cast<const ICPUGraphicsPipeline*>(_other);
        return IAsset::compatible(_other) && (other->m_params == m_params);
	}
};

}
}

#endif