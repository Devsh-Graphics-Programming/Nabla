#ifndef __IRR_I_CPU_SAMPLER_H_INCLUDED__
#define __IRR_I_CPU_SAMPLER_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ISampler.h"

namespace irr
{
namespace asset
{

class ICPUSampler : public ISampler, public IAsset
{
	protected:
		virtual ~ICPUSampler() = default;

	public:
		ICPUSampler(const SParams& _params) : ISampler(_params), IAsset() {}

        core::smart_refctd_ptr<IAsset> clone(uint32_t = ~0u) const override
        {
            auto cp = core::make_smart_refctd_ptr<ICPUSampler>(m_params);
            clone_common(cp.get());

            return cp;
        }

		size_t conservativeSizeEstimate() const override { return sizeof(m_params); }
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override 
        {
            convertToDummyObject_common(referenceLevelsBelowToConvert);
        }
		E_TYPE getAssetType() const override { return ET_SAMPLER; }
};

}
}

#endif