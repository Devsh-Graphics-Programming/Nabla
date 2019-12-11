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

		size_t conservativeSizeEstimate() const override { return sizeof(m_params); }
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override { }
		E_TYPE getAssetType() const override { return ET_SAMPLER; }
};

}
}

#endif