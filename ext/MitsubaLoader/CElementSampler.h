#ifndef __C_ELEMENT_SAMPLER_H_INCLUDED__
#define __C_ELEMENT_SAMPLER_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

enum class ESamplerType
{
	NONE,
	INDEPENDEND,
	STRATIFIED,
	LDSAMPLER,
	HALTON,
	HAMMERSLEY,
	SOBOL
};

struct SSamplerMetadata
{
	ESamplerType type;
	int sampleCount;
	union
	{
		int dimension;
		int scramble;
	};
};

class CElementSampler : public IElement
{
public:
	CElementSampler()
		:data({ESamplerType::NONE, 4, 4}) {};

	virtual bool processAttributes(const char** _atts) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager) override;
	virtual IElement::Type getType() const override { return IElement::Type::SAMPLER; }
	virtual std::string getLogName() const override { return "sampler"; }

	SSamplerMetadata getMetadata() const { return data; };

private:
	SSamplerMetadata data;

};



}
}
}

#endif