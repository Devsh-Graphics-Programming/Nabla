#ifndef __C_ELEMENT_SAMPLER_H_INCLUDED__
#define __C_ELEMENT_SAMPLER_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CGlobalMitsubaMetadata;

class CElementSampler : public IElement
{
	public:
		enum Type
		{
			NONE,
			INDEPENDENT,
			STRATIFIED,
			LDSAMPLER,
			HALTON,
			HAMMERSLEY,
			SOBOL
		};

		bool addProperty(SPropertyElementData&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata) override;
		IElement::Type getType() const override { return IElement::Type::SAMPLER; }
		std::string getLogName() const override { return "sampler"; }

		// make these public
		Type type;
		int sampleCount;
		union
		{
			int dimension;
			int scramble;
		};
	private:
		bool dimensionSet = false;
		bool scrambleSet = false;
};



}
}
}

#endif