#ifndef __C_ELEMENT_R_FILTER_H_INCLUDED__
#define __C_ELEMENT_R_FILTER_H_INCLUDED__

#include "../../ext/MitsubaLoader/PropertyElement.h"

#include "../../ext/MitsubaLoader/IElement.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CGlobalMitsubaMetadata;

class CElementRFilter : public IElement
{
	public:
		enum Type
		{
			INVALID,
			BOX,
			TENT,
			GAUSSIAN,
			MITCHELL,
			CATMULLROM,
			LANCZOS
		};
		struct Gaussian
		{
			float sigma = NAN; // can't look at mitsuba source to figure out the default it uses
		};
		struct MitchellNetravali
		{
			float B = 1.f / 3.f;
			float C = 1.f / 3.f;
		};
		struct LanczosSinc
		{
			int32_t lobes = 3;
		};

		CElementRFilter(const char* id) : IElement(id), type(GAUSSIAN)
		{
			gaussian = Gaussian();
		}
		virtual ~CElementRFilter() {}

		bool addProperty(SNamedPropertyElement&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata) override;
		IElement::Type getType() const override { return IElement::Type::RFILTER; }
		std::string getLogName() const override { return "rfilter"; }

		// make these public
		Type type;
		union
		{
			Gaussian			gaussian;
			MitchellNetravali	mitchell;
			MitchellNetravali	catmullrom;
			LanczosSinc			lanczos;
		};
};


}
}
}

#endif