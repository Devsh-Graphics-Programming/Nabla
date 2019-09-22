#ifndef __I_ELEMENT_FACTORY_H_INCLUDED__
#define __I_ELEMENT_FACTORY_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class ParserManager;

class CElementIntegrator : public IElement
{
	bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override) override { return true; }
	IElement::Type getType() const override { return IElement::Type::INTEGRATOR; }
	std::string getLogName() const override { return "integrator"; }

	// TODO refactor
	void addProperty(const SPropertyElementData& _property) override {}

	void addProperty(SPropertyElementData&& _property) override {}
};
class CElementSensor {};
class CElementFilm {};
class CElementRFilter {};
class CElementSampler {};
class CElementShape {};
class CElementBSDF {};
class CElementTexture {};
class CElementEmitter {};

class CElementFactory
{
	public:
		using element_creation_func = IElement*(*)(const char**, ParserManager*);
		const static core::unordered_map<std::string, std::pair<element_creation_func,bool>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> createElementTable;

		//constructs certain elements based on element's name and its attributes
		template<class element_type>
		static IElement* createElement(const char** _atts, ParserManager* _util);

};

}
}
}

#endif