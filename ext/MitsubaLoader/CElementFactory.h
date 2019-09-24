#ifndef __I_ELEMENT_FACTORY_H_INCLUDED__
#define __I_ELEMENT_FACTORY_H_INCLUDED__

#include "../../ext/MitsubaLoader/CElementSensor.h"
#include "../../ext/MitsubaLoader/CElementIntegrator.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class ParserManager;

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
		//
		static IElement* processAlias(const char** _atts, ParserManager* _util);
		static IElement* processRef(const char** _atts, ParserManager* _util);
};


}
}
}

#endif