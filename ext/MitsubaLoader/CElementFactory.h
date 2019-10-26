#ifndef __I_ELEMENT_FACTORY_H_INCLUDED__
#define __I_ELEMENT_FACTORY_H_INCLUDED__

#include "../../ext/MitsubaLoader/CElementSensor.h"
#include "../../ext/MitsubaLoader/CElementIntegrator.h"
#include "../../ext/MitsubaLoader/CElementShape.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class ParserManager;

class CElementFactory
{
	public:
		using return_type = std::pair<IElement*,std::string>;
		using element_creation_func = return_type(*)(const char**, ParserManager*);
		const static core::unordered_map<std::string, std::pair<element_creation_func,bool>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> createElementTable;

		//constructs certain elements based on element's name and its attributes
		template<class element_type>
		static return_type createElement(const char** _atts, ParserManager* _util);
		//
		static return_type processAlias(const char** _atts, ParserManager* _util);
		static return_type processRef(const char** _atts, ParserManager* _util);
};


}
}
}

#endif