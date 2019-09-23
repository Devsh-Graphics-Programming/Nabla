#include "../../ext/MitsubaLoader/CElementFactory.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

const core::unordered_map<std::string, std::pair<CElementFactory::element_creation_func,bool>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> CElementFactory::createElementTable =
{
	{"integrator",	{CElementFactory::createElement<CElementIntegrator>,true}},
	//{"sensor",	{CElementFactory::createElement<CElementSensor>,true}},
	{"film",		{CElementFactory::createElement<CElementFilm>,true}},
	{"rfilter",		{CElementFactory::createElement<CElementRFilter>,true}},
	{"sampler",		{CElementFactory::createElement<CElementSampler>,true}}
};
/*
_IRR_STATIC_INLINE_CONSTEXPR const char* complexElements[] = {
	"alias","transform","ref","shape","bsdf","texture","emitter"
};
*/

#if 0
IElement* CElementFactory::createElement(const char* _el, const char** _atts)
{
	//should be removing white spaces performed before string comparison?
	IElement* result = nullptr;
	
	
	if (!std::strcmp(_el, "scene"))
	{
		return parseScene(_el, _atts);
	}
	else
	if (!std::strcmp(_el, "shape"))
	{
		return parseShape(_el, _atts);
	}
	else
	if (!std::strcmp(_el, "transform"))
	{
		CElementTransform* transform = new CElementTransform();

		if (!transform->processAttributes(_atts))
		{
			delete transform;
			_IRR_DEBUG_BREAK_IF(true);
			return nullptr;
		}

		return transform;
	}
	else
	if (!std::strcmp(_el, "emitter"))
	{
		CElementEmitter* emitter = new CElementEmitter();

		if (!emitter->processAttributes(_atts))
		{
			delete emitter;
			_IRR_DEBUG_BREAK_IF(true);
			return nullptr;
		}

		return emitter;
	}
}
IElement* CElementFactory::parseShape(const char* _el, const char** _atts)
{
	CShape* result = new CShape();

	if (!result->processAttributes(_atts))
	{
		_IRR_DEBUG_BREAK_IF(true);
		delete result;
		return nullptr;
	}

	//ParserLog::mitsubaLoaderError("There is no type attribute for shape element. \n");
	return result;
}
#endif

}
}
}