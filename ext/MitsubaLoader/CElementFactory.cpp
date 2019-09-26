#include "../../ext/MitsubaLoader/CElementFactory.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


IElement* CElementFactory::processAlias(const char** _atts, ParserManager* _util)
{
	const char* id = nullptr;
	const char* as = nullptr;
	if (IElement::areAttributesInvalid(_atts, 2u))
		return nullptr;
	if (core::strcmpi(_atts[0], "id")==0)
	{
		if (core::strcmpi(_atts[2], "id")==0)
			return nullptr;
		id = _atts[3];
		if (core::strcmpi(_atts[0], "as")==0)
			as = _atts[1];
	}
	else
	{
		id = _atts[1];
		if (core::strcmpi(_atts[2], "as")==0)
			as = _atts[3];
	}

	auto* original = _util->handles[id];
	_util->handles[as] = original;
	return original;
}
IElement* CElementFactory::processRef(const char** _atts, ParserManager* _util)
{
	assert(false);
	return nullptr;
}


const core::unordered_map<std::string, std::pair<CElementFactory::element_creation_func,bool>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> CElementFactory::createElementTable =
{
	{"integrator",	{CElementFactory::createElement<CElementIntegrator>,true}},
	{"sensor",		{CElementFactory::createElement<CElementSensor>,true}},
	{"film",		{CElementFactory::createElement<CElementFilm>,true}},
	{"rfilter",		{CElementFactory::createElement<CElementRFilter>,true}},
	{"sampler",		{CElementFactory::createElement<CElementSampler>,true}},
	{"transform",	{CElementFactory::createElement<CElementTransform>,true}},
	//{"animation",	{CElementFactory::createElement<CElementAnimation>,true}},
	{"alias",		{CElementFactory::processAlias,true}},
	{"ref",			{CElementFactory::processRef,false}}
};
/*
_IRR_STATIC_INLINE_CONSTEXPR const char* complexElements[] = {
	"shape","bsdf","texture","emitter"
};
*/

}
}
}