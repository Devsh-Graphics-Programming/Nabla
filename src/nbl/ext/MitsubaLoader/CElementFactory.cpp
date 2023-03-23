// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/MitsubaLoader/CElementFactory.h"

#include "nbl/ext/MitsubaLoader/ParserUtil.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{


CElementFactory::return_type CElementFactory::processAlias(const char** _atts, ParserManager* _util)
{
	const char* id = nullptr;
	const char* as = nullptr;
	std::string name;
	if (IElement::areAttributesInvalid(_atts, 4u))
		return CElementFactory::return_type(nullptr,std::move(name));

	while (*_atts)
	{
		if (core::strcmpi(_atts[0], "id")==0)
			id = _atts[1];
		else if (core::strcmpi(_atts[0], "as")==0)
			as = _atts[1];
		else if (core::strcmpi(_atts[0], "name")==0)
			name = _atts[1];
		_atts += 2;
	}

	if (!id || !as)
		return CElementFactory::return_type(nullptr,std::move(name));

	auto* original = _util->handles[id];
	_util->handles[as] = original;
	return CElementFactory::return_type(original,std::move(name));
}

CElementFactory::return_type CElementFactory::processRef(const char** _atts, ParserManager* _util)
{
	const char* id;
	std::string name;
	if (!IElement::getIDAndName(id,name,_atts))
	{
		os::Printer::log("[ERROR] Malformed `<ref>` element!", ELL_ERROR);
		return CElementFactory::return_type(nullptr, std::move(name));
	}
	auto* original = _util->handles[id];
	if (!original)
		os::Printer::log(std::string("[ERROR] Used a `<ref name=\""+name+"\" id=\"")+id+"\">` element but referenced element not defined in preceeding XML!", ELL_ERROR);
	return CElementFactory::return_type(original, std::move(name));
}


const core::unordered_map<std::string, std::pair<CElementFactory::element_creation_func,bool>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> CElementFactory::createElementTable =
{
	{"integrator",	{CElementFactory::createElement<CElementIntegrator>,true}},
	{"sensor",		{CElementFactory::createElement<CElementSensor>,true}},
	{"film",		{CElementFactory::createElement<CElementFilm>,true}},
	{"rfilter",		{CElementFactory::createElement<CElementRFilter>,true}},
	{"sampler",		{CElementFactory::createElement<CElementSampler>,true}},
	{"shape",		{CElementFactory::createElement<CElementShape>,true}},
	{"transform",	{CElementFactory::createElement<CElementTransform>,true}},
	//{"animation",	{CElementFactory::createElement<CElementAnimation>,true}},
	{"bsdf",		{CElementFactory::createElement<CElementBSDF>,true}},
	{"texture",		{CElementFactory::createElement<CElementTexture>,true}},
	{"emitter",		{CElementFactory::createElement<CElementEmitter>,true}},
	{"alias",		{CElementFactory::processAlias,true}},
	{"ref",			{CElementFactory::processRef,true}}
};

}
}
}