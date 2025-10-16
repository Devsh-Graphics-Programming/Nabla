// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/IElement.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"


#define NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY(PROP_TYPE,NAME) retval.registerCallback(SNamedPropertyElement::Type::PROP_TYPE,NAME,{.func=[](this_t* _this, SNamedPropertyElement&& _property, const system::logger_opt_ptr logger)->bool

/*
template<>
CElementFactory::return_type CElementFactory::createElement<CElementSampler>(const char** _atts, ParserManager* _util)
{
	const char* type;
	const char* id;
	std::string name;
	if (!IElement::getTypeIDAndNameStrings(type, id, name, _atts))
		return CElementFactory::return_type(nullptr, "");

	static const core::unordered_map<std::string, CElementSampler::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		{"independent", CElementSampler::Type::INDEPENDENT},
		{"stratified", CElementSampler::Type::STRATIFIED},
		{"ldsampler", CElementSampler::Type::LDSAMPLER},
		{"halton", CElementSampler::Type::HALTON},
		{"hammersley", CElementSampler::Type::HAMMERSLEY},
		{"sobol", CElementSampler::Type::SOBOL}
	};

	auto found = StringToType.find(type);
	if (found==StringToType.end())
	{
		ParserLog::invalidXMLFileStructure("unknown type");
		_NBL_DEBUG_BREAK_IF(false);
		return CElementFactory::return_type(nullptr, "");
	}

	CElementSampler* obj = _util->objects.construct<CElementSampler>(id);
	if (!obj)
		return CElementFactory::return_type(nullptr, "");

	obj->type = found->second;
	obj->sampleCount = 4;
	//validation
	switch (obj->type)
	{
		case CElementSampler::Type::STRATIFIED:
			[[fallthrough]];
		case CElementSampler::Type::LDSAMPLER:
			obj->dimension = 4;
			break;
		case CElementSampler::Type::HALTON:
			[[fallthrough]];
		case CElementSampler::Type::HAMMERSLEY:
			obj->scramble = -1;
			break;
		case CElementSampler::Type::SOBOL:
			obj->scramble = 0;
			break;
		default:
			break;
	}
	return CElementFactory::return_type(obj, std::move(name));
}

bool CElementSampler::addProperty(SNamedPropertyElement&& _property)
{
	if (_property.type == SNamedPropertyElement::Type::INTEGER &&
		_property.name == "sampleCount")
	{
		sampleCount = _property.ivalue;
		switch (type)
		{
			case Type::STRATIFIED:
				sampleCount = ceilf(sqrtf(sampleCount));
				break;
			case Type::LDSAMPLER:
				//sampleCount = core::roundUpToPoT<int32_t>(sampleCount);
				break;
			default:
				break;
		}
	}
	else
	if (_property.type == SNamedPropertyElement::Type::INTEGER &&
		_property.name == "dimension")
	{
		dimension = _property.ivalue;
		if (type == Type::INDEPENDENT || type == Type::HALTON || type == Type::HAMMERSLEY)
		{
			ParserLog::invalidXMLFileStructure("this sampler type does not take these parameters");
			_NBL_DEBUG_BREAK_IF(true);
			return false;
		}
	}
	else
	if (_property.type == SNamedPropertyElement::Type::INTEGER &&
		_property.name == "scramble")
	{
		scramble = _property.ivalue;
		if (type==Type::INDEPENDENT || type==Type::STRATIFIED || type == Type::LDSAMPLER)
		{
			ParserLog::invalidXMLFileStructure("this sampler type does not take these parameters");
			_NBL_DEBUG_BREAK_IF(true);
			return false;
		}
	}
	else
	{
		_NBL_DEBUG_BREAK_IF(true);
		return false;
	}

	return true;
}
*/

#define NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(NON_FATAL) if (type==Type::INVALID) \
{ \
	invalidXMLFileStructure(logger,getLogName()+": type not specified"); \
	return NON_FATAL; \
}