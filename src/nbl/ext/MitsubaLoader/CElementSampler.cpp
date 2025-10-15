// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CElementSampler.h"
#include "nbl/ext/MitsubaLoader/ElementMacros.h"


namespace nbl::ext::MitsubaLoader
{


bool CElementSampler::addProperty(SNamedPropertyElement&& _property, system::logger_opt_ptr logger)
{
	if (_property.type==SNamedPropertyElement::Type::INTEGER && _property.name=="sampleCount")
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
	else if (_property.type == SNamedPropertyElement::Type::INTEGER && _property.name == "dimension")
	{
		dimension = _property.ivalue;
		if (type == Type::INDEPENDENT || type == Type::HALTON || type == Type::HAMMERSLEY)
		{
			invalidXMLFileStructure(logger,"this sampler type ("+std::to_string(type)+") does not take these parameters");
			return false;
		}
	}
	else if (_property.type == SNamedPropertyElement::Type::INTEGER && _property.name == "scramble")
	{
		scramble = _property.ivalue;
		if (type==Type::INDEPENDENT || type==Type::STRATIFIED || type == Type::LDSAMPLER)
		{
			invalidXMLFileStructure(logger,"this sampler type ("+std::to_string(type)+") does not take these parameters");
			return false;
		}
	}
	else
	{
		invalidXMLFileStructure(logger,"unknown property named `"+_property.name+"` of type "+std::to_string(_property.type));
		return false;
	}

	return true;
}

bool CElementSampler::onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger)
{
	NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(true);

	// TODO: Validation

	return true;
}

}