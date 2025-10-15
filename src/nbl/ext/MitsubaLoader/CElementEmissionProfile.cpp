// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CElementEmissionProfile.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"

#include <functional>


namespace nbl::ext::MitsubaLoader
{

bool CElementEmissionProfile::addProperty(SNamedPropertyElement&& _property, system::logger_opt_ptr logger)
{
	if (_property.name=="filename")
	{
		if (_property.type!=SPropertyElementData::Type::STRING)
		{
			invalidXMLFileStructure(logger,"<emissionprofile>'s `filename` must be a string type, instead it's: "+_property.type);
			return false;
		}
		filename = _property.getProperty<SPropertyElementData::Type::STRING>();
		return true;
	}
	else if (_property.name=="normalization")
	{
		if (_property.type!=SPropertyElementData::Type::STRING)
		{
			invalidXMLFileStructure(logger,"<emissionprofile>'s `normalization` must be a string type, instead it's: "+_property.type);
			return false;
		}

		const auto normalizeS = std::string(_property.getProperty<SPropertyElementData::Type::STRING>());

		if (normalizeS=="UNIT_MAX")
			normalization = EN_UNIT_MAX;
		else if(normalizeS=="UNIT_AVERAGE_OVER_IMPLIED_DOMAIN")
			normalization = EN_UNIT_AVERAGE_OVER_IMPLIED_DOMAIN;
		else if(normalizeS=="UNIT_AVERAGE_OVER_FULL_DOMAIN")
			normalization = EN_UNIT_AVERAGE_OVER_FULL_DOMAIN;
		else
		{
			invalidXMLFileStructure(logger,"<emissionprofile>'s `normalization` is unrecognized: "+ normalizeS);
			normalization = EN_NONE;
		}

		return true;
	}
	else if (_property.name=="flatten") 
	{
		if (_property.type!=SPropertyElementData::Type::FLOAT)
			return false;

		flatten = _property.getProperty<SPropertyElementData::Type::FLOAT>();
		return true;
	}
	else
	{
		invalidXMLFileStructure(logger,"No emission profile can have such property set with name: "+_property.name);
		return false;
	}
}

bool CElementEmissionProfile::processChildData(IElement* _child, const std::string& name)
{
	if (!_child)
		return true;
	return false;
}

}