// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/core/string/stringutil.h"

#include "nbl/ext/MitsubaLoader/CElementRFilter.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"

#include "nbl/ext/MitsubaLoader/ElementMacros.h"


namespace nbl::ext::MitsubaLoader
{

bool CElementRFilter::addProperty(SNamedPropertyElement&& _property, system::logger_opt_ptr logger)
{
	if (_property.type == SNamedPropertyElement::Type::INTEGER)
	{
		if (core::strcmpi(_property.name,std::string("lobes")))
		{
			invalidXMLFileStructure(logger,"\"lobes\" must be an integer property");
			return false;
		}
		lanczos.lobes = _property.ivalue;
		return true;
	}
	else if (_property.type == SNamedPropertyElement::Type::FLOAT)
	{
		if (core::strcmpi(_property.name,std::string("b"))==0)
		{
			mitchell.B = _property.fvalue;
			return true;
		}
		else if (core::strcmpi(_property.name,std::string("c"))==0)
		{
			mitchell.C = _property.fvalue;
			return true;
		}
		else if (core::strcmpi(_property.name,std::string("kappa"))==0)
		{
			kappa = _property.fvalue;
			return true;
		}
		else if (core::strcmpi(_property.name,std::string("Emin"))==0)
		{
			Emin = _property.fvalue;
			return true;
		}
		else
			invalidXMLFileStructure(logger,"unsupported rfilter property called: "+_property.name);
	}
	else
		invalidXMLFileStructure(logger,"this reconstruction filter type does not take this parameter type for parameter: " + _property.name);

	return false;
}

bool CElementRFilter::onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger)
{
	NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(true);

	// TODO: Validation

	return true;
}

}