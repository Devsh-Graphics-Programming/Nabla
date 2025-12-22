// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CElementEmissionProfile.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"

#include "nbl/ext/MitsubaLoader/ElementMacros.h"

#include <functional>


namespace nbl::ext::MitsubaLoader
{

auto CElementEmissionProfile::compAddPropertyMap() -> AddPropertyMap<CElementEmissionProfile>
{
	using this_t = CElementEmissionProfile;
	AddPropertyMap<CElementEmissionProfile> retval;
	
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(filename,STRING);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("normalization",STRING)
		{
			const auto normalizeS = std::string(_property.svalue);
			if (normalizeS == "UNIT_MAX")
				_this->normalization = EN_UNIT_MAX;
			else if (normalizeS == "UNIT_AVERAGE_OVER_IMPLIED_DOMAIN")
				_this->normalization = EN_UNIT_AVERAGE_OVER_IMPLIED_DOMAIN;
			else if (normalizeS == "UNIT_AVERAGE_OVER_FULL_DOMAIN")
				_this->normalization = EN_UNIT_AVERAGE_OVER_FULL_DOMAIN;
			else
			{
				logger.log("<emissionprofile>'s `normalization` is unrecognized: \"%s\"",system::ILogger::ELL_ERROR,normalizeS.c_str());
				_this->normalization = EN_NONE;
			}
			return true;
		}
	});
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(flatten,FLOAT);

	return retval;
}

bool CElementEmissionProfile::processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger)
{
	if (!_child)
		return true;
	return false;
}

}