// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/core/string/stringutil.h"

#include "nbl/ext/MitsubaLoader/CElementRFilter.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"

#include "nbl/ext/MitsubaLoader/ElementMacros.h"


namespace nbl::ext::MitsubaLoader
{
	
auto CElementRFilter::compAddPropertyMap() -> AddPropertyMap<CElementRFilter>
{
	using this_t = CElementRFilter;
	AddPropertyMap<CElementRFilter> retval;
	
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(sigma,FLOAT,std::is_same,Gaussian);

	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(B,FLOAT,is_any_of,MitchellNetravali,CatmullRom);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(C,FLOAT,is_any_of,MitchellNetravali,CatmullRom);

	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(lobes,INTEGER,std::is_same,LanczosSinc);

	// common
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(kappa,FLOAT);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(Emin,FLOAT);

	return retval;
}

bool CElementRFilter::onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger)
{
	NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(true);

	// TODO: Validation

	return true;
}

}