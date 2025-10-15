// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_EMISSION_PROFILE_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_EMISSION_PROFILE_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/CElementTexture.h"
#include "nbl/ext/MitsubaLoader/CElementTransform.h"


namespace nbl::ext::MitsubaLoader
{

struct CElementEmissionProfile final : public IElement
{
	inline CElementEmissionProfile(const char* id) : IElement(id), normalization(EN_NONE), flatten(0.0) /*no blending by default*/ {}
	inline CElementEmissionProfile() : IElement(""), normalization(EN_NONE) {}
	inline CElementEmissionProfile(const CElementEmissionProfile& other) : IElement("")
	{
		operator=(other);
	}
	inline CElementEmissionProfile(CElementEmissionProfile&& other) : IElement("")
	{
		operator=(std::move(other));
	}

	inline CElementEmissionProfile& operator=(const CElementEmissionProfile& other)
	{
		IElement::operator=(other);
		filename = other.filename;
		return *this;
	}

	inline CElementEmissionProfile& operator=(CElementEmissionProfile&& other)
	{
		IElement::operator=(std::move(other));
		std::swap(filename, other.filename);
		return *this;
	}

	inline ~CElementEmissionProfile()
	{
	}

	bool addProperty(SNamedPropertyElement&& _property, system::logger_opt_ptr logger) override;
	inline bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override {return true;}
	bool processChildData(IElement* _child, const std::string& name) override;
	inline IElement::Type getType() const override { return IElement::Type::EMISSION_PROFILE; }
	inline std::string getLogName() const override { return "emissionprofile "; }
	
	enum E_NORMALIZE : uint8_t
	{
		EN_UNIT_MAX,									//! normalize the intensity by dividing out the maximum intensity
		EN_UNIT_AVERAGE_OVER_IMPLIED_DOMAIN,			//! normalize by energy - integrate the profile over the hemisphere as well as the solid angles where the profile has emission above 0.
		EN_UNIT_AVERAGE_OVER_FULL_DOMAIN,				//! similar to UNIT_AVERAGE_OVER_IMPLIED_DOMAIN but in this case we presume the soild angle of the domain is (CIESProfile::vAngles.front()-CIESProfile::vAngles.back())*4.f
		EN_NONE											//! no normalization

	};

	std::string filename; // TODO: test destructor runs
	E_NORMALIZE normalization;
	float flatten; // TODO: why is this named this way?
};

}
#endif