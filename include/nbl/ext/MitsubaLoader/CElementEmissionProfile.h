// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_ELEMENT_EMISSION_PROFILE_H_INCLUDED__
#define __C_ELEMENT_EMISSION_PROFILE_H_INCLUDED__

#include "vectorSIMD.h"
#include "nbl/ext/MitsubaLoader/CElementTexture.h"
#include "nbl/ext/MitsubaLoader/CElementTransform.h"


namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

struct CElementEmissionProfile : public IElement {

	CElementEmissionProfile(const char* id) : IElement(id), normalizeEnergy(1.0f), flatten(0.0) /*no blending by default*/ {}
	CElementEmissionProfile() : IElement(""), normalizeEnergy(1.0f) {}
	CElementEmissionProfile(const CElementEmissionProfile& other) : IElement("")
	{
		operator=(other);
	}
	CElementEmissionProfile(CElementEmissionProfile&& other) : IElement("")
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

	virtual ~CElementEmissionProfile()
	{
	}

	bool addProperty(SNamedPropertyElement&& _property) override;
	bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* globalMetadata) override {
		return true;
	}
	bool processChildData(IElement* _child, const std::string& name) override;
	IElement::Type getType() const override { return IElement::Type::EMISSION_PROFILE; }
	std::string getLogName() const override { return "emissionprofile "; }

	std::string filename;
	float normalizeEnergy, flatten;
};

}
}
}

#endif