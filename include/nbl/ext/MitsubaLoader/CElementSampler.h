// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MITSUBA_LOADER_C_ELEMENT_SAMPLER_H_INCLUDED_
#define _NBL_EXT_MITSUBA_LOADER_C_ELEMENT_SAMPLER_H_INCLUDED_

#include "nbl/ext/MitsubaLoader/IElement.h"

namespace nbl::ext::MitsubaLoader
{

class CGlobalMitsubaMetadata;

class CElementSampler : public IElement
{
	public:
		enum Type
		{
			INVALID,
			INDEPENDENT,
			STRATIFIED,
			LDSAMPLER,
			HALTON,
			HAMMERSLEY,
			SOBOL
		};

		CElementSampler(const char* id) : IElement(id), type(INVALID), sampleCount(4) {}
		virtual ~CElementSampler() {}

		bool addProperty(SNamedPropertyElement&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* globalMetadata) override;
		IElement::Type getType() const override { return IElement::Type::SAMPLER; }
		std::string getLogName() const override { return "sampler"; }

		// make these public
		Type type;
		int32_t sampleCount;
		union
		{
			int32_t dimension;
			int32_t scramble;
		};
};

}

#endif