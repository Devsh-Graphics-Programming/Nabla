// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MITSUBA_LOADER_C_ELEMENT_R_FILTER_H_INCLUDED_
#define _NBL_EXT_MITSUBA_LOADER_C_ELEMENT_R_FILTER_H_INCLUDED_

#include "nbl/ext/MitsubaLoader/PropertyElement.h"
#include "nbl/ext/MitsubaLoader/IElement.h"

namespace nbl::ext::MitsubaLoader
{

class CElementRFilter : public IElement
{
	public:
		enum Type
		{
			INVALID,
			BOX,
			TENT,
			GAUSSIAN,
			MITCHELL,
			CATMULLROM,
			LANCZOS
		};
		struct Gaussian
		{
			float sigma = NAN; // can't look at mitsuba source to figure out the default it uses
		};
		struct MitchellNetravali
		{
			float B = 1.f / 3.f;
			float C = 1.f / 3.f;
		};
		struct LanczosSinc
		{
			int32_t lobes = 3;
		};

		CElementRFilter(const char* id) : IElement(id), type(GAUSSIAN)
		{
			gaussian = Gaussian();
		}
		virtual ~CElementRFilter() {}

		bool addProperty(SNamedPropertyElement&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* globalMetadata) override;
		IElement::Type getType() const override { return IElement::Type::RFILTER; }
		std::string getLogName() const override { return "rfilter"; }

		// make these public
		Type type;
		union
		{
			Gaussian			gaussian;
			MitchellNetravali	mitchell;
			MitchellNetravali	catmullrom;
			LanczosSinc			lanczos;
		};
};

}

#endif