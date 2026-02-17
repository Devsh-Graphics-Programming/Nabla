// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_SAMPLER_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_SAMPLER_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/IElement.h"


namespace nbl::ext::MitsubaLoader
{
class CElementSampler : public IElement
{
	public:
		enum Type : uint8_t
		{
			INDEPENDENT,
			STRATIFIED,
			LDSAMPLER,
			HALTON,
			HAMMERSLEY,
			SOBOL,
			INVALID
		};
		static inline core::unordered_map<core::string,Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> compStringToTypeMap()
		{
			return {
				{"independent",	Type::INDEPENDENT},
				{"stratified",	Type::STRATIFIED},
				{"ldsampler",	Type::LDSAMPLER},
				{"halton",		Type::HALTON},
				{"hammersley",	Type::HAMMERSLEY},
				{"sobol",		Type::SOBOL}
			};
		}
		static AddPropertyMap<CElementSampler> compAddPropertyMap();

		inline CElementSampler(const char* id) : IElement(id), type(INVALID), sampleCount(4) {}
		inline ~CElementSampler() {}

		inline void initialize()
		{		
			sampleCount = 4;
			switch (type)
			{
				case CElementSampler::Type::STRATIFIED:
					[[fallthrough]];
				case CElementSampler::Type::LDSAMPLER:
					dimension = 4;
					break;
				case CElementSampler::Type::HALTON:
					[[fallthrough]];
				case CElementSampler::Type::HAMMERSLEY:
					scramble = -1;
					break;
				case CElementSampler::Type::SOBOL:
					scramble = 0;
					break;
				default:
					break;
			}
		}

		bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override;

		constexpr static inline auto ElementType = IElement::Type::SAMPLER;
		inline IElement::Type getType() const override { return ElementType; }
		inline std::string getLogName() const override { return "sampler"; }

		// make these public
		// TODO: these should be bitfields of a uint64_t, or pack into 8 bytes somehow
		Type type;
		int32_t sampleCount;
		union
		{
			int32_t dimension;
			// TODO: document scramble seed?
			int32_t scramble;
		};
};


}
#endif