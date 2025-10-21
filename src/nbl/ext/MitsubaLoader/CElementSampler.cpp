// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CElementSampler.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/ElementMacros.h"


namespace nbl::ext::MitsubaLoader
{
	
auto CElementSampler::compAddPropertyMap() -> AddPropertyMap<CElementSampler>
{
	using this_t = CElementSampler;
	AddPropertyMap<CElementSampler> retval;

	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("sampleCount",INTEGER)
		{
			auto& sampleCount = _this->sampleCount;
			sampleCount = _property.ivalue;
			switch (_this->type)
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
			return true;
		}
	});
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("dimension",INTEGER)
		{
			_this->dimension = _property.ivalue;
			switch (_this->type)
			{
				case Type::INDEPENDENT: [[fallthrough]];
				case Type::HALTON: [[fallthrough]];
				case Type::HAMMERSLEY:
					invalidXMLFileStructure(logger,"this sampler type ("+std::to_string(_this->type)+") does not take these parameters");
					return false;
				default:
					return true;
			}
		}
	});
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("scramble",INTEGER)
		{
			_this->scramble = _property.ivalue;
			switch (_this->type)
			{
				case Type::INDEPENDENT: [[fallthrough]];
				case Type::HALTON: [[fallthrough]];
				case Type::HAMMERSLEY:
					invalidXMLFileStructure(logger,"this sampler type ("+std::to_string(_this->type)+") does not take these parameters");
					return false;
				default:
					return true;
			}
		}
	});

	return retval;
}

bool CElementSampler::onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger)
{
	NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(true);

	// TODO: Validation

	return true;
}

}