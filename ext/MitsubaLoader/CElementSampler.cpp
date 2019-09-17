#include "../../ext/MitsubaLoader/CElementSampler.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/PropertyElement.h"

namespace irr { namespace ext { namespace MitsubaLoader {

bool CElementSampler::processAttributes(const char** _atts)
{
	static const core::unordered_map<std::string, ESamplerType> acceptableTypes = {
		std::make_pair("independent", ESamplerType::INDEPENDENT),
		std::make_pair("stratified", ESamplerType::STRATIFIED),
		std::make_pair("ldsampler", ESamplerType::LDSAMPLER),
		std::make_pair("halton", ESamplerType::HALTON),
		std::make_pair("hammersley", ESamplerType::HAMMERSLEY),
		std::make_pair("sobol", ESamplerType::SOBOL)
	};

	//only type is an acceptable argument
	for (int i = 0; _atts[i]; i += 2)
	{
		if (std::strcmp(_atts[i], "type"))
		{
			ParserLog::invalidXMLFileStructure(std::string(_atts[i]) + " is not attribute of shape element.");
			return false;
		}
		else
		{
			auto samplerType = acceptableTypes.find(_atts[i + 1]);
			if (samplerType == acceptableTypes.end())
			{
				ParserLog::invalidXMLFileStructure("unknown type");
				_IRR_DEBUG_BREAK_IF(false);
				return false;
			}

			data.type = samplerType->second;
		}
	}

	return true;
}

bool CElementSampler::onEndTag(asset::IAssetManager* _assetManager)
{
	bool dimensionSet = false;
	bool scrambleSet = false;

	for (const auto& property : properties)
	{
		if (property.type == SPropertyElementData::Type::INTEGER &&
			property.name == "sampleCount")
		{
			data.sampleCount = CPropertyElementManager::retriveIntValue(property.value);
		}
		else
		if (property.type == SPropertyElementData::Type::INTEGER &&
			property.name == "dimension")
		{
			data.dimension = CPropertyElementManager::retriveIntValue(property.value);
			dimensionSet = true;
		}
		else
		if (property.type == SPropertyElementData::Type::INTEGER &&
			property.name == "scramble")
		{
			data.scramble = CPropertyElementManager::retriveIntValue(property.value);
			scrambleSet = true;
		}
		else
		{
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}
	}

	//validation
	switch (data.type)
	{
	case ESamplerType::NONE:
	{
		ParserLog::invalidXMLFileStructure(getLogName() + ": type not specified");

		_IRR_DEBUG_BREAK_IF(true);
		return false;
	}
	case ESamplerType::INDEPENDENT:
	{
		if (dimensionSet || scrambleSet)
		{
			ParserLog::invalidXMLFileStructure("eeeeee");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}

		data.dimension = 0;

		return true;
	}
	case ESamplerType::STRATIFIED:
	case ESamplerType::LDSAMPLER:
	{
		if (scrambleSet)
		{
			ParserLog::invalidXMLFileStructure("eeeeee");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}

		return true;
	}
	case ESamplerType::HALTON:
	case ESamplerType::HAMMERSLEY:
	case ESamplerType::SOBOL:
	{
		if (dimensionSet)
		{
			ParserLog::invalidXMLFileStructure("eeeeee");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}

		//default value
		if (!scrambleSet)
			data.scramble = (data.type == ESamplerType::SOBOL) ? 0 : -1;

		return true;
	}
	
	}

	return true;
}

}
}
}