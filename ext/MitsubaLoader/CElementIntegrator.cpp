#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementFactory.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


template<>
IElement* CElementFactory::createElement<CElementSampler>(const char** _atts, ParserManager* _util)
{
	if (IElement::invalidAttributeCount(_atts, 2u))
		return nullptr;
	if (core::strcmpi(_atts[0], "type"))
		return nullptr;

	static const core::unordered_map<std::string, CElementSampler::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		std::make_pair("independent", CElementSampler::Type::INDEPENDENT),
		std::make_pair("stratified", CElementSampler::Type::STRATIFIED),
		std::make_pair("ldsampler", CElementSampler::Type::LDSAMPLER),
		std::make_pair("halton", CElementSampler::Type::HALTON),
		std::make_pair("hammersley", CElementSampler::Type::HAMMERSLEY),
		std::make_pair("sobol", CElementSampler::Type::SOBOL)
	};

	auto found = StringToType.find(_atts[1]);
	if (found==StringToType.end())
	{
		ParserLog::invalidXMLFileStructure("unknown type");
		_IRR_DEBUG_BREAK_IF(false);
		return nullptr;
	}

	CElementSampler* obj = _util->objects.construct<CElementSampler>();
	if (!obj)
		return nullptr;

	obj->type = found->second;
	obj->sampleCount = 4;
	//validation
	switch (obj->type)
	{
		case CElementSampler::Type::STRATIFIED:
			_IRR_FALLTHROUGH;
		case CElementSampler::Type::LDSAMPLER:
			obj->dimension = 4;
			break;
		case CElementSampler::Type::HALTON:
			_IRR_FALLTHROUGH;
		case CElementSampler::Type::HAMMERSLEY:
			obj->scramble = -1;
			break;
		case CElementSampler::Type::SOBOL:
			obj->scramble = 0;
			break;
		default:
			break;
	}
}

bool CElementSampler::addProperty(SPropertyElementData&& _property)
{
	if (_property.type == SPropertyElementData::Type::INTEGER &&
		_property.name == "sampleCount")
	{
		sampleCount = _property.ivalue;
		switch (type)
		{
			case Type::STRATIFIED:
				sampleCount = ceilf(sqrtf(sampleCount));
				break;
			case Type::LDSAMPLER:
				sampleCount = core::roundUpToPoT(sampleCount);
				break;
			default:
				break;
		}
	}
	else
	if (_property.type == SPropertyElementData::Type::INTEGER &&
		_property.name == "dimension")
	{
		dimension = _property.ivalue;
		if (type == Type::INDEPENDENT || type == Type::HALTON || type == Type::HAMMERSLEY || )
		{
			ParserLog::invalidXMLFileStructure("this sampler type does not take these parameters");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}
	}
	else
	if (_property.type == SPropertyElementData::Type::INTEGER &&
		_property.name == "scramble")
	{
		scramble = _property.ivalue;
		if (type==Type::INDEPENDENT || type==Type::STRATIFIED || type == Type::LDSAMPLER || )
		{
			ParserLog::invalidXMLFileStructure("this sampler type does not take these parameters");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}
	}
	else
	{
		_IRR_DEBUG_BREAK_IF(true);
		return false;
	}

	return true;
}

bool CElementSampler::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata)
{
	if (type == Type::NONE)
	{
		ParserLog::invalidXMLFileStructure(getLogName() + ": type not specified");
		_IRR_DEBUG_BREAK_IF(true);
		return true;
	}

	// add to global metadata
	if (type != Type::NONE)
	{
		ParserLog::invalidXMLFileStructure(getLogName() + ": cannot have two samplers in a scene");
		_IRR_DEBUG_BREAK_IF(true);
		return true;
	}
	else
		globalMetadata->sampler = *this;


	return true;
}

}
}
}