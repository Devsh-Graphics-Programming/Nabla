#include "irr/ext/MitsubaLoader/ParserUtil.h"
#include "irr/ext/MitsubaLoader/CElementFactory.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


template<>
CElementFactory::return_type CElementFactory::createElement<CElementSampler>(const char** _atts, ParserManager* _util)
{
	const char* type;
	const char* id;
	std::string name;
	if (!IElement::getTypeIDAndNameStrings(type, id, name, _atts))
		return CElementFactory::return_type(nullptr, "");

	static const core::unordered_map<std::string, CElementSampler::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		std::make_pair("independent", CElementSampler::Type::INDEPENDENT),
		std::make_pair("stratified", CElementSampler::Type::STRATIFIED),
		std::make_pair("ldsampler", CElementSampler::Type::LDSAMPLER),
		std::make_pair("halton", CElementSampler::Type::HALTON),
		std::make_pair("hammersley", CElementSampler::Type::HAMMERSLEY),
		std::make_pair("sobol", CElementSampler::Type::SOBOL)
	};

	auto found = StringToType.find(type);
	if (found==StringToType.end())
	{
		ParserLog::invalidXMLFileStructure("unknown type");
		_IRR_DEBUG_BREAK_IF(false);
		return CElementFactory::return_type(nullptr, "");
	}

	CElementSampler* obj = _util->objects.construct<CElementSampler>(id);
	if (!obj)
		return CElementFactory::return_type(nullptr, "");

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
	return CElementFactory::return_type(obj, std::move(name));
}

bool CElementSampler::addProperty(SNamedPropertyElement&& _property)
{
	if (_property.type == SNamedPropertyElement::Type::INTEGER &&
		_property.name == "sampleCount")
	{
		sampleCount = _property.ivalue;
		switch (type)
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
	}
	else
	if (_property.type == SNamedPropertyElement::Type::INTEGER &&
		_property.name == "dimension")
	{
		dimension = _property.ivalue;
		if (type == Type::INDEPENDENT || type == Type::HALTON || type == Type::HAMMERSLEY)
		{
			ParserLog::invalidXMLFileStructure("this sampler type does not take these parameters");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}
	}
	else
	if (_property.type == SNamedPropertyElement::Type::INTEGER &&
		_property.name == "scramble")
	{
		scramble = _property.ivalue;
		if (type==Type::INDEPENDENT || type==Type::STRATIFIED || type == Type::LDSAMPLER)
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
	if (type == Type::INVALID)
	{
		ParserLog::invalidXMLFileStructure(getLogName() + ": type not specified");
		_IRR_DEBUG_BREAK_IF(true);
		return true;
	}

	// TODO: Validation

	return true;
}

}
}
}