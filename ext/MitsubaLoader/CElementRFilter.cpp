#include "irr/core/string/stringutil.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementFactory.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


template<>
CElementFactory::return_type CElementFactory::createElement<CElementRFilter>(const char** _atts, ParserManager* _util)
{
	const char* type;
	const char* id;
	std::string name;
	if (!IElement::getTypeIDAndNameStrings(type, id, name, _atts))
		return CElementFactory::return_type(nullptr, "");

	static const core::unordered_map<std::string,CElementRFilter::Type, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
	{
		std::make_pair("box", CElementRFilter::Type::BOX),
		std::make_pair("tent", CElementRFilter::Type::TENT),
		std::make_pair("gaussian", CElementRFilter::Type::GAUSSIAN),
		std::make_pair("mitchell", CElementRFilter::Type::MITCHELL),
		std::make_pair("catmullrom", CElementRFilter::Type::CATMULLROM),
		std::make_pair("lanczos", CElementRFilter::Type::LANCZOS)
	};

	auto found = StringToType.find(type);
	if (found==StringToType.end())
	{
		ParserLog::invalidXMLFileStructure("unknown type");
		_IRR_DEBUG_BREAK_IF(false);
		return CElementFactory::return_type(nullptr, "");
	}

	CElementRFilter* obj = _util->objects.construct<CElementRFilter>(id);
	if (!obj)
		return CElementFactory::return_type(nullptr, "");

	obj->type = found->second;
	//validation
	switch (obj->type)
	{
		case CElementRFilter::Type::BOX:
			_IRR_FALLTHROUGH;
		case CElementRFilter::Type::TENT:
			break;
		case CElementRFilter::Type::GAUSSIAN:
			obj->gaussian = CElementRFilter::Gaussian();
			break;
		case CElementRFilter::Type::MITCHELL:
			obj->mitchell = CElementRFilter::MitchellNetravali();
			break;
		case CElementRFilter::Type::CATMULLROM:
			obj->catmullrom = CElementRFilter::MitchellNetravali();
			break;
		case CElementRFilter::Type::LANCZOS:
			obj->lanczos = CElementRFilter::LanczosSinc();
			break;
		default:
			break;
	}
	return CElementFactory::return_type(obj, std::move(name));
}

bool CElementRFilter::addProperty(SNamedPropertyElement&& _property)
{
	if (_property.type == SNamedPropertyElement::Type::INTEGER)
	{
		if (core::strcmpi(_property.name,std::string("lobes")))
		{
			ParserLog::invalidXMLFileStructure("\"lobes\" must be an integer property");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}
		lanczos.lobes = _property.ivalue;
		return true;
	}
	else
	if (_property.type == SNamedPropertyElement::Type::FLOAT)
	{
		if (core::strcmpi(_property.name,std::string("b"))==0)
		{
			mitchell.B = _property.fvalue;
			return true;
		}
		else if (core::strcmpi(_property.name,std::string("c"))==0)
		{
			mitchell.C = _property.fvalue;
			return true;
		}
		else
			ParserLog::invalidXMLFileStructure("unsupported rfilter property called: "+_property.name);
	}
	else
	{
		ParserLog::invalidXMLFileStructure("this reconstruction filter type does not take this parameter type for parameter: " + _property.name);
		_IRR_DEBUG_BREAK_IF(true);
	}

	return false;
}

bool CElementRFilter::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata)
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