#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementFactory.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

	
template<>
IElement* CElementFactory::createElement<CElementTransform>(const char** _atts, ParserManager* _util)
{
	if (IElement::invalidAttributeCount(_atts, 2u))
		return nullptr;
	if (core::strcmpi(_atts[0], "name"))
		return nullptr;
	
	return _util->objects.construct<CElementTransform>(_atts[1]);
}

bool CElementTransform::addProperty(SPropertyElementData&& _property)
{
	switch (_property.type)
	{
		case SPropertyElementData::Type::MATRIX:
			_IRR_FALLTHROUGH;
		case SPropertyElementData::Type::TRANSLATE:
			_IRR_FALLTHROUGH;
		case SPropertyElementData::Type::ROTATE:
			_IRR_FALLTHROUGH;
		case SPropertyElementData::Type::SCALE:
			_IRR_FALLTHROUGH;
		case SPropertyElementData::Type::LOOKAT:
			matrix = core::concatenateBFollowedByA(_property.mvalue, matrix);
			break;
		default:
			{
				ParserLog::invalidXMLFileStructure("The transform element does not take child property: "+_property.type);
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
			break;
	}

	return true;
}

}
}
}