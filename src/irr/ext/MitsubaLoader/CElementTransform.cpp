#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementFactory.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

	
template<>
CElementFactory::return_type CElementFactory::createElement<CElementTransform>(const char** _atts, ParserManager* _util)
{
	if (IElement::invalidAttributeCount(_atts, 2u))
		return CElementFactory::return_type(nullptr,"");
	if (core::strcmpi(_atts[0], "name"))
		return CElementFactory::return_type(nullptr,"");
	
	return CElementFactory::return_type(_util->objects.construct<CElementTransform>(),_atts[1]);
}

bool CElementTransform::addProperty(SNamedPropertyElement&& _property)
{
	switch (_property.type)
	{
		case SNamedPropertyElement::Type::MATRIX:
			_IRR_FALLTHROUGH;
		case SNamedPropertyElement::Type::TRANSLATE:
			_IRR_FALLTHROUGH;
		case SNamedPropertyElement::Type::ROTATE:
			_IRR_FALLTHROUGH;
		case SNamedPropertyElement::Type::SCALE:
			_IRR_FALLTHROUGH;
		case SNamedPropertyElement::Type::LOOKAT:
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