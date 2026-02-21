// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CElementFactory.h"

namespace nbl
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
			[[fallthrough]];
		case SNamedPropertyElement::Type::TRANSLATE:
			[[fallthrough]];
		case SNamedPropertyElement::Type::ROTATE:
			[[fallthrough]];
		case SNamedPropertyElement::Type::SCALE:
			[[fallthrough]];
		case SNamedPropertyElement::Type::LOOKAT:
			matrix = core::concatenateBFollowedByA(_property.mvalue, matrix);
			break;
		default:
			{
				ParserLog::invalidXMLFileStructure("The transform element does not take child property: "+_property.type);
				_NBL_DEBUG_BREAK_IF(true);
				return false;
			}
			break;
	}

	return true;
}

}
}
}