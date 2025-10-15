// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CElementTransform.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"


namespace nbl::ext::MitsubaLoader
{
	
template<>
auto ParserManager::createElement<CElementTransform>(const char** _atts, SessionContext* ctx) -> SNamedElement
{
	if (IElement::invalidAttributeCount(_atts,2u))
		return {};
	if (core::strcmpi(_atts[0],"name"))
		return {};
	
	return {ctx->objects.construct<CElementTransform>(),_atts[1]};
}

bool CElementTransform::addProperty(SNamedPropertyElement&& _property, system::logger_opt_ptr logger)
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
			matrix = hlsl::mul(matrix,_property.mvalue);
			break;
		default:
			{
				invalidXMLFileStructure(logger,"The transform element does not take child property: "+_property.type);
				return false;
			}
			break;
	}

	return true;
}

}