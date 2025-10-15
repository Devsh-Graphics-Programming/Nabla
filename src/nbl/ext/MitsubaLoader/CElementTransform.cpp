// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CElementTransform.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"


namespace nbl::ext::MitsubaLoader
{

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