// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CElementFactory.h"
#include "nbl/ext/MitsubaLoader/CElementEmissionProfile.h"

#include <functional>

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

template<>
CElementFactory::return_type CElementFactory::createElement<CElementEmissionProfile>(const char** _atts, ParserManager* _util)
{
	const char* type;
	const char* id;
	std::string name;
	if (!IElement::getTypeIDAndNameStrings(type, id, name, _atts))
		return CElementFactory::return_type(nullptr, "");

	CElementEmissionProfile* obj = _util->objects.construct<CElementEmissionProfile>(id);
	if (!obj)
		return CElementFactory::return_type(nullptr, "");

	return CElementFactory::return_type(obj, std::move(name));
}

bool CElementEmissionProfile::addProperty(SNamedPropertyElement&& _property)  {
	if (_property.name == "filename") {
		if (_property.type != SPropertyElementData::Type::STRING) {
			return false;
		}
		filename = _property.getProperty<SPropertyElementData::Type::STRING>();
		return true;
	}
	else if (_property.name == "normalizeEnergy") {
		if (_property.type != SPropertyElementData::Type::FLOAT) {
			return false;
		}
		normalizeEnergy = _property.getProperty<SPropertyElementData::Type::FLOAT>();
		return true;
	}
	else {
		ParserLog::invalidXMLFileStructure("No emission profile can have such property set with name: " + _property.name);
		return false;
	}
}

bool CElementEmissionProfile::processChildData(IElement* _child, const std::string& name)
{
	if (!_child)
		return true;
	return false;
}

}
}
}