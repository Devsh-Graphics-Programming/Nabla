#include "CElementFactory.h"
#include "irrlicht.h"

#include <string>

#include "ParserUtil.h"
#include "IElement.h"
#include "CElementShapeCube.h"
#include "CElementShapeOBJ.h"
#include "CSimpleElement.h"

namespace irr { namespace ext { namespace MitsubaLoader {

//TODO: elementFactory should be an actuall class with distinct private member functions..

IElement* CElementFactory::createElement(const char* _el, const char** _atts)
{
	//should be removing white spaces performed before string comparison?
	IElement* result = nullptr;
	if (!std::strcmp(_el, "scene"))
	{
		return parseScene(_el, _atts);
	}
	if (!std::strcmp(_el, "shape"))
	{
		return parseShape(_el, _atts);
	}
	if (!std::strcmp(_el, "float"))
	{
		return parseSimpleElement(_el, _atts, IElement::Type::FLOAT);
	}
	if (!std::strcmp(_el, "integer"))
	{
		return parseSimpleElement(_el, _atts, IElement::Type::INTEGER);
	}
	if (!std::strcmp(_el, "boolean"))
	{
		return parseSimpleElement(_el, _atts, IElement::Type::BOOLEAN);
	}
	if (!std::strcmp(_el, "point"))
	{
		return parseSimpleElement(_el, _atts, IElement::Type::POINT);
	}
	if (!std::strcmp(_el, "string"))
	{
		return parseSimpleElement(_el, _atts, IElement::Type::STRING);
	}
	else
	{
		ParserLog::mitsubaLoaderError("invalid .xml file structure: element " + std::string(_el) + "is unknown. \n");
		return nullptr;
	}
}

IElement* CElementFactory::parseScene(const char* _el, const char** _atts)
{
	return new CMitsubaScene();
}

IElement* CElementFactory::parseShape(const char* _el, const char** _atts)
{
	IElement* result = nullptr;

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "type"))
		{
			if (!std::strcmp(_atts[i + 1], "cube"))
			{
				return new CElementShapeCube();
			}
			if (!std::strcmp(_atts[i + 1], "obj"))
			{
				return new CElementShapeOBJ();
			}
			else
			{
				ParserLog::mitsubaLoaderError(std::string(_atts[i + 1]) + "is not a type of shape element. \n");
				return nullptr;
			}
		}
	}

	ParserLog::mitsubaLoaderError("There is no type attribute for shape element. \n");
	return nullptr;
}

IElement* CElementFactory::parseSimpleElement(const char* _el, const char** _atts, IElement::Type type)
{
	switch (type)
	{
	case IElement::Type::FLOAT:
		return new CElementFloat();

	case IElement::Type::STRING:
		return new CElementString();

	case IElement::Type::INTEGER:
	{
		//not implemented
		_IRR_DEBUG_BREAK_IF(true);
		return nullptr;
	}
	case IElement::Type::BOOLEAN:
		return new CElementBoolean();

	case IElement::Type::POINT:
		return new CElementPoint();

	default:
		_IRR_DEBUG_BREAK_IF(true);
		return nullptr;

	}
}

}
}
}