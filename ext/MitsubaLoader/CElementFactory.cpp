#include "CElementFactory.h"
#include "irrlicht.h"

#include <string>

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/IElement.h"
#include "../../ext/MitsubaLoader/CElementShapeCube.h"
#include "../../ext/MitsubaLoader/CElementSphere.h"
#include "../../ext/MitsubaLoader/CElementShapeOBJ.h"
#include "../../ext/MitsubaLoader/CElementShapeCylinder.h"
#include "../../ext/MitsubaLoader/CSimpleElement.h"
#include "../../ext/MitsubaLoader/CElementMatrix.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "../../ext/MitsubaLoader/CElementColor.h"
#include "../../ext/MitsubaLoader/CElementShapePLY.h"

namespace irr { namespace ext { namespace MitsubaLoader {

//TODO: elementFactory should be an actuall class with distinct private member functions..

IElement* CElementFactory::createElement(const char* _el, const char** _atts)
{
	//should be removing white spaces performed before string comparison?
	IElement* result = nullptr;
	if (!std::strcmp(_el, "rgb"))
	{
		return new CElementColor();
	}
	else
	if (!std::strcmp(_el, "srgb"))
	{
		return new CElementColor(true);
	}
	else
	if (!std::strcmp(_el, "scene"))
	{
		return parseScene(_el, _atts);
	}
	else
	if (!std::strcmp(_el, "shape"))
	{
		return parseShape(_el, _atts);
	}
	else
	if (!std::strcmp(_el, "float"))
	{
		return parseSimpleElement(_el, _atts, IElement::Type::FLOAT);
	}
	else
	if (!std::strcmp(_el, "integer"))
	{
		return parseSimpleElement(_el, _atts, IElement::Type::INTEGER);
	}
	else
	if (!std::strcmp(_el, "boolean"))
	{
		return parseSimpleElement(_el, _atts, IElement::Type::BOOLEAN);
	}
	else
	if (!std::strcmp(_el, "point"))
	{
		return parseSimpleElement(_el, _atts, IElement::Type::POINT);
	}
	else
	if (!std::strcmp(_el, "string"))
	{
		return parseSimpleElement(_el, _atts, IElement::Type::STRING);
	}
	else
	if (!std::strcmp(_el, "matrix"))
	{
		return parseMatrix(_el, _atts, CElementMatrix::Type::ARBITRARY);
	}
	else
	if (!std::strcmp(_el, "translate"))
	{
		return parseMatrix(_el, _atts, CElementMatrix::Type::TRANSLATION);
	}
	else
	if (!std::strcmp(_el, "rotate"))
	{
		return parseMatrix(_el, _atts, CElementMatrix::Type::ROTATION);
	}
	else
	if (!std::strcmp(_el, "scale"))
	{
		return parseMatrix(_el, _atts, CElementMatrix::Type::SCALE);
	}
	else
	if (!std::strcmp(_el, "transform"))
	{
		return new CElementTransform();
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
			else if (!std::strcmp(_atts[i + 1], "obj"))
			{
				return new CElementShapeOBJ();
			}
			else if (!std::strcmp(_atts[i + 1], "ply"))
			{
				return new CElementShapePLY();
			}
			else if (!std::strcmp(_atts[i + 1], "sphere"))
			{
				return new CElementSphere();
			}
			else if (!std::strcmp(_atts[i + 1], "cylinder"))
			{
				return new CElementShapeCylinder();
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

IElement* CElementFactory::parseMatrix(const char* _el, const char** _atts, CElementMatrix::Type type)
{
	return new CElementMatrix(type);
}

}
}
}