#include "CElementFactory.h"
#include "irrlicht.h"

#include <string>

#include "ParserUtil.h"
#include "CElementShapeCube.h"

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
				result = new CElementShapeCube();
				os::Printer::print("We have done it! we created a cube! \n");
				return result;
			}
			else
			{
				ParserLog::mitsubaLoaderError(std::string(_atts[i + 1]) + "is not a type of shape element. \n");
				return nullptr;
			}
		}
	}

	ParserLog::mitsubaLoaderError("There is no type attribute for shape element. \n");
}

//shape element processing loop
/*std::string shapeType = "";
for (int i = 0; _attr[i]; i += 2)
{
	if (!std::strcmp(_attr[i], "type"))
	{
		if (!std::strcmp(_attr[i + 1], "cube"))
		{
			return nullptr;
		}
		else
		{
			std::cout << _attr[i + 1] << " is unknown shape type. \n";
		}
	}
	else
	{
		std::cout << _attr[i] << "is unknown paramater of element shape.\n";
	}

}*/

}
}
}