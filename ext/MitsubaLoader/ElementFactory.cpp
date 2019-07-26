#include "IElement.h"

#include <string>
#include <iostream>

#include "CElementShapeCube.h"

namespace irr { namespace ext { namespace MitsubaLoader {

//TODO: elementFactory should be an actuall class with distinct private member functions..

IElement* elementFactory(const char* _el, const char** _atts)
{
	//should be removing white spaces performed before string comparison?
	IElement* result = nullptr;

	if (!std::strcmp(_el, "shape"))
	{
		for (int i = 0; _atts[i]; i += 2)
		{
			if (!std::strcmp(_atts[i], "type"))
			{
				if (!std::strcmp(_atts[i + 1], "cube"))
				{
					result = new CElementShapeCube();
					result->processAttributes(_atts);
					std::cout << "We have done it! we created a cube! \n";
					return result;
				}
				else
				{
					std::cout << "invalid .xml file structure: " << _atts[i + 1] << "is not a type of shape element. \n";
					result = nullptr;
				}
			}
		}

		std::cout << "There is no type attribute for shape element. \n";
		
	}
	else
	{
		std::cout << "invalid .xml file structure: element " << _el << "is unknown. \n";
		return nullptr;
	}

	return nullptr;
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