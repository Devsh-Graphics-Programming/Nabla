#include "../../ext/MitsubaLoader/bsdf/CElementBSDFDiffuse.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CSimpleElement.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "irr/asset/IMeshManipulator.h"

namespace irr { namespace ext { namespace MitsubaLoader {

bool CElementBSDFDiffuse::processAttributes(const char** _atts)
{
	//only type is an acceptable argument
	for (int i = 0; _atts[i]; i += 2)
	{
		if (std::strcmp(_atts[i], "type"))
		{
			ParserLog::wrongAttribute(_atts[i], getLogName());
			return false;
		}
	}

	return true;
}

bool CElementBSDFDiffuse::onEndTag(asset::IAssetManager& _assetManager, IElement* _parent)
{
	return _parent->processChildData(this);
}

bool CElementBSDFDiffuse::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::COLOR:
	{
		CElementString* stringElement = static_cast<CElementString*>(_child);
		std::string elementName = stringElement->getNameAttribute();

		if (elementName == "filename")
		{
			
		}
		else
		{
			ParserLog::mitsubaLoaderError("Unqueried attribute " + elementName + " in element \"shape\"");
		}

		return true;
	}
	case IElement::Type::TEXTURE:
	{
		CElementFloat* floatElement = static_cast<CElementFloat*>(_child);
		std::string elementName = floatElement->getNameAttribute();

		if (elementName == "maxSmoothAngle")
		{
		}
		else
		{
			//warning
			ParserLog::mitsubaLoaderError("Unqueried attribute " + elementName + " in element \"shape\"");
		}

		return true;

	}
	case IElement::Type::BOOLEAN:
	{
		CElementBoolean* boolElement = static_cast<CElementBoolean*>(_child);
		std::string elementName = boolElement->getNameAttribute();

		if (elementName == "faceNormals")
		{
			
		}
		else if (elementName == "flipNormals")
		{
			
		}
		else if (elementName == "flipTexCoords")
		{
			
		}
		else if (elementName == "collapse")
		{
			
		}
		else
		{
			//warning
			ParserLog::mitsubaLoaderError("Unqueried attribute " + elementName + " in element \"shape\"");
		}

		return true;
	}
	default:
		ParserLog::wrongChildElement(getLogName(), _child->getLogName());
		return false;
	}
}

}
}
}