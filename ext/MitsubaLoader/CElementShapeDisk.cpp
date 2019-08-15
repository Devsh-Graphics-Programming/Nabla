#include "../../ext/MitsubaLoader/CElementShapeDisk.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "../../ext/MitsubaLoader/CSimpleElement.h"

namespace irr { namespace ext { namespace MitsubaLoader {


bool CElementShapeDisk::processAttributes(const char** _atts)
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

bool CElementShapeDisk::onEndTag(asset::IAssetManager& _assetManager, IElement* _parent)
{
	mesh = _assetManager.getGeometryCreator()->createDiskMesh(1.0f, 64);

	if (!mesh)
		return false;

	if (flipNormalsFlag)
		flipNormals(_assetManager);

	return _parent->processChildData(this);
}

bool CElementShapeDisk::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::TRANSFORM:
	{
		CElementTransform* transformElement = static_cast<CElementTransform*>(_child);

		if (transformElement->getName() == "toWorld")
			this->transform = static_cast<CElementTransform*>(_child)->getMatrix();
		else
			ParserLog::mitsubaLoaderError("Unqueried attribute '" + transformElement->getName() + "' in element 'shape'");

		return true;
	}
	case IElement::Type::BOOLEAN:
	{
		CElementBoolean* boolElement = static_cast<CElementBoolean*>(_child);
		const std::string  elementName = boolElement->getNameAttribute();

		if (elementName == "flipNormals")
		{
			flipNormalsFlag = boolElement->getValueAttribute();
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