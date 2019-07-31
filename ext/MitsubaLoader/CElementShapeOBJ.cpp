#include "CElementShapeOBJ.h"

#include "ParserUtil.h"
#include "CSimpleElement.h"

namespace irr { namespace ext { namespace MitsubaLoader {


CElementShapeOBJ::~CElementShapeOBJ()
{
	if (mesh)
		mesh->drop();
}

bool CElementShapeOBJ::processAttributes(const char** _atts)
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

bool CElementShapeOBJ::onEndTag(asset::IAssetManager& _assetManager, IElement* _parent)
{
	if (!fileName.size())
	{
		_IRR_DEBUG_BREAK_IF(true);
		//print error (name of the .obj file is not set)
		return false;
	}

	//getAssetInHierarchy?
	mesh = static_cast<asset::ICPUMesh*>(_assetManager.getAsset(fileName, asset::IAssetLoader::SAssetLoadParams()));

	if (!mesh)
		return false;

	//transform mesh by this->transformMatrix

	return _parent->processChildData(this);
}

bool CElementShapeOBJ::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::TRANSFORM:
		return true;

	case IElement::Type::STRING:
	{
		CElementString* stringElement = static_cast<CElementString*>(_child);

		if (stringElement->getNameAttribute() == "filename")
		{
			fileName = stringElement->getValueAttribute();
			return true;
		}
		else
		{
			return false;
		}
	}
	default:
		ParserLog::wrongChildElement(getLogName(), _child->getLogName());
		return false;
	}
}

}
}
}