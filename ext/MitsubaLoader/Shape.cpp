#include "../../ext/MitsubaLoader/Shape.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "../../ext/MitsubaLoader/CShapeCreator.h"

namespace irr { namespace ext { namespace MitsubaLoader {

bool CShape::processAttributes(const char** _atts)
{
	//only type is an acceptable argument
	for (int i = 0; _atts[i]; i += 2)
	{
		if (std::strcmp(_atts[i], "type"))
		{
			ParserLog::invalidXMLFileStructure(std::string(_atts[i]) + " is not attribute of shape element.");
			return false;
		}
		else
		{
			type = _atts[i + 1];
		}
	}

	return true;
}

bool CShape::processChildData(IElement* _child)
{
	IElement::Type childType = _child->getType();

	switch (childType)
	{
	case IElement::Type::TRANSFORM:
	{
		CElementTransform* tr = static_cast<CElementTransform*>(_child);

		if (tr->getName() != "toWorld")
		{
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}

		transform = tr->getMatrix();

		return true;
	}
	//case::IElement::Type::EMITTER: return true;
	default:
	{
		ParserLog::invalidXMLFileStructure(_child->getLogName() + " is not a child of shape element.");
		return false;
	}
	}	
}

bool CShape::onEndTag(asset::IAssetManager* _assetManager)
{
	if (type == "cube")
	{
		mesh = CShapeCreator::createCube(_assetManager, properties);
	}
	else if (type == "sphere")
	{
		mesh = CShapeCreator::createSphere(_assetManager, properties, getTransformMatrix());
	}
	else if (type == "cylinder")
	{
		mesh = CShapeCreator::createCylinder(_assetManager, properties, getTransformMatrix());
	}
	else if (type == "rectangle")
	{
		mesh = CShapeCreator::createRectangle(_assetManager, properties);
	}
	else if (type == "disk")
	{
		mesh = CShapeCreator::createDisk(_assetManager, properties);
	}
	else if (type == "obj")
	{
		mesh = CShapeCreator::createOBJ(_assetManager, properties);
	}
	else if (type == "ply")
	{
		mesh = CShapeCreator::createPLY(_assetManager, properties);
	}
	else
	{
		ParserLog::invalidXMLFileStructure(type + " is not available shape type");
		return false;
	}

	if (mesh.get() == nullptr)
		return false;

	return true;
}

}
}
}