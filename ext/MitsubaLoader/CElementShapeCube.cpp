#include "CElementShapeCube.h"

#include "ParserUtil.h"

namespace irr { namespace ext { namespace MitsubaLoader {

CElementShapeCube::CElementShapeCube()
{

}

bool CElementShapeCube::processAttributes(const char** _atts)
{
	os::Printer::print("SHAPE CUBE ON BEGIN TAG");

	//only type is an acceptable argument
	for (int i = 0; _atts[i]; i += 2)
	{
		if (std::strcmp(_atts[i], "type"))
		{
			ParserLog::wrongAttribute(_atts[i], getName());
			return false;
		}
	}

	return true;
}

bool CElementShapeCube::onEndTag(asset::IAssetManager& _assetManager, IElement* _parent)
{
	asset::ICPUMesh* cubeMesh = _assetManager.getGeometryCreator()->createCubeMesh(core::vector3df(2.0f, 2.0f, 2.0f));

	os::Printer::print("SHAPE CUBE ON END TAG");

	if (!cubeMesh)
		return false;

	//transform cubeMesh by this->transformMatrix

	return _parent->processChildData(this);
}

bool CElementShapeCube::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::TO_WORLD_TRANSFORM:
		return true;

	default:
		ParserLog::wrongChildElement(getName(), _child->getName());
		return false;
	}
}

}
}
}