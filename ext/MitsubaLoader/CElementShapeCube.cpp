#include "CElementShapeCube.h"

namespace irr { namespace ext { namespace MitsubaLoader {

CElementShapeCube::CElementShapeCube()
{

}

bool CElementShapeCube::processAttributes(const char** _args)
{
	//TODO: only type is an acceptable argument
	return true;
}

void CElementShapeCube::onEndTag(asset::IAssetManager& _assetManager, IElement* _parent)
{
	asset::ICPUMesh* cubeMesh = _assetManager.getGeometryCreator()->createCubeMesh(core::vector3df(2.0f, 2.0f, 2.0f));
	//transform cubeMesh by this->transformMatrix
	_parent->processChildData(this);
}

void CElementShapeCube::processChildData(IElement* child)
{
	switch (child->getType())
	{
	case IElement::Type::TO_WORLD_TRANSFORM:
		break;

	default:
		std::cout << "Invalid .xml file structure: this is not a child of shape cube element. \n";
		break;

	}
}

}
}
}