#include "CMitsubaScene.h"
#include <iostream>
#include <string>

namespace irr { namespace ext { namespace MitsubaLoader {


bool CMitsubaScene::processAttributes(const char** _atts)
{
	if (std::strcmp(_atts[0], "version"))
	{
		std::cout << "Invalid .xml file structure: " << _atts[0] << "is not attribute of the scene element \n";
		return false;
		//return false and then stop parsing and return nullptr from CMitsubaLoader::loadAsset...
	}
	else
	{
		//temporary solution
		std::cout << "version: " << _atts[1] << '\n';
		return true;
	}
}

void CMitsubaScene::onEndTag(asset::IAssetManager& assetManager, IElement* parent) 
{

}

void CMitsubaScene::processChildData(IElement* _child)
{
	//general idea is to retrive all asset data from child elements and put it in this->mesh
	switch (_child->getType())
	{
	case IElement::Type::SHAPE_OBJ:
		//add contents of the mesh held by child here
		//for example:
		/*
		CElementShapeObj childObjElement = static_cast<CElementShapeObj*>(_child);
		_child->getMesh()->getMeshBufferCount();
		*/
	break;
	case IElement::Type::SHAPE_CUBE:
		std::cout << "Alright! Cube has been added to the scene! \n";
		
	break;
	default:
		std::cout << "Invalid .xml file structure: this is not a child of the scene element \n";
	break;
	}
}


}
}
}