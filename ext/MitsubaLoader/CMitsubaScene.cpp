#include "CMitsubaScene.h"
#include <iostream>
#include <string>

#include "irrlicht.h"
#include "ParserUtil.h"

namespace irr { namespace ext { namespace MitsubaLoader {


bool CMitsubaScene::processAttributes(const char** _atts)
{

	if (std::strcmp(_atts[0], "version"))
	{
		ParserLog::wrongAttribute(_atts[0], getLogName());
		return false;
	}
	else
	{
		//temporary solution
		std::cout << "version: " << _atts[1] << '\n';
		return true;
	}
}

bool CMitsubaScene::onEndTag(asset::IAssetManager& assetManager, IElement* parent) 
{
	return true;
}

bool CMitsubaScene::processChildData(IElement* _child)
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
		return false;

	case IElement::Type::SHAPE_CUBE:
		os::Printer::print("Alright! Cube has been added to the scene! \n");
		
		return true;

	default:
		ParserLog::wrongChildElement(getLogName(), _child->getLogName());

		return true;
	}
}


}
}
}