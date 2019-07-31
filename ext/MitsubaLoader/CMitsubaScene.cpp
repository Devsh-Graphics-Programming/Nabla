#include "CMitsubaScene.h"
#include <iostream>
#include <string>

#include "irrlicht.h"
#include "ParserUtil.h"

#include "CElementSHapeOBJ.h"
#include "CSimpleElement.h"

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
	{
		CElementShapeOBJ* shape = static_cast<CElementShapeOBJ*>(_child);
		const asset::ICPUMesh* shapeMesh = shape->getMesh();


		if (!shapeMesh)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}
			

		//is it possible that shapeMesh->getMeshBufferCount() > 1 ?
		for (int i = 0; i < shapeMesh->getMeshBufferCount(); i++)
			mesh->addMeshBuffer(shapeMesh->getMeshBuffer(i));
		


		return true;
	}
	case IElement::Type::SHAPE_CUBE:
		os::Printer::print("Alright! Cube has been added to the scene! \n");
		
		return true;

	case IElement::Type::FLOAT:
		
		std::cout << static_cast<CElementFloat*>(_child)->getValueAttribute() << std::endl;

		return true;


	default:
		ParserLog::wrongChildElement(getLogName(), _child->getLogName());

		return true;
	}
}


}
}
}