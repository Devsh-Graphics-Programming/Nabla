#include "../../ext/MitsubaLoader/CMitsubaScene.h"
#include <iostream>
#include <string>

#include "irrlicht.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"

#include "../../ext/MitsubaLoader/Shape.h"
#include "matrix4SIMD.h"
#include "../../ext/MitsubaLoader/CElementSampler.h"

namespace irr { namespace ext { namespace MitsubaLoader {


bool CMitsubaScene::processAttributes(const char** _atts)
{
	if (std::strcmp(_atts[0], "version"))
	{
		ParserLog::invalidXMLFileStructure(std::string(_atts[0]) + " is not an attribute of scene element");
		return false;
	}
	else
	{
		//temporary solution
		std::cout << "version: " << _atts[1] << '\n';
		return true;
	}

	return false;
}

bool CMitsubaScene::onEndTag(asset::IAssetManager* _assetManager) 
{
	return true;
}

bool CMitsubaScene::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::SHAPE:
	{
		CShape* shape = static_cast<CShape*>(_child);

		if (!shape)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}

		const core::smart_refctd_ptr<asset::ICPUMesh> shapeMesh = shape->getMesh();
		meshes.push_back(core::smart_refctd_ptr<asset::ICPUMesh>(shapeMesh));

		return true;
	}
	default:
		ParserLog::invalidXMLFileStructure(_child->getLogName() + " is not a child element of the scene element");
		return true;
	}

	return true;
}


}
}
}