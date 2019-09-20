#include "../../ext/MitsubaLoader/CMitsubaScene.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"

#include "../../ext/MitsubaLoader/Shape.h"

namespace irr { namespace ext { namespace MitsubaLoader {


bool CMitsubaScene::processAttributes(const char** _atts)
{
	if (IElement::areAttributesInvalid(_atts, 2u))
		return false;

	if (core::strcmpi(_atts[0], "version"))
	{
		ParserLog::invalidXMLFileStructure(std::string(_atts[0]) + " is not an attribute of scene element");
		return false;
	}
	else if (core::strcmpi(_atts[1], "0.5.0"))
	{
		ParserLog::invalidXMLFileStructure("Version "+std::string(_atts[1]) + " is unsupported");
		return false;
	}

	return true;
}

bool CMitsubaScene::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override) 
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
			_IRR_DEBUG_BREAK_IF(true);
			ParserLog::invalidXMLFileStructure(_child->getLogName() + " is not a child element of the scene element");
			return true;
	}

	return true;
}


}
}
}