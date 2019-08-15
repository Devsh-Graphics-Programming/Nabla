#include "../../ext/MitsubaLoader/CElementShapeOBJ.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CSimpleElement.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "irr/asset/IMeshManipulator.h"

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
		ParserLog::mitsubaLoaderError("property filename has been not specified");
		_IRR_DEBUG_BREAK_IF(true);
		return false;
	}

	mesh = static_cast<asset::ICPUMesh*>(_assetManager.getAsset(fileName, asset::IAssetLoader::SAssetLoadParams()));

	if (!mesh)
		return false;

	if (flipNormalsFlag)
		flipNormals(_assetManager);

	if (faceNormals)
	{
		asset::SCPUMesh* newMesh = new asset::SCPUMesh();

		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
		{
			asset::ICPUMeshBuffer* upBuffer = _assetManager.getMeshManipulator()->createMeshBufferUniquePrimitives(mesh->getMeshBuffer(i));

			_assetManager.getMeshManipulator()->calculateSmoothNormals(upBuffer, false, 1.525e-5f, asset::EVAI_ATTR3,
				[](const asset::IMeshManipulator::SSNGVertexData& a, const asset::IMeshManipulator::SSNGVertexData& b, asset::ICPUMeshBuffer* buffer)
				{
					return a.indexOffset == b.indexOffset;
				});

			newMesh->addMeshBuffer(upBuffer);
		}

		mesh = newMesh;
	}
	else if (smoothNormals)
	{
		asset::SCPUMesh* newMesh = new asset::SCPUMesh();
		asset::IMeshManipulator::SErrorMetric metrics[16];

		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
		{
			asset::ICPUMeshBuffer* upBuffer = _assetManager.getMeshManipulator()->createMeshBufferUniquePrimitives(mesh->getMeshBuffer(i));
			const float smoothAngleCos = cos(maxSmoothAngle);

			_assetManager.getMeshManipulator()->calculateSmoothNormals(upBuffer, false, 1.525e-5f, asset::EVAI_ATTR3,
				[&](const asset::IMeshManipulator::SSNGVertexData& a, const asset::IMeshManipulator::SSNGVertexData& b, asset::ICPUMeshBuffer* buffer)
				{
					return a.parentTriangleFaceNormal.dotProductAsFloat(b.parentTriangleFaceNormal) >= smoothAngleCos;
				});

			newMesh->addMeshBuffer(upBuffer);

			_assetManager.getMeshManipulator()->createMeshBufferWelded(upBuffer, metrics);
		}

		mesh = newMesh;
	}
	
	

	if (flipTexCoords)
	{
		//TODO
	}

	if (collapse)
	{
		//TODO
	}

	return _parent->processChildData(this);
}

bool CElementShapeOBJ::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::TRANSFORM:
	{
		CElementTransform* transform = static_cast<CElementTransform*>(_child);

		if (transform->getName() == "toWorld")
			this->transform = static_cast<CElementTransform*>(_child)->getMatrix();
		else
			ParserLog::mitsubaLoaderError("Unqueried attribute '" + transform->getName() + "' in element 'shape'");
		return true;
	}
	case IElement::Type::STRING:
	{
		CElementString* stringElement = static_cast<CElementString*>(_child);
		std::string elementName = stringElement->getNameAttribute();

		if (elementName == "filename")
		{
			fileName = stringElement->getValueAttribute();
		}
		else
		{
			ParserLog::mitsubaLoaderError("Unqueried attribute " + elementName + " in element \"shape\"");
		}

		return true;
	}
	case IElement::Type::FLOAT:
	{
		CElementFloat* floatElement = static_cast<CElementFloat*>(_child);
		std::string elementName = floatElement->getNameAttribute();

		if (elementName == "maxSmoothAngle")
		{
			maxSmoothAngle = floatElement->getValueAttribute();
			smoothNormals = true;
		}
		else
		{
			//warning
			ParserLog::mitsubaLoaderError("Unqueried attribute " + elementName + " in element \"shape\"");
		}

		return true;

	}
	case IElement::Type::BOOLEAN:
	{
		CElementBoolean* boolElement = static_cast<CElementBoolean*>(_child);
		std::string elementName = boolElement->getNameAttribute();

		if (elementName == "faceNormals")
		{
			faceNormals = boolElement->getValueAttribute();
		}
		else if(elementName == "flipNormals")
		{
			flipNormalsFlag = boolElement->getValueAttribute();
		}
		else if (elementName == "flipTexCoords")
		{
			flipTexCoords = boolElement->getValueAttribute();
		}
		else if (elementName == "collapse")
		{
			collapse = boolElement->getValueAttribute();
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