#include "../../ext/MitsubaLoader/CMitsubaScene.h"
#include <iostream>
#include <string>

#include "irrlicht.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"

#include "../../ext/MitsubaLoader/CElementShapeOBJ.h"
#include "../../ext/MitsubaLoader/CElementShapeCube.h"
#include "../../ext/MitsubaLoader/CSimpleElement.h"
#include "../../ext/MitsubaLoader/IShape.h"
#include "../../include/matrix4SIMD.h"

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
	mesh->recalculateBoundingBox();

	return true;
}

bool CMitsubaScene::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::SHAPE:
	{
		IShape* shape = dynamic_cast<IShape*>(_child);

		if (!shape)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}

		const asset::ICPUMesh* shapeMesh = shape->getMesh();
		return appendMesh(shapeMesh, shape->getTransformMatrix());
	}
	default:
		ParserLog::wrongChildElement(getLogName(), _child->getLogName());

		return true;
	}
}

bool CMitsubaScene::appendMesh(const asset::ICPUMesh* _mesh, const core::matrix4SIMD& _transform)
{
	if (!_mesh)
	{
		_IRR_DEBUG_BREAK_IF(true);
		return false;
	}


	_mesh->grab();

	for (int i = 0; i < _mesh->getMeshBufferCount(); i++)
	{
		//here vertex positions and normals are premultiplied with world transfrom matrix
		//TODO: skip this step if transform is identity matrix

		asset::ICPUMeshBuffer* submesh = _mesh->getMeshBuffer(i);
		
		core::matrix4SIMD normalTransform; //normalTransform = transpose(inverse(_transform));
		_transform.getInverseTransform(normalTransform);
		normalTransform = normalTransform.getTransposed();

		const size_t vxCount = submesh->calcVertexCount();
		for (int i = 0; i < vxCount; i++)
		{
			core::vectorSIMDf pos = submesh->getPosition(i);
			core::vectorSIMDf normal;
			submesh->getAttribute(normal, asset::E_VERTEX_ATTRIBUTE_ID::EVAI_ATTR3, i);

			_transform.transformVect(pos);
			normalTransform.transformVect(normal);

			submesh->setAttribute(pos,submesh->getPositionAttributeIx(), i);
			submesh->setAttribute(normal, asset::E_VERTEX_ATTRIBUTE_ID::EVAI_ATTR3, i);

			
		}
		
		mesh->addMeshBuffer(submesh);
	}

	return true;
}


}
}
}