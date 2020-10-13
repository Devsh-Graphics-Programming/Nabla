#include "irr/ext/MitsubaLoader/CShapeCreator.h"
#include "irr/ext/MitsubaLoader/PropertyElement.h"
#include "irr/ext/MitsubaLoader/ParserUtil.h"

namespace irr { namespace ext { namespace MitsubaLoader {

core::smart_refctd_ptr<asset::ICPUMesh> CShapeCreator::createSphere(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties, core::matrix4SIMD& transform)
{
#ifdef NEW_MITSUBA
	bool flipNormalsFlag = false;
	core::vector3df_SIMD center;
	float radius = 1.0f;

	for (int i = 0; i < properties.size(); i++)
	{
		if (properties[i].type == SPropertyElementData::Type::BOOLEAN &&
			properties[i].name == "flipNormals")
		{
			flipNormalsFlag = CPropertyElementManager::retriveBooleanValue(properties[i].value);
		}
		else 
		if (properties[i].type == SPropertyElementData::Type::POINT &&
			properties[i].name == "center")
		{
			center = CPropertyElementManager::retriveVector(properties[i].value);
		}
		else
		if (properties[i].type == SPropertyElementData::Type::FLOAT &&
			properties[i].name == "radius")
		{
			radius = CPropertyElementManager::retriveFloatValue(properties[i].value);
		}
		else
		{
			ParserLog::invalidXMLFileStructure(properties[i].name + " wat is this?");
			return nullptr;
		}
	}

	auto mesh = _assetManager->getGeometryCreator()->createSphereMesh(radius, 32, 32);

	if (!mesh)
		return nullptr;

	if (flipNormalsFlag)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager->getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

	transform.setTranslation(transform.getTranslation() + center);

	return mesh;
#endif
	return nullptr;
}

core::smart_refctd_ptr<asset::ICPUMesh> CShapeCreator::createCylinder(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties, core::matrix4SIMD& _transform)
{
	_IRR_DEBUG_BREAK_IF(true);
	return nullptr;
}

core::smart_refctd_ptr<asset::ICPUMesh> CShapeCreator::createRectangle(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties)
{
#ifdef NEW_MITSUBA
	bool flipNormalsFlag = false;

	for (int i = 0; i < properties.size(); i++)
	{
		if (properties[i].type == SPropertyElementData::Type::BOOLEAN &&
			properties[i].name == "flipNormals")
		{
			flipNormalsFlag = CPropertyElementManager::retriveBooleanValue(properties[i].value);
		}
		else
		{
			ParserLog::invalidXMLFileStructure(properties[i].name + " wat is this?");
			return nullptr;
		}
	}

	auto mesh = _assetManager->getGeometryCreator()->createRectangleMesh(core::vector2df_SIMD(1.0f, 1.0f));

	if (!mesh)
		return nullptr;

	if (flipNormalsFlag)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager->getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

	return mesh;
}

core::smart_refctd_ptr<asset::ICPUMesh> CShapeCreator::createDisk(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties)
{
	bool flipNormalsFlag = false;

	for (int i = 0; i < properties.size(); i++)
	{
		if (properties[i].type == SPropertyElementData::Type::BOOLEAN &&
			properties[i].name == "flipNormals")
		{
			flipNormalsFlag = CPropertyElementManager::retriveBooleanValue(properties[i].value);
		}
		else
		{
			ParserLog::invalidXMLFileStructure(properties[i].name + " wat is this?");
			_IRR_DEBUG_BREAK_IF(true);
			return nullptr;
		}
	}

	auto mesh = _assetManager->getGeometryCreator()->createDiskMesh(1.0f, 64);

	if (!mesh)
		return nullptr;

	if (flipNormalsFlag)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager->getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

	return mesh;
#endif
	return nullptr;
}

core::smart_refctd_ptr<asset::ICPUMesh> CShapeCreator::createOBJ(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties)
{
#ifdef NEW_MITSUBA
	std::string fileName;
	bool flipNormalsFlag = false;
	bool faceNormals = false;
	bool smoothNormals = false;
	float maxSmoothAngle = 0.0f;
	bool flipTexCoords = true;
	bool collapse = false;

	for (int i = 0; i < properties.size(); i++)
	{
		if (properties[i].type == SPropertyElementData::Type::BOOLEAN &&
			properties[i].name == "flipNormals")
		{
			flipNormalsFlag = CPropertyElementManager::retriveBooleanValue(properties[i].value);
		}
		else
		if (properties[i].type == SPropertyElementData::Type::STRING &&
			properties[i].name == "filename")
		{
			fileName = properties[i].value;
		}
		else
		if (properties[i].type == SPropertyElementData::Type::FLOAT &&
			properties[i].name == "maxSmoothAngle")
		{
			smoothNormals = true;
			maxSmoothAngle = CPropertyElementManager::retriveFloatValue(properties[i].value);
		}
		else
		if (properties[i].type == SPropertyElementData::Type::BOOLEAN &&
			properties[i].name == "flipTexCoords")
		{
			flipTexCoords = CPropertyElementManager::retriveBooleanValue(properties[i].value);
		}
		else
		if (properties[i].type == SPropertyElementData::Type::BOOLEAN &&
			properties[i].name == "collapse")
		{
			flipTexCoords = CPropertyElementManager::retriveBooleanValue(properties[i].value);
		}
		else
		if (properties[i].type == SPropertyElementData::Type::BOOLEAN &&
			properties[i].name == "faceNormals")
		{
			faceNormals = CPropertyElementManager::retriveBooleanValue(properties[i].value);
		}
		else
		{
			ParserLog::invalidXMLFileStructure(properties[i].name + " wat is this?");
			return nullptr;
		}
	}

	if (fileName == "")
	{
		ParserLog::invalidXMLFileStructure("file name not set");
		return nullptr;
	}

	core::smart_refctd_ptr<asset::ICPUMesh> mesh;
	{
		asset::SAssetBundle bundle = _assetManager->getAsset(fileName, asset::IAssetLoader::SAssetLoadParams());
		mesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*bundle.getContents().first);
	}

	if (!mesh)
		return nullptr;

	else if (smoothNormals)
	{
		auto newMesh = core::make_smart_refctd_ptr<asset::CCPUMesh>();
		asset::IMeshManipulator::SErrorMetric metrics[16];

		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
		{
			auto upBuffer = _assetManager->getMeshManipulator()->createMeshBufferUniquePrimitives(mesh->getMeshBuffer(i));
			const float smoothAngleCos = cos(maxSmoothAngle);

			_assetManager->getMeshManipulator()->calculateSmoothNormals(upBuffer.get(), false, 1.525e-5f, asset::EVAI_ATTR3,
				[&](const asset::IMeshManipulator::SSNGVertexData& a, const asset::IMeshManipulator::SSNGVertexData& b, asset::ICPUMeshBuffer* buffer)
				{
					return a.parentTriangleFaceNormal.dotProductAsFloat(b.parentTriangleFaceNormal) >= smoothAngleCos;
				});

			_assetManager->getMeshManipulator()->createMeshBufferWelded(upBuffer.get(), metrics);

			newMesh->addMeshBuffer(std::move(upBuffer));
		}

		mesh = newMesh;
	}

	if (flipTexCoords)
	{
		//TODO
		assert(false);
	}

	if (collapse)
	{
		//TODO
		assert(false);
	}

	return mesh;
#endif
	return nullptr;
}

core::smart_refctd_ptr<asset::ICPUMesh> CShapeCreator::createPLY(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties)
{
	return nullptr;
}

}
}
}