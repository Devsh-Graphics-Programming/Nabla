#include "../../ext/MitsubaLoader/CShapeCreator.h"
#include "../../ext/MitsubaLoader/PropertyElement.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"

namespace irr { namespace ext { namespace MitsubaLoader {

asset::ICPUMesh* CShapeCreator::createCube(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties)
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

	asset::ICPUMesh* mesh = _assetManager.getGeometryCreator()->createCubeMesh(core::vector3df(1.0f, 1.0f, 1.0f));

	if (mesh == nullptr)
		return nullptr;

	if (flipNormalsFlag)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager.getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

	return mesh;
}

asset::ICPUMesh* CShapeCreator::createSphere(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties, core::matrix4SIMD& transform)
{
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

	asset::ICPUMesh* mesh = _assetManager.getGeometryCreator()->createSphereMesh(radius, 32, 32);

	if (mesh == nullptr)
		return nullptr;

	if (flipNormalsFlag)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager.getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

	transform.setTranslation(transform.getTranslation() + center);

	return mesh;
}

asset::ICPUMesh* CShapeCreator::createCylinder(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties, core::matrix4SIMD& _transform)
{
	/*bool flipNormalsFlag = false;
	core::vectorSIMDf p0(0.0f);
	core::vectorSIMDf p1(0.0f, 0.0f, 1.0f);
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
			properties[i].name == "p0")
		{
			p0 = CPropertyElementManager::retriveVector(properties[i].value);
		}
		else
		if (properties[i].type == SPropertyElementData::Type::POINT &&
			properties[i].name == "p0")
		{
			p1 = CPropertyElementManager::retriveVector(properties[i].value);
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

	const float cylinderHeight = (p0 - p1).getLengthAsFloat();

	asset::ICPUMesh* mesh = _assetManager.getGeometryCreator()->createCylinderMesh(radius, cylinderHeight, 32, 0xffffffff, false);
	if (!mesh)
		return false;

	core::vectorSIMDf vec = p1 - p0;
	vec /= vec.getLengthAsFloat();

	core::quaternion rotation = core::quaternion::rotationFromTo(core::vectorSIMDf(0.0f, 1.0f, 0.0f), vec);
	core::matrix4x3 rotationMatrix = rotation.getMatrix();

	core::vectorSIMDf r1;
	r1.x = rotationMatrix.getColumn(0).X;
	r1.y = rotationMatrix.getColumn(1).X;
	r1.z = rotationMatrix.getColumn(2).X;
	r1.w = rotationMatrix.getColumn(3).X;

	core::vectorSIMDf r2;
	r2.x = rotationMatrix.getColumn(0).Y;
	r2.y = rotationMatrix.getColumn(1).Y;
	r2.z = rotationMatrix.getColumn(2).Y;
	r2.w = rotationMatrix.getColumn(3).Y;

	core::vectorSIMDf r3;
	r3.x = rotationMatrix.getColumn(0).Z;
	r3.y = rotationMatrix.getColumn(1).Z;
	r3.z = rotationMatrix.getColumn(2).Z;
	r3.w = rotationMatrix.getColumn(3).Z;

	core::matrix4SIMD matrix(r1, r2, r3, core::vectorSIMDf(0.0f, 0.0f, 0.0f, 1.0f));
	matrix.setTranslation(p0);

	_transform = core::concatenateBFollowedByA(matrix, _transform);

	if (flipNormalsFlag)
	{
		{
			for (int i = 0; i < mesh->getMeshBufferCount(); i++)
				_assetManager.getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
		}
	}

	return mesh;*/
	_IRR_DEBUG_BREAK_IF(true);
	return nullptr;
}

asset::ICPUMesh* CShapeCreator::createRectangle(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties)
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
			return nullptr;
		}
	}

	asset::ICPUMesh* mesh = _assetManager.getGeometryCreator()->createRectangleMesh(core::vector2df_SIMD(1.0f, 1.0f));

	if (mesh == nullptr)
		return nullptr;

	if (flipNormalsFlag)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager.getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

	return mesh;
}

asset::ICPUMesh* CShapeCreator::createDisk(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties)
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

	asset::ICPUMesh* mesh = _assetManager.getGeometryCreator()->createDiskMesh(1.0f, 64);

	if (mesh == nullptr)
		return nullptr;

	if (flipNormalsFlag)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager.getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

	return mesh;
}

asset::ICPUMesh* CShapeCreator::createOBJ(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties)
{
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

	asset::SAssetBundle bundle = _assetManager.getAsset(fileName, asset::IAssetLoader::SAssetLoadParams());
	asset::ICPUMesh* mesh = static_cast<asset::ICPUMesh*>(bundle.getContents().first->get());

	if (mesh == nullptr)
		return nullptr;

	if (flipNormalsFlag)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager.getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

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

	return mesh;
}

asset::ICPUMesh* CShapeCreator::createPLY(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties)
{
	return nullptr;
}

}
}
}