#include "os.h"

#include "../../ext/MitsubaLoader/CMitsubaLoader.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"


namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


CMitsubaLoader::CMitsubaLoader(asset::IAssetManager* _manager) : asset::IAssetLoader(), manager(_manager)
{
#ifdef _IRR_DEBUG
	setDebugName("CMitsubaLoader");
#endif
}

bool CMitsubaLoader::isALoadableFileFormat(io::IReadFile* _file) const
{
	constexpr uint32_t stackSize = 16u*1024u;
	char tempBuff[stackSize+1];
	tempBuff[stackSize] = 0;

	const char* stringsToFind[] = { "<?xml", "version", "scene"};
	constexpr uint32_t maxStringSize = 8u; // "version\0"
	static_assert(stackSize > 2u*maxStringSize, "WTF?");

	const auto fileSize = _file->getSize();
	while (true)
	{
		auto pos = _file->getPos();
		if (pos >= fileSize)
			break;
		if (pos > maxStringSize)
			_file->seek(_file->getPos()-maxStringSize);
		_file->read(tempBuff,stackSize);
		for (auto i=0u; i<sizeof(stringsToFind)/sizeof(const char*); i++)
		if (strstr(tempBuff, stringsToFind[i]))
		{
			_file->seek(0);
			return true;
		}
	}
	_file->seek(0);
	return false;
}

const char** CMitsubaLoader::getAssociatedFileExtensions() const
{
	static const char* ext[]{ "xml", nullptr };
	return ext;
}


asset::SAssetBundle CMitsubaLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	XML_Parser parser = XML_ParserCreate(nullptr);
	if (!parser)
	{
		os::Printer::log("Could not create XML Parser!", ELL_ERROR);
		return {};
	}

	XML_SetElementHandler(parser, elementHandlerStart, elementHandlerEnd);

	ParserManager parserManager(_override, parser);
	
	//from now data (instance of ParserData struct) will be visible to expat handlers
	XML_SetUserData(parser, &parserManager);

	char* buff = (char*)_IRR_ALIGNED_MALLOC(_file->getSize(),4096u);
	_file->read((void*)buff, _file->getSize());

	XML_Status parseStatus = XML_Parse(parser, buff, _file->getSize(), 0);
	_IRR_ALIGNED_FREE(buff);

	switch (parseStatus)
	{
		case XML_STATUS_ERROR:
			{
				os::Printer::log("Parse status: XML_STATUS_ERROR", ELL_ERROR);
				return {};
			}
			break;
		case XML_STATUS_OK:
			os::Printer::log("Parse status: XML_STATUS_OK", ELL_INFORMATION);
			break;

		case XML_STATUS_SUSPENDED:
			{
				os::Printer::log("Parse status: XML_STATUS_SUSPENDED", ELL_INFORMATION);
				return {};
			}
			break;
	}

	XML_ParserFree(parser);	

	//
	auto* creator = manager->getGeometryCreator();
	auto* manipulator = manager->getMeshManipulator();
	core::vector<core::smart_refctd_ptr<asset::ICPUMesh>> meshes;

	for (auto& shapepair : parserManager.shapegroups)
	{
		const auto* shapedef = shapepair.first;

		core::smart_refctd_ptr<asset::ICPUMesh> mesh;
		auto metadata = core::make_smart_refctd_ptr<IMeshMetadata>(
								core::smart_refctd_ptr(parserManager.m_globalMetadata),
								std::move(shapepair.second));
		bool flipNormals = false;
		auto applyTransformToMB = [](asset::ICPUMeshBuffer* meshbuffer, core::matrix3x4SIMD tform) -> void
		{
			const auto index = meshbuffer->getPositionAttributeIx();
			core::vectorSIMDf vpos;
			for (uint32_t i = 0u; meshbuffer->getAttribute(vpos, index, i); i++)
			{
				tform.transformVect(vpos);
				meshbuffer->setAttribute(vpos, index, i);
			}
		};
		switch (shapedef->type)
		{
			case CElementShape::Type::CUBE:
				mesh = creator->createCubeMesh(core::vector3df(1.f));
				flipNormals = shapedef->cube.flipNormals;
				break;
			case CElementShape::Type::SPHERE:
				mesh = creator->createSphereMesh(shapedef->sphere.radius);
				flipNormals = shapedef->sphere.flipNormals;
				{
					core::matrix3x4SIMD tform;
					tform.setTranslation(shapedef->sphere.center);
					applyTransformToMB(mesh->getMeshBuffer(0),tform);
				}
				break;
			case CElementShape::Type::CYLINDER:
				{
					auto diff = shapedef->cylinder.p0-shapedef->cylinder.p1;
					mesh = creator->createCylinderMesh(shapedef->cylinder.radius, core::length(diff).x, 64);
					core::vectorSIMDf up(0.f);
					float maxDot = diff[0];
					uint32_t index = 0u;
					for (auto i = 1u; i < 3u; i++)
						if (diff[i] < maxDot)
						{
							maxDot = diff[i];
							index = i;
						}
					up[index] = 1.f;
					auto tform = core::matrix3x4SIMD::buildCameraLookAtMatrixRH(shapedef->cylinder.p0,shapedef->cylinder.p1,up);
					applyTransformToMB(mesh->getMeshBuffer(0), tform);
				}
				flipNormals = shapedef->cylinder.flipNormals;
				break;
			case CElementShape::Type::RECTANGLE:
				mesh = creator->createRectangleMesh(core::vector2df_SIMD(1.f,1.f));
				flipNormals = shapedef->rectangle.flipNormals;
				break;
			case CElementShape::Type::DISK:
				mesh = creator->createDiskMesh(1.f,64);
				flipNormals = shapedef->disk.flipNormals;
				break;
			case CElementShape::Type::OBJ:
				break;
			case CElementShape::Type::PLY:
				break;
			case CElementShape::Type::SERIALIZED:
				_IRR_DEBUG_BREAK_IF(true);
				break;
			case CElementShape::Type::SHAPEGROUP:
				_IRR_DEBUG_BREAK_IF(true);
				break;
			case CElementShape::Type::INSTANCE:
				_IRR_DEBUG_BREAK_IF(true);
				break;
			default:
				_IRR_DEBUG_BREAK_IF(true);
				break;
		}
		// flip normals if necessary
		if (flipNormals)
		for (auto i=0u; i<mesh->getMeshBufferCount(); i++)
			manipulator->flipSurfaces(mesh->getMeshBuffer(i));

		metadata->instances.push_back(shapedef->transform.matrix.extractSub3x4());
		if (mesh)
		{
			manager->setAssetMetadata(mesh.get(), std::move(metadata));
			meshes.push_back(std::move(mesh));
		}
	}

	return {meshes};
}

}
}
}