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

	static const char* stringsToFind[] = { "<?xml", "version", "scene"};
	constexpr uint32_t maxStringSize = 8u; // "version\0"
	static_assert(stackSize > 2u*maxStringSize, "WTF?");

	const size_t prevPos = _file->getPos();
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
			_file->seek(prevPos);
			return true;
		}
	}
	_file->seek(prevPos);
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
		// TODO: use references and aliases
		const auto* shapedef = shapepair.first;

		core::smart_refctd_ptr<asset::ICPUMesh> mesh;

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
		auto loadModel = [&](const ext::MitsubaLoader::SPropertyElementData& filename, uint32_t index=0) -> core::smart_refctd_ptr<asset::ICPUMesh>
		{
			assert(filename.type==ext::MitsubaLoader::SPropertyElementData::Type::STRING);
			// TODO: hierarchy stuff
			auto path = io::IFileSystem::getFileDir(_file->getFileName());
			path += "/";
			path += filename.svalue;
			auto retval = manager->getAsset(path.c_str(), _params, _override);
			auto contentRange = retval.getContents();
			if (contentRange.first + index < contentRange.second)
			{
				auto asset = contentRange.first[index];
				if (asset && asset->getAssetType()==asset::IAsset::ET_MESH)
					return core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(asset);
				else
					return nullptr;
			}
			else
				return nullptr;
		};

		bool flipNormals = false;
		bool faceNormals = false;
		float maxSmoothAngle = NAN;
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
				mesh = loadModel(shapedef->obj.filename);
				flipNormals = !shapedef->obj.flipNormals;
				faceNormals = shapedef->obj.faceNormals;
				maxSmoothAngle = shapedef->obj.maxSmoothAngle;
				{
					core::matrix3x4SIMD tform;
					tform.rows[0].x = -1.f; // restore handedness
					for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
						applyTransformToMB(mesh->getMeshBuffer(i), tform);
				}
				if (mesh && shapedef->obj.flipTexCoords)
				{
					for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
					{
						auto meshbuffer = mesh->getMeshBuffer(i);
						core::vectorSIMDf uv;
						for (uint32_t i=0u; meshbuffer->getAttribute(uv, asset::EVAI_ATTR2, i); i++)
						{
							uv.y = -uv.y;
							meshbuffer->setAttribute(uv, asset::EVAI_ATTR2, i);
						}
					}
				}
				// collapse parameter gets ignored
				break;
			case CElementShape::Type::PLY:
				_IRR_DEBUG_BREAK_IF(true); // this code has never been tested
				mesh = loadModel(shapedef->ply.filename);
				flipNormals = !shapedef->ply.flipNormals;
				faceNormals = shapedef->ply.faceNormals;
				maxSmoothAngle = shapedef->ply.maxSmoothAngle;
				if (mesh && shapedef->ply.srgb)
				{
					uint32_t totalVertexCount = 0u;
					for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
						totalVertexCount += mesh->getMeshBuffer(i)->calcVertexCount();
					if (totalVertexCount)
					{
						constexpr uint32_t hidefRGBSize = 4u;
						auto newRGB = core::make_smart_refctd_ptr<asset::ICPUBuffer>(hidefRGBSize*totalVertexCount);
						uint32_t* it = reinterpret_cast<uint32_t*>(newRGB->getPointer());
						for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
						{
							auto meshbuffer = mesh->getMeshBuffer(i);
							uint32_t offset = reinterpret_cast<uint8_t*>(it)-reinterpret_cast<uint8_t*>(newRGB->getPointer());
							core::vectorSIMDf rgb;
							for (uint32_t i=0u; meshbuffer->getAttribute(rgb, asset::EVAI_ATTR1, i); i++,it++)
							{
								for (auto i=0; i<3u; i++)
									rgb[i] = video::impl::srgb2lin(rgb[i]);
								meshbuffer->setAttribute(rgb,it,asset::EF_A2B10G10R10_UNORM_PACK32);
							}
							meshbuffer->getMeshDataAndFormat()->setVertexAttrBuffer(
									core::smart_refctd_ptr(newRGB),asset::EVAI_ATTR1,asset::EF_A2B10G10R10_UNORM_PACK32,hidefRGBSize,offset);
						}
					}
				}
				break;
			case CElementShape::Type::SERIALIZED:
				mesh = loadModel(shapedef->serialized.filename,shapedef->serialized.shapeIndex);
				faceNormals = shapedef->serialized.faceNormals;
				maxSmoothAngle = shapedef->serialized.maxSmoothAngle;
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
		//
		if (!mesh)
			continue;

		// flip normals if necessary
		if (flipNormals)
		for (auto i=0u; i<mesh->getMeshBufferCount(); i++)
			manipulator->flipSurfaces(mesh->getMeshBuffer(i));
		// flip normals if necessary
		if (faceNormals || !std::isnan(maxSmoothAngle))
		{
			auto newMesh = core::make_smart_refctd_ptr<asset::CCPUMesh>();
			float smoothAngleCos = cos(core::radians(maxSmoothAngle));
			for (auto i=0u; i<mesh->getMeshBufferCount(); i++)
			{
				auto newMeshBuffer = manipulator->createMeshBufferUniquePrimitives(mesh->getMeshBuffer(i));

				manipulator->calculateSmoothNormals(newMeshBuffer.get(), false, 0.f, asset::EVAI_ATTR3,
					[&](const asset::IMeshManipulator::SSNGVertexData& a, const asset::IMeshManipulator::SSNGVertexData& b, asset::ICPUMeshBuffer* buffer)
					{
						if (faceNormals)
							return a.indexOffset == b.indexOffset;
						else
							return core::dot(a.parentTriangleFaceNormal,b.parentTriangleFaceNormal).x >= smoothAngleCos;
					});

				asset::IMeshManipulator::SErrorMetric metrics[16];
				newMeshBuffer = manipulator->createOptimizedMeshBuffer(newMeshBuffer.get(),metrics);

				newMesh->addMeshBuffer(std::move(newMeshBuffer));
			}
			mesh = std::move(newMesh);
		}

		// pipeline processing
		for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
			mesh->getMeshBuffer(i)->getMaterial() = processBSDF(shapedef->bsdf); // TODO: change this with shader pipeline

		auto metadata = core::make_smart_refctd_ptr<IMeshMetadata>(
								core::smart_refctd_ptr(parserManager.m_globalMetadata),
								std::move(shapepair.second));
		metadata->instances.push_back(shapedef->transform.matrix.extractSub3x4());
		manager->setAssetMetadata(mesh.get(), std::move(metadata));

		meshes.push_back(std::move(mesh));
	}

	return {meshes};
}

//! TODO: change to CPU graphics pipeline
video::SCPUMaterial CMitsubaLoader::processBSDF(CElementBSDF* bsdf)
{
	if (!bsdf)
		return video::SCPUMaterial(); 

	auto found = pipelineCache.find(bsdf);
	if (found != pipelineCache.end())
		return found->second;

	// shader construction would take place here in the new pipeline
	video::SCPUMaterial pipeline;
	switch (bsdf->type)
	{
		case CElementBSDF::Type::DIFFUSE:
			break;
		case CElementBSDF::Type::TWO_SIDED:
			{
				pipeline = processBSDF(bsdf->twosided.bsdf[0]);
			}
			break;
		default:
			//_IRR_DEBUG_BREAK_IF(true);
			break;
	}
	pipeline.BackfaceCulling = false;
	//pipeline.
	return pipeline;
}

}
}
}