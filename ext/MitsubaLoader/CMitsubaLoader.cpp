#include "os.h"

#include <cwchar>

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
	static const wchar_t* stringsToFindW[] = { L"<?xml", L"version", L"scene"};
	constexpr uint32_t maxStringSize = 8u; // "version\0"
	static_assert(stackSize > 2u*maxStringSize, "WTF?");

	const size_t prevPos = _file->getPos();
	const auto fileSize = _file->getSize();
	if (fileSize < maxStringSize)
		return false;

	_file->seek(0);
	_file->read(tempBuff, 3u);
	bool utf16 = false;
	if (tempBuff[0]==0xEFu && tempBuff[1]==0xBBu && tempBuff[2]==0xBFu)
		utf16 = false;
	else if (reinterpret_cast<uint16_t*>(tempBuff)[0]==0xFEFFu)
	{
		utf16 = true;
		_file->seek(2);
	}
	else
		_file->seek(0);
	while (true)
	{
		auto pos = _file->getPos();
		if (pos >= fileSize)
			break;
		if (pos > maxStringSize)
			_file->seek(_file->getPos()-maxStringSize);
		_file->read(tempBuff,stackSize);
		for (auto i=0u; i<sizeof(stringsToFind)/sizeof(const char*); i++)
		if (utf16 ? (wcsstr(reinterpret_cast<wchar_t*>(tempBuff),stringsToFindW[i])!=nullptr):(strstr(tempBuff, stringsToFind[i])!=nullptr))
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
	const std::string relativeDir = (io::IFileSystem::getFileDir(_file->getFileName())+"/").c_str();
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
			auto retval = interm_getAssetInHierarchy(manager, relativeDir+filename.svalue, {}, _hierarchyLevel/*+ICPUSCene::MESH_HIERARCHY_LEVELS_BELOW*/, _override);
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
				// do nothing, only instance can make it appear
				break;
			case CElementShape::Type::INSTANCE:
				mesh = instantiateShapeGroup(shapedef->instance.shapegroup,shapedef->transform.matrix);
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

		// meshbuffer processing
		if (shapedef->type!=CElementShape::Type::INSTANCE)
		for (auto i=0u; i<mesh->getMeshBufferCount(); i++)
		{
			auto* meshbuffer = mesh->getMeshBuffer(i);
			// add some metadata
			///auto meshbuffermeta = core::make_smart_refctd_ptr<IMeshBufferMetadata>(shapedef->type,shapedef->emitter ? shapedef->emitter.area:CElementEmitter::Area());
			///manager->setAssetMetadata(meshbuffer,std::move(meshbuffermeta));
			// TODO: change this with shader pipeline
			meshbuffer->getMaterial() = getBSDF(relativeDir,shapedef->bsdf,_hierarchyLevel+asset::ICPUMesh::MESHBUFFER_HIERARCHYLEVELS_BELOW,_override);
		}

		auto metadata = core::make_smart_refctd_ptr<IMeshMetadata>(
								core::smart_refctd_ptr(parserManager.m_globalMetadata),
								std::move(shapepair.second));
		metadata->instances.push_back(shapedef->transform.matrix.extractSub3x4());
		manager->setAssetMetadata(mesh.get(), std::move(metadata));

		meshes.push_back(std::move(mesh));
	}

	return {meshes};
}


CMitsubaLoader::group_ass_type CMitsubaLoader::instantiateShapeGroup(CElementShape::ShapeGroup* shapegroup, const core::matrix4SIMD& tform)
{
	if (!shapegroup)
		return nullptr;

	// will only return the group mesh once so it is only added once
	auto found = groupCache.find(shapegroup);
	if (found != groupCache.end())
	{
		static_cast<IMeshMetadata*>(found->second->getMetadata())->instances.push_back(tform.extractSub3x4());
		return nullptr;
	}

	auto mesh = core::make_smart_refctd_ptr<asset::CCPUMesh>();
	for (auto i=0u; shapegroup->children[i] && i<shapegroup->childCount; i++)
	{
		auto lowermesh = getMesh(shapegroup->children[i]);
		if (lowermesh)
		for (auto j=0u; j<lowermesh->getMeshBufferCount(); j++)
			mesh->addMeshBuffer(core::smart_refctd_ptr<asset::ICPUMeshBuffer>(lowermesh->getMeshBuffer(j)));
	}
	mesh->recalculateBoundingBox();
	return mesh;
}

CMitsubaLoader::shape_ass_type CMitsubaLoader::getMesh(CElementShape* shape)
{
	if (!shape)
		return nullptr;

	auto found = shapeCache.find(shape);
	if (found != shapeCache.end())
		return found->second;

	return nullptr;
}


//! TODO: change to CPU graphics pipeline
CMitsubaLoader::bsdf_ass_type CMitsubaLoader::getBSDF(const std::string& relativeDir, CElementBSDF* bsdf, uint32_t _hierarchyLevel, asset::IAssetLoader::IAssetLoaderOverride* _override)
{
	if (!bsdf)
		return video::SCPUMaterial(); 

	auto found = pipelineCache.find(bsdf);
	if (found != pipelineCache.end())
		return found->second;

	// shader construction would take place here in the new pipeline
	video::SCPUMaterial pipeline;
	NastyTemporaryBitfield nasty = { 0u };
	auto getColor = [](const SPropertyElementData& data) -> core::vectorSIMDf
	{
		switch (data.type)
		{
			case SPropertyElementData::Type::FLOAT:
				return core::vectorSIMDf(data.fvalue);
			case SPropertyElementData::Type::RGB:
				_IRR_FALLTHROUGH;
			case SPropertyElementData::Type::SRGB:
				return data.vvalue;
				break;
			default:
				assert(false);
				break;
		}
		return core::vectorSIMDf();
	};
	auto setTextureOrColorFrom = [&](const CElementTexture::SpectrumOrTexture& spctex) -> void
	{
		if (spctex.value.type!=SPropertyElementData::INVALID)
		{
			_mm_storeu_ps((float*)&pipeline.AmbientColor, getColor(spctex.value).getAsRegister());
		}
		else
		{
			constexpr uint32_t IMAGEVIEW_HIERARCHYLEVEL_BELOW = 1u; // below ICPUMesh, will move it there eventually with shader pipeline and become 2
			pipeline.TextureLayer[0] = getTexture(relativeDir,spctex.texture,_hierarchyLevel+IMAGEVIEW_HIERARCHYLEVEL_BELOW,_override);
			nasty._bitfield |= MITS_USE_TEXTURE;
		}
	};
	// @criss you know that I'm doing absolutely nothing worth keeping around (not caring about BSDF actually)
	switch (bsdf->type)
	{
		case CElementBSDF::Type::DIFFUSE:
		case CElementBSDF::Type::ROUGHDIFFUSE:
			setTextureOrColorFrom(bsdf->diffuse.reflectance);
			break;
		case CElementBSDF::Type::DIELECTRIC:
		case CElementBSDF::Type::THINDIELECTRIC: // basically glass with no refraction
		case CElementBSDF::Type::ROUGHDIELECTRIC:
			{
				core::vectorSIMDf color(bsdf->dielectric.extIOR/bsdf->dielectric.intIOR);
				_mm_storeu_ps((float*)& pipeline.AmbientColor, color.getAsRegister());
			}
			break;
		case CElementBSDF::Type::CONDUCTOR:
		case CElementBSDF::Type::ROUGHCONDUCTOR:
			{
				auto color = core::vectorSIMDf(1.f)-getColor(bsdf->conductor.k);
				_mm_storeu_ps((float*)& pipeline.AmbientColor, color.getAsRegister());
			}
			break;
		case CElementBSDF::Type::PLASTIC:
		case CElementBSDF::Type::ROUGHPLASTIC:
			setTextureOrColorFrom(bsdf->plastic.diffuseReflectance);
			break;
		case CElementBSDF::Type::TWO_SIDED:
			{
				pipeline = getBSDF(relativeDir,bsdf->twosided.bsdf[0],_hierarchyLevel,_override);
				nasty._bitfield |= MITS_TWO_SIDED|reinterpret_cast<uint32_t&>(pipeline.MaterialTypeParam);				
			}
			break;
		case CElementBSDF::Type::MASK:
			{
				pipeline = getBSDF(relativeDir,bsdf->mask.bsdf[0],_hierarchyLevel,_override);
				//bsdf->mask.opacity // ran out of space in SMaterial (can be texture or constant)
				nasty._bitfield |= /*MITS_MASK|*/reinterpret_cast<uint32_t&>(pipeline.MaterialTypeParam);				
			}
			break;
		default:
			_IRR_DEBUG_BREAK_IF(true); // TODO: more BSDF untangling!
			break;
	}
	reinterpret_cast<uint32_t&>(pipeline.MaterialTypeParam) = nasty._bitfield;
	pipeline.BackfaceCulling = false;

	pipelineCache.insert({bsdf,pipeline});
	return pipeline;
}

CMitsubaLoader::tex_ass_type CMitsubaLoader::getTexture(const std::string& relativeDir, CElementTexture* tex, uint32_t _hierarchyLevel, asset::IAssetLoader::IAssetLoaderOverride* _override)
{
	if (!tex)
		return {};

	auto found = textureCache.find(tex);
	if (found != textureCache.end())
		return found->second;

	video::SMaterialLayer<asset::ICPUTexture> layer;
	switch (tex->type)
	{
		case CElementTexture::Type::BITMAP:
			{
				auto retval = interm_getAssetInHierarchy(manager,relativeDir+tex->bitmap.filename.svalue,{},_hierarchyLevel,_override);
				auto contentRange = retval.getContents();
				if (contentRange.first < contentRange.second)
				{
					auto asset = contentRange.first[0];
					if (asset && asset->getAssetType() == asset::IAsset::ET_IMAGE)
					{/*
						auto 
						{
							const void* src_container[4] = { data, nullptr, nullptr, nullptr };
							video::convertColor<EF_R8_UNORM, EF_R8_SRGB>(src_container, data, dim.X, dim); 
						}*/
						video::convertColor<asset::EF_R8G8B8A8_SRGB, asset::EF_B8G8R8A8_SRGB>(nullptr, nullptr, 0, core::vector3d<uint32_t>(0,0,0));
						auto texture = core::smart_refctd_ptr_static_cast<asset::ICPUTexture>(asset);
						auto getSingleChannelFormat = [](asset::E_FORMAT format, uint32_t index) -> asset::E_FORMAT
						{
							auto ratio = asset::getBytesPerPixel(format);
							ratio = decltype(ratio)(ratio.getNumerator(),ratio.getDenominator()*asset::getFormatChannelCount(format));
							switch (ratio.getRoundedUpInteger())
							{
								case 1:
									if (asset::isSRGBFormat(format))
										return index!=3 ? format:asset::EF_R8_UNORM;
									else
									{
										bool _signed = asset::isSignedFormat(format);
										if (asset::isIntegerFormat(format))
											return _signed ? asset::EF_R8_SINT : asset::EF_R8_UINT;
										else
											return _signed ? asset::EF_R8_SNORM:asset::EF_R8_UNORM;
									}
									break;
								case 2:
									if (asset::isFloatingPointFormat(format))
										return asset::EF_R16_SFLOAT;
									else
									{
										bool _signed = asset::isSignedFormat(format);
										if (asset::isIntegerFormat(format))
											return _signed ? asset::EF_R16_SINT : asset::EF_R16_UINT;
										else
											return _signed ? asset::EF_R16_SNORM:asset::EF_R16_UNORM;
									}
								case 3:
									_IRR_FALLTHROUGH;
								case 4:
									if (asset::isFloatingPointFormat(format))
										return asset::EF_R32_SFLOAT;
									else if (asset::isSignedFormat(format))
										return asset::EF_R32_SINT;
									else
										return asset::EF_R32_UINT;
									break;
								default:
									break;
							}
							return format;
						};
						auto extractChannel = [&](uint32_t index) -> core::smart_refctd_ptr<asset::ICPUTexture>
						{
							core::vector<asset::CImageData*> subimages; // this will leak like crazy
							for (uint32_t level=0u; level<=texture->getHighestMip(); level)
							{
								auto rng = texture->getMipMap(level);
								for (auto it=rng.first; it!=rng.second; it++)
								{
									auto* olddata = *it;
									auto format = getSingleChannelFormat(olddata->getColorFormat(), index);
									auto* data = new asset::CImageData(nullptr,olddata->getSliceMin(),olddata->getSliceMax(),olddata->getSupposedMipLevel(),format);
									_IRR_DEBUG_BREAK_IF(true);
									subimages.push_back(data);
								}
							}
							return core::smart_refctd_ptr<asset::ICPUTexture>(asset::ICPUTexture::create(std::move(subimages), texture->getSourceFilename(), texture->getType()), core::dont_grab);
						};
						switch (tex->bitmap.channel)
						{
							// no GL_R8_SRGB support yet
							case CElementTexture::Bitmap::CHANNEL::R:
								layer.Texture = extractChannel(0);
								break;
							case CElementTexture::Bitmap::CHANNEL::G:
								layer.Texture = extractChannel(1);
								break;
							case CElementTexture::Bitmap::CHANNEL::B:
								layer.Texture = extractChannel(2);
								break;
							case CElementTexture::Bitmap::CHANNEL::A:
								layer.Texture = extractChannel(3);
								break;/* special conversions needed to CIE space
							case CElementTexture::Bitmap::CHANNEL::X:
							case CElementTexture::Bitmap::CHANNEL::Y:
							case CElementTexture::Bitmap::CHANNEL::Z:*/
							default:
								layer.Texture = std::move(texture);
								break;
						}
						//! TODO: this stuff (custom shader sampling code?)
						_IRR_DEBUG_BREAK_IF(tex->bitmap.uoffset != 0.f);
						_IRR_DEBUG_BREAK_IF(tex->bitmap.voffset != 0.f);
						_IRR_DEBUG_BREAK_IF(tex->bitmap.uscale != 1.f);
						_IRR_DEBUG_BREAK_IF(tex->bitmap.vscale != 1.f);
					}
				}
				// adjust gamma on pixels (painful and long process)
				if (!std::isnan(tex->bitmap.gamma))
				{
					_IRR_DEBUG_BREAK_IF(true); // TODO
				}
				switch (tex->bitmap.filterType)
				{
					case CElementTexture::Bitmap::FILTER_TYPE::EWA:
						_IRR_FALLTHROUGH; // we dont support this fancy stuff
					case CElementTexture::Bitmap::FILTER_TYPE::TRILINEAR:
						layer.SamplingParams.MinFilter = video::ETFT_LINEAR_LINEARMIP;
						layer.SamplingParams.MaxFilter = video::ETFT_LINEAR_NO_MIP;
						break;
					default:
						layer.SamplingParams.MinFilter = video::ETFT_NEAREST_NEARESTMIP;
						layer.SamplingParams.MaxFilter = video::ETFT_NEAREST_NO_MIP;
						break;
				}
				layer.SamplingParams.AnisotropicFilter = core::max(core::findMSB(uint32_t(tex->bitmap.maxAnisotropy)),1);
				auto getWrapMode = [](CElementTexture::Bitmap::WRAP_MODE mode) -> video::E_TEXTURE_CLAMP
				{
					switch (mode)
					{
						case CElementTexture::Bitmap::WRAP_MODE::CLAMP:
							return video::ETC_CLAMP_TO_EDGE;
							break;
						case CElementTexture::Bitmap::WRAP_MODE::MIRROR:
							return video::ETC_MIRROR;
							break;
						case CElementTexture::Bitmap::WRAP_MODE::ONE:
							_IRR_DEBUG_BREAK_IF(true); // TODO : replace whole texture?
							break;
						case CElementTexture::Bitmap::WRAP_MODE::ZERO:
							_IRR_DEBUG_BREAK_IF(true); // TODO : replace whole texture?
							break;
						default:
							break;
					}
					return video::ETC_REPEAT;
				};
				layer.SamplingParams.TextureWrapU = getWrapMode(tex->bitmap.wrapModeU);
				layer.SamplingParams.TextureWrapV = getWrapMode(tex->bitmap.wrapModeV);
			}
			break;
		case CElementTexture::Type::SCALE:
			_IRR_DEBUG_BREAK_IF(true); // TODO
			break;
		default:
			_IRR_DEBUG_BREAK_IF(true);
			break;
	}
	textureCache.insert({tex,layer});
	return std::move(layer);
}


}
}
}