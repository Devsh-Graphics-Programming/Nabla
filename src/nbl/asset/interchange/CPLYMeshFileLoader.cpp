// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/asset/IAssetManager.h"

#ifdef _NBL_COMPILE_WITH_PLY_LOADER_

#include <numeric>

#include "IReadFile.h"
#include "nbl_os.h"

#include "nbl/asset/utils/IMeshManipulator.h"

#include "CPLYMeshFileLoader.h"

namespace nbl
{
namespace asset
{

CPLYMeshFileLoader::CPLYMeshFileLoader(IAssetManager* _am) 
	: IRenderpassIndependentPipelineLoader(_am)
{

}

CPLYMeshFileLoader::~CPLYMeshFileLoader() {}

bool CPLYMeshFileLoader::isALoadableFileFormat(io::IReadFile* _file) const
{
    const char* headers[3]{
        "format ascii 1.0",
        "format binary_little_endian 1.0",
        "format binary_big_endian 1.0"
    };

    const size_t prevPos = _file->getPos();

    char buf[40];
    _file->seek(0u);
    _file->read(buf, sizeof(buf));
    _file->seek(prevPos);

    char* header = buf;
    if (strncmp(header, "ply", 3u) != 0)
        return false;
    
    header += 4;
    char* lf = strstr(header, "\n");
    if (!lf)
        return false;
    *lf = 0;

    for (uint32_t i = 0u; i < 3u; ++i)
        if (strcmp(header, headers[i]) == 0)
            return true;
    return false;
}

void CPLYMeshFileLoader::initialize()
{
	IRenderpassIndependentPipelineLoader::initialize();

	auto precomputeAndCachePipeline = [&](CPLYMeshFileLoader::E_TYPE type, bool indexBufferBindingAvailable)
	{
		constexpr std::array<std::pair<uint8_t, std::pair<std::string_view, std::string_view>>, 3> avaiableOptionsForShaders
		{
			std::make_pair(ET_COL, std::make_pair("nbl/builtin/material/debug/vertex_color/specialized_shader.vert", "nbl/builtin/material/debug/vertex_color/specialized_shader.frag")),
			std::make_pair(ET_UV, std::make_pair("nbl/builtin/material/debug/vertex_uv/specialized_shader.vert", "nbl/builtin/material/debug/vertex_uv/specialized_shader.frag")),
			std::make_pair(ET_NORM, std::make_pair("nbl/builtin/material/debug/vertex_normal/specialized_shader.vert", "nbl/builtin/material/debug/vertex_normal/specialized_shader.frag"))
		};

		auto chooseShaderPaths = [&]() -> std::pair<std::string_view, std::string_view>
		{
			switch (type)
			{
				case ET_POS:
				case ET_COL:
					return std::make_pair("nbl/builtin/material/debug/vertex_color/specialized_shader.vert", "nbl/builtin/material/debug/vertex_color/specialized_shader.frag");
				case ET_UV: 
					return std::make_pair("nbl/builtin/material/debug/vertex_uv/specialized_shader.vert", "nbl/builtin/material/debug/vertex_uv/specialized_shader.frag");
				case ET_NORM:
					return std::make_pair("nbl/builtin/material/debug/vertex_normal/specialized_shader.vert", "nbl/builtin/material/debug/vertex_normal/specialized_shader.frag");
				default:
					return {};
			}
		};

		auto defaultOverride = IAssetLoaderOverride(m_assetMgr);
		const std::string pipelineCacheHash = getPipelineCacheKey(type, indexBufferBindingAvailable);
		const uint32_t _hierarchyLevel = 0;
		const IAssetLoader::SAssetLoadContext fakeContext(IAssetLoader::SAssetLoadParams{}, nullptr);

		const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_RENDERPASS_INDEPENDENT_PIPELINE, (asset::IAsset::E_TYPE)0u };
		auto pipelineBundle = defaultOverride.findCachedAsset(pipelineCacheHash, types, fakeContext, _hierarchyLevel + ICPURenderpassIndependentPipeline::DESC_SET_HIERARCHYLEVELS_BELOW);
		if (pipelineBundle.getContents().empty())
		{
			auto mbVertexShader = core::smart_refctd_ptr<ICPUSpecializedShader>();
			auto mbFragmentShader = core::smart_refctd_ptr<ICPUSpecializedShader>();
			{
				const IAsset::E_TYPE types[]{ IAsset::E_TYPE::ET_SPECIALIZED_SHADER, static_cast<IAsset::E_TYPE>(0u) };
				const auto shaderPaths = chooseShaderPaths();

				auto vertexShaderBundle = m_assetMgr->findAssets(shaderPaths.first.data(), types);
				auto fragmentShaderBundle = m_assetMgr->findAssets(shaderPaths.second.data(), types);

				mbVertexShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(vertexShaderBundle->begin()->getContents().begin()[0]);
				mbFragmentShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(fragmentShaderBundle->begin()->getContents().begin()[0]);
			}

			auto mbPipelineLayout = defaultOverride.findDefaultAsset<ICPUPipelineLayout>("nbl/builtin/pipeline_layout/loader/PLY", fakeContext, 0u).first;

			const std::array<SVertexInputAttribParams, 4> vertexAttribParamsAllOptions =
			{
				SVertexInputAttribParams(0u, EF_R32G32B32_SFLOAT, 0),
				SVertexInputAttribParams(1u, EF_R32G32B32A32_SFLOAT, 0),
				SVertexInputAttribParams(2u, EF_R32G32_SFLOAT, 0),
				SVertexInputAttribParams(3u, EF_R32G32B32_SFLOAT, 0)
			};

			SVertexInputParams inputParams;
			
			std::vector<uint8_t> availableAttributes = { ET_POS };
			if (type != ET_POS)
				availableAttributes.push_back(static_cast<uint8_t>(type));

			for (auto& attrib : availableAttributes)
			{
				const auto currentBitmask = core::createBitmask({ attrib });
				inputParams.enabledBindingFlags |= currentBitmask;
				inputParams.enabledAttribFlags |= currentBitmask;
				inputParams.bindings[attrib] = { asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(vertexAttribParamsAllOptions[attrib].format)), EVIR_PER_VERTEX };
				inputParams.attributes[attrib] = vertexAttribParamsAllOptions[attrib];
			}
		
			SBlendParams blendParams;
			SPrimitiveAssemblyParams primitiveAssemblyParams;
			if (indexBufferBindingAvailable)
				primitiveAssemblyParams.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST;
			else
				primitiveAssemblyParams.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_POINT_LIST;

			SRasterizationParams rastarizationParmas;

			auto mbPipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(mbPipelineLayout), nullptr, nullptr, inputParams, blendParams, primitiveAssemblyParams, rastarizationParmas);
			{
				mbPipeline->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX, mbVertexShader.get());
				mbPipeline->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX, mbFragmentShader.get());
			
				asset::SAssetBundle newPipelineBundle(nullptr, { core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(mbPipeline) });
				defaultOverride.insertAssetIntoCache(newPipelineBundle, pipelineCacheHash, fakeContext, _hierarchyLevel + ICPURenderpassIndependentPipeline::DESC_SET_HIERARCHYLEVELS_BELOW);
			}
		}
		else
			return;
	};

	/*
		Pipeline permutations are cached
	*/

	precomputeAndCachePipeline(ET_POS, false);
	precomputeAndCachePipeline(ET_COL, false);
	precomputeAndCachePipeline(ET_UV, false);
	precomputeAndCachePipeline(ET_NORM, false);

	precomputeAndCachePipeline(ET_POS, true);
	precomputeAndCachePipeline(ET_COL, true);
	precomputeAndCachePipeline(ET_UV, true);
	precomputeAndCachePipeline(ET_NORM, true);
}

//! creates/loads an animated mesh from the file.
asset::SAssetBundle CPLYMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
		return {};

	SContext ctx = {
		asset::IAssetLoader::SAssetLoadContext{
			_params,
			_file
		},
		_hierarchyLevel,
		_override
	};

	// attempt to allocate the buffer and fill with data
	if (!allocateBuffer(ctx))
	{
		return {};
	}

	// start with empty mesh
    core::smart_refctd_ptr<asset::ICPUMesh> mesh;
	uint32_t vertCount=0;

	// Currently only supports ASCII meshes
	if (strcmp(getNextLine(ctx), "ply"))
	{
		os::Printer::log("Not a valid PLY file", ctx.inner.mainFile->getFileName().c_str(), ELL_ERROR);
	}
	else
	{
		// cut the next line out
		getNextLine(ctx);
		// grab the word from this line
		char* word = getNextWord(ctx);

		// ignore comments
		while (strcmp(word, "comment") == 0)
		{
			getNextLine(ctx);
			word = getNextWord(ctx);
		}

		bool readingHeader = true;
		bool continueReading = true;
		ctx.IsBinaryFile = false;
		ctx.IsWrongEndian= false;

		do
		{
			if (strcmp(word, "format") == 0)
			{
				word = getNextWord(ctx);

				if (strcmp(word, "binary_little_endian") == 0)
				{
					ctx.IsBinaryFile = true;
				}
				else if (strcmp(word, "binary_big_endian") == 0)
				{
					ctx.IsBinaryFile = true;
					ctx.IsWrongEndian = true;
				}
				else if (strcmp(word, "ascii"))
				{
					// abort if this isn't an ascii or a binary mesh
					os::Printer::log("Unsupported PLY mesh format", word, ELL_ERROR);
					continueReading = false;
				}

				if (continueReading)
				{
					word = getNextWord(ctx);
					if (strcmp(word, "1.0"))
					{
						os::Printer::log("Unsupported PLY mesh version", word, ELL_WARNING);
					}
				}
			}
			else if (strcmp(word, "property") == 0)
			{
				word = getNextWord(ctx);

				if (!ctx.ElementList.size())
				{
					os::Printer::log("PLY property found before element", word, ELL_WARNING);
				}
				else
				{
					// get element
					SPLYElement* el = ctx.ElementList.back().get();
				
					// fill property struct
					SPLYProperty prop;
					prop.Type = getPropertyType(word);
					el->KnownSize += prop.size();

					if (prop.Type == EPLYPT_LIST)
					{
						el->IsFixedWidth = false;

						word = getNextWord(ctx);

						prop.Data.List.CountType = getPropertyType(word);
						if (ctx.IsBinaryFile && prop.Data.List.CountType == EPLYPT_UNKNOWN)
						{
							os::Printer::log("Cannot read binary PLY file containing data types of unknown length", word, ELL_ERROR);
							continueReading = false;
						}
						else
						{
							word = getNextWord(ctx);
							prop.Data.List.ItemType = getPropertyType(word);
							if (ctx.IsBinaryFile && prop.Data.List.ItemType == EPLYPT_UNKNOWN)
							{
								os::Printer::log("Cannot read binary PLY file containing data types of unknown length", word, ELL_ERROR);
								continueReading = false;
							}
						}
					}
					else if (ctx.IsBinaryFile && prop.Type == EPLYPT_UNKNOWN)
					{
						os::Printer::log("Cannot read binary PLY file containing data types of unknown length", word, ELL_ERROR);
						continueReading = false;
					}

					prop.Name = getNextWord(ctx);

					// add property to element
					el->Properties.push_back(prop);
				}
			}
			else if (strcmp(word, "element") == 0)
			{
                auto el = std::make_unique<SPLYElement>();
				el->Name = getNextWord(ctx);
				el->Count = atoi(getNextWord(ctx));
				el->IsFixedWidth = true;
				el->KnownSize = 0;
				if (el->Name == "vertex")
					vertCount = el->Count;

                ctx.ElementList.emplace_back(std::move(el));

			}
			else if (strcmp(word, "end_header") == 0)
			{
				readingHeader = false;
				if (ctx.IsBinaryFile)
				{
					ctx.StartPointer = ctx.LineEndPointer + 1;
				}
			}
			else if (strcmp(word, "comment") == 0)
			{
				// ignore line
			}
			else
			{
				os::Printer::log("Unknown item in PLY file", word, ELL_WARNING);
			}

			if (readingHeader && continueReading)
			{
				getNextLine(ctx);
				word = getNextWord(ctx);
			}
		}
		while (readingHeader && continueReading);

		// now to read the actual data from the file
		if (continueReading)
		{
			// create a mesh buffer
			auto mb = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>();

			mb->setNormalAttributeIx(3u);
      
			asset::SBufferBinding<asset::ICPUBuffer> attributes[4];
			core::vector<uint32_t> indices;

			bool hasNormals = true;

			// loop through each of the elements
			for (uint32_t i=0; i<ctx.ElementList.size(); ++i)
			{
				// do we want this element type?
				if (ctx.ElementList[i]->Name == "vertex")
				{
					auto& plyVertexElement = *ctx.ElementList[i];

					for (auto& vertexProperty : plyVertexElement.Properties)
					{
						const auto propertyName = vertexProperty.Name;

						if (propertyName == "x" || propertyName == "y" || propertyName == "z")
						{
							if (!attributes[ET_POS].buffer)
							{
								attributes[ET_POS].offset = 0u;
								attributes[ET_POS].buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(asset::getTexelOrBlockBytesize(EF_R32G32B32_SFLOAT) * plyVertexElement.Count);
							}
						}
						else if(propertyName == "nx" || propertyName == "ny" || propertyName == "nz")
						{
							if (!attributes[ET_NORM].buffer)
							{
								attributes[ET_NORM].offset = 0u;
								attributes[ET_NORM].buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(asset::getTexelOrBlockBytesize(EF_R32G32B32_SFLOAT) * plyVertexElement.Count);
							}
						}
						else if (propertyName == "u" || propertyName == "s" || propertyName == "v" || propertyName == "t")
						{
							if (!attributes[ET_UV].buffer)
							{
								attributes[ET_UV].offset = 0u;
								attributes[ET_UV].buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(asset::getTexelOrBlockBytesize(EF_R32G32_SFLOAT) * plyVertexElement.Count);
							}
						}
						else if (propertyName == "red" || propertyName == "green" || propertyName == "blue" || propertyName == "alpha")
						{
							if (!attributes[ET_COL].buffer)
							{
								attributes[ET_COL].offset = 0u;
								attributes[ET_COL].buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(asset::getTexelOrBlockBytesize(EF_R32G32B32A32_SFLOAT) * plyVertexElement.Count);
							}
						}			
					}

					// loop through vertex properties
					for (uint32_t j=0; j<ctx.ElementList[i]->Count; ++j)
						hasNormals &= readVertex(ctx, plyVertexElement, attributes, j, _params);
				}
				else if (ctx.ElementList[i]->Name == "face")
				{
					const size_t indicesCount = ctx.ElementList[i]->Count;

					// read faces
					for (uint32_t j=0; j < indicesCount; ++j)
						readFace(ctx, *ctx.ElementList[i], indices);
				}
				else
				{
					// skip these elements
					for (uint32_t j=0; j < ctx.ElementList[i]->Count; ++j)
						skipElement(ctx, *ctx.ElementList[i]);
				}
			}

			mb->setPositionAttributeIx(0);

            if (indices.size())
            {
				asset::SBufferBinding<ICPUBuffer> indexBinding = { 0, core::make_smart_refctd_ptr<asset::ICPUBuffer>(indices.size() * sizeof(uint32_t)) };
				memcpy(indexBinding.buffer->getPointer(), indices.data(), indexBinding.buffer->getSize());
				
				mb->setIndexCount(indices.size());
				mb->setIndexBufferBinding(std::move(indexBinding));
				mb->setIndexType(asset::EIT_32BIT);

				if (!genVertBuffersForMBuffer(mb.get(), attributes, ctx))
					return {};
            }
            else
            {
				mb->setIndexCount(attributes[ET_POS].buffer->getSize());
				mb->setIndexType(EIT_UNKNOWN);

				if (!genVertBuffersForMBuffer(mb.get(), attributes, ctx))
					return {};
            }

			IMeshManipulator::recalculateBoundingBox(mb.get());

			mesh = core::make_smart_refctd_ptr<ICPUMesh>();
			mesh->getMeshBufferVector().emplace_back(std::move(mb));

			IMeshManipulator::recalculateBoundingBox(mesh.get());
		}
	}
	
	auto* mbPipeline = mesh->getMeshBuffers().begin()[0]->getPipeline();
	auto meta = core::make_smart_refctd_ptr<CPLYMetadata>(1u, std::move(m_basicViewParamsSemantics));
	meta->placeMeta(0u, mbPipeline);

	return SAssetBundle(std::move(meta),{ std::move(mesh) });
}

static void performActionBasedOnOrientationSystem(const asset::IAssetLoader::SAssetLoadParams& _params, std::function<void()> performOnRightHanded, std::function<void()> performOnLeftHanded)
{
	if (_params.loaderFlags & IAssetLoader::ELPF_RIGHT_HANDED_MESHES)
		performOnRightHanded();
	else
		performOnLeftHanded();
}
  
bool CPLYMeshFileLoader::readVertex(SContext& _ctx, const SPLYElement& Element, asset::SBufferBinding<asset::ICPUBuffer> outAttributes[4], const uint32_t& currentVertexIndex, const asset::IAssetLoader::SAssetLoadParams& _params)
{
	if (!_ctx.IsBinaryFile)
		getNextLine(_ctx);

	std::pair<bool, core::vectorSIMDf> attribs[4];
	attribs[ET_COL].second.W = 1.f;
	attribs[ET_NORM].second.Y = 1.f;

	constexpr auto ET_POS_BYTESIZE = asset::getTexelOrBlockBytesize<EF_R32G32B32_SFLOAT>();
	constexpr auto ET_NORM_BYTESIZE = asset::getTexelOrBlockBytesize<EF_R32G32B32_SFLOAT>();
	constexpr auto ET_UV_BYTESIZE = asset::getTexelOrBlockBytesize<EF_R32G32_SFLOAT>();
	constexpr auto ET_COL_BYTESIZE = asset::getTexelOrBlockBytesize<EF_R32G32B32A32_SFLOAT>();

	bool result = false;
	for (uint32_t i = 0; i < Element.Properties.size(); ++i)
	{
		E_PLY_PROPERTY_TYPE t = Element.Properties[i].Type;

		if (Element.Properties[i].Name == "x")
		{
			auto& value = attribs[ET_POS].second.X = getFloat(_ctx, t);
			attribs[ET_POS].first = true;

			if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
				performActionBasedOnOrientationSystem<float>(value, [](float& varToFlip) { varToFlip = -varToFlip; });

			const size_t propertyOffset = ET_POS_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_POS].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[0] = value;
		}
		else if (Element.Properties[i].Name == "y")
		{
			auto& value = attribs[ET_POS].second.Y = getFloat(_ctx, t);
			attribs[ET_POS].first = true;

			const size_t propertyOffset = ET_POS_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_POS].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[1] = value;
		}
		else if (Element.Properties[i].Name == "z")
		{
			auto& value = attribs[ET_POS].second.Z = getFloat(_ctx, t);
			attribs[ET_POS].first = true;

			const size_t propertyOffset = ET_POS_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_POS].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[2] = value;
		}
		else if (Element.Properties[i].Name == "nx")
		{
			auto& value = attribs[ET_NORM].second.X = getFloat(_ctx, t);
			attribs[ET_NORM].first = result = true;

			if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
				performActionBasedOnOrientationSystem<float>(attribs[ET_NORM].second.X, [](float& varToFlip) { varToFlip = -varToFlip; });

			const size_t propertyOffset = ET_NORM_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_NORM].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[0] = value;
		}
		else if (Element.Properties[i].Name == "ny")
		{
			auto& value = attribs[ET_NORM].second.Y = getFloat(_ctx, t);
			attribs[ET_NORM].first = result = true;

			const size_t propertyOffset = ET_NORM_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_NORM].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[1] = value;
		}
		else if (Element.Properties[i].Name == "nz")
		{
			auto& value = attribs[ET_NORM].second.Z = getFloat(_ctx, t);
			attribs[ET_NORM].first = result = true;

			const size_t propertyOffset = ET_NORM_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_NORM].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[2] = value;
		}
		// there isn't a single convention for the UV, some softwares like Blender or Assimp use "st" instead of "uv"
		else if (Element.Properties[i].Name == "u" || Element.Properties[i].Name == "s")
		{
			auto& value = attribs[ET_UV].second.X = getFloat(_ctx, t);
			attribs[ET_UV].first = true;

			const size_t propertyOffset = ET_UV_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_UV].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[0] = value;
		}
		else if (Element.Properties[i].Name == "v" || Element.Properties[i].Name == "t")
		{
			auto& value = attribs[ET_UV].second.Y = getFloat(_ctx, t);
			attribs[ET_UV].first = true;

			const size_t propertyOffset = ET_UV_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_UV].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[1] = value;
		}
		else if (Element.Properties[i].Name == "red")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(getInt(_ctx, t)) / 255.f;
			attribs[ET_COL].second.X = value;
			attribs[ET_COL].first = true;

			const size_t propertyOffset = ET_COL_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_COL].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[0] = value;
		}
		else if (Element.Properties[i].Name == "green")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(getInt(_ctx, t)) / 255.f;
			attribs[ET_COL].second.Y = value;
			attribs[ET_COL].first = true;

			const size_t propertyOffset = ET_COL_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_COL].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[1] = value;
		}
		else if (Element.Properties[i].Name == "blue")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(getInt(_ctx, t)) / 255.f;
			attribs[ET_COL].second.Z = value;
			attribs[ET_COL].first = true;

			const size_t propertyOffset = ET_COL_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_COL].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[2] = value;
		}
		else if (Element.Properties[i].Name == "alpha")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(getInt(_ctx, t)) / 255.f;
			attribs[ET_COL].second.W = value;
			attribs[ET_COL].first = true;

			const size_t propertyOffset = ET_COL_BYTESIZE * currentVertexIndex;
			uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_COL].buffer->getPointer()) + propertyOffset;

			reinterpret_cast<float*>(data)[3] = value;
		}
		else
			skipProperty(_ctx, Element.Properties[i]);
	}

	return result;
}


bool CPLYMeshFileLoader::readFace(SContext& _ctx, const SPLYElement& Element, core::vector<uint32_t>& _outIndices)
{
	if (!_ctx.IsBinaryFile)
		getNextLine(_ctx);

	for (uint32_t i = 0; i < Element.Properties.size(); ++i)
	{
		if ((Element.Properties[i].Name == "vertex_indices" ||
			Element.Properties[i].Name == "vertex_index") && Element.Properties[i].Type == EPLYPT_LIST)
		{
			int32_t count = getInt(_ctx, Element.Properties[i].Data.List.CountType);
			//_NBL_DEBUG_BREAK_IF(count != 3)

			uint32_t a = getInt(_ctx, Element.Properties[i].Data.List.ItemType),
				b = getInt(_ctx, Element.Properties[i].Data.List.ItemType),
				c = getInt(_ctx, Element.Properties[i].Data.List.ItemType);
			int32_t j = 3;

			_outIndices.push_back(a);
			_outIndices.push_back(b);
			_outIndices.push_back(c);

			for (; j < count; ++j)
			{
				b = c;
				c = getInt(_ctx, Element.Properties[i].Data.List.ItemType);
				_outIndices.push_back(a);
				_outIndices.push_back(c);
				_outIndices.push_back(b);
			}
		}
		else if (Element.Properties[i].Name == "intensity")
		{
			// todo: face intensity
			skipProperty(_ctx, Element.Properties[i]);
		}
		else
			skipProperty(_ctx, Element.Properties[i]);
	}
	return true;
}


// skips an element and all properties. return false on EOF
void CPLYMeshFileLoader::skipElement(SContext& _ctx, const SPLYElement& Element)
{
	if (_ctx.IsBinaryFile)
		if (Element.IsFixedWidth)
			moveForward(_ctx, Element.KnownSize);
		else
			for (uint32_t i = 0; i < Element.Properties.size(); ++i)
				skipProperty(_ctx, Element.Properties[i]);
	else
		getNextLine(_ctx);
}


void CPLYMeshFileLoader::skipProperty(SContext& _ctx, const SPLYProperty &Property)
{
	if (Property.Type == EPLYPT_LIST)
	{
		int32_t count = getInt(_ctx, Property.Data.List.CountType);

		for (int32_t i=0; i < count; ++i)
			getInt(_ctx, Property.Data.List.CountType);
	}
	else
	{
		if (_ctx.IsBinaryFile)
			moveForward(_ctx, Property.size());
		else
			getNextWord(_ctx);
	}
}


bool CPLYMeshFileLoader::allocateBuffer(SContext& _ctx)
{
	// Destroy the element list if it exists
	_ctx.ElementList.clear();

	if (!_ctx.Buffer)
        _ctx.Buffer = _NBL_NEW_ARRAY(char, PLY_INPUT_BUFFER_SIZE);

	// not enough memory?
	if (!_ctx.Buffer)
		return false;

	// blank memory
	memset(_ctx.Buffer, 0, PLY_INPUT_BUFFER_SIZE);

	_ctx.StartPointer = _ctx.Buffer;
	_ctx.EndPointer = _ctx.Buffer;
	_ctx.LineEndPointer = _ctx.Buffer - 1;
	_ctx.WordLength = -1;
	_ctx.EndOfFile = false;

	// get data from the file
	fillBuffer(_ctx);

	return true;
}


// gets more data from the file. returns false on EOF
void CPLYMeshFileLoader::fillBuffer(SContext& _ctx)
{
	if (_ctx.EndOfFile)
		return;

	uint32_t length = (uint32_t)(_ctx.EndPointer - _ctx.StartPointer);
	if (length && _ctx.StartPointer != _ctx.Buffer)
	{
		// copy the remaining data to the start of the buffer
		memcpy(_ctx.Buffer, _ctx.StartPointer, length);
	}
	// reset start position
	_ctx.StartPointer = _ctx.Buffer;
	_ctx.EndPointer = _ctx.StartPointer + length;

	if (_ctx.inner.mainFile->getPos() == _ctx.inner.mainFile->getSize())
	{
		_ctx.EndOfFile = true;
	}
	else
	{
		// read data from the file
		uint32_t count = _ctx.inner.mainFile->read(_ctx.EndPointer, PLY_INPUT_BUFFER_SIZE - length);

		// increment the end pointer by the number of bytes read
		_ctx.EndPointer += count;

		// if we didn't completely fill the buffer
		if (count != PLY_INPUT_BUFFER_SIZE - length)
		{
			// blank the rest of the memory
			memset(_ctx.EndPointer, 0, _ctx.Buffer + PLY_INPUT_BUFFER_SIZE - _ctx.EndPointer);

			// end of file
			_ctx.EndOfFile = true;
		}
	}
}


// skips x bytes in the file, getting more data if required
void CPLYMeshFileLoader::moveForward(SContext& _ctx, uint32_t bytes)
{
	if (_ctx.StartPointer + bytes >= _ctx.EndPointer)
		fillBuffer(_ctx);
	if (_ctx.StartPointer + bytes < _ctx.EndPointer)
		_ctx.StartPointer += bytes;
	else
		_ctx.StartPointer = _ctx.EndPointer;
}

bool CPLYMeshFileLoader::genVertBuffersForMBuffer(
	asset::ICPUMeshBuffer* _mbuf,
	const asset::SBufferBinding<asset::ICPUBuffer> attributes[4],
	SContext& context
) const
{
	core::vector<uint8_t> availableAttributes;
	for (auto i = 0; i < 4; ++i)
		if (attributes[i].buffer)
			availableAttributes.push_back(i);

	{
		size_t check = attributes[0].buffer->getSize();
		for (size_t i = 1u; i < 4u; ++i)
		{
			if (attributes[i].buffer && attributes[i].buffer->getSize() != check)
				return false;
			else if (attributes[i].buffer)
				check = attributes[i].buffer->getSize();
		}
	}

	auto getPipeline = [&]() -> core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>
	{
		constexpr std::array<uint8_t, 3> avaiableOptionsForShaders { ET_COL, ET_UV, ET_NORM };

		auto fetchPipelineFromCache = [&](CPLYMeshFileLoader::E_TYPE attribute)
		{
			const IAssetLoader::SAssetLoadContext fakeContext(IAssetLoader::SAssetLoadParams{}, nullptr);
			const std::string hash = getPipelineCacheKey(attribute, _mbuf->getIndexBufferBinding().buffer.get());

			const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_RENDERPASS_INDEPENDENT_PIPELINE, (asset::IAsset::E_TYPE)0u };
			auto pipelineBundle = context.loaderOverride->findCachedAsset(hash, types, fakeContext, context.topHierarchyLevel + ICPURenderpassIndependentPipeline::DESC_SET_HIERARCHYLEVELS_BELOW);
			{
				bool status = !pipelineBundle.getContents().empty();
				assert(status);
			}

			auto mbPipeline = core::smart_refctd_ptr_static_cast<asset::ICPURenderpassIndependentPipeline>(pipelineBundle.getContents().begin()[0]);

			return mbPipeline;
		};

		core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline> mbPipeline;
		{
			for (auto& anOption : avaiableOptionsForShaders)
			{
				auto found = std::find(availableAttributes.begin(), availableAttributes.end(), anOption);
				if (found != availableAttributes.end())
					mbPipeline = fetchPipelineFromCache(static_cast<E_TYPE>(anOption));
			}

			if(!mbPipeline)
				mbPipeline = fetchPipelineFromCache(ET_POS);
		}

		return mbPipeline;
	};

	auto mbPipeline = getPipeline();

	for (auto index = 0; index < 4; ++index)
	{
		auto attribute = attributes[index];
		if (attribute.buffer)
			_mbuf->setVertexBufferBinding(std::move(attribute), index);
	}
	
	_mbuf->setPipeline(std::move(mbPipeline));

    return true;
}

E_PLY_PROPERTY_TYPE CPLYMeshFileLoader::getPropertyType(const char* typeString) const
{
	if (strcmp(typeString, "char") == 0 ||
		strcmp(typeString, "uchar") == 0 ||
		strcmp(typeString, "int8") == 0 ||
		strcmp(typeString, "uint8") == 0)
	{
		return EPLYPT_INT8;
	}
	else if (strcmp(typeString, "uint") == 0 ||
		strcmp(typeString, "int16") == 0 ||
		strcmp(typeString, "uint16") == 0 ||
		strcmp(typeString, "short") == 0 ||
		strcmp(typeString, "ushort") == 0)
	{
		return EPLYPT_INT16;
	}
	else if (strcmp(typeString, "int") == 0 ||
		strcmp(typeString, "long") == 0 ||
		strcmp(typeString, "ulong") == 0 ||
		strcmp(typeString, "int32") == 0 ||
		strcmp(typeString, "uint32") == 0)
	{
		return EPLYPT_INT32;
	}
	else if (strcmp(typeString, "float") == 0 ||
		strcmp(typeString, "float32") == 0)
	{
		return EPLYPT_FLOAT32;
	}
	else if (strcmp(typeString, "float64") == 0 ||
		strcmp(typeString, "double") == 0)
	{
		return EPLYPT_FLOAT64;
	}
	else if (strcmp(typeString, "list") == 0)
	{
		return EPLYPT_LIST;
	}
	else
	{
		// unsupported type.
		// cannot be loaded in binary mode
		return EPLYPT_UNKNOWN;
	}
}


// Split the string data into a line in place by terminating it instead of copying.
char* CPLYMeshFileLoader::getNextLine(SContext& _ctx)
{
	// move the start pointer along
	_ctx.StartPointer = _ctx.LineEndPointer + 1;

	// crlf split across buffer move
	if (*_ctx.StartPointer == '\n')
	{
		*_ctx.StartPointer = '\0';
		++_ctx.StartPointer;
	}

	// begin at the start of the next line
	char* pos = _ctx.StartPointer;
	while (pos < _ctx.EndPointer && *pos && *pos != '\r' && *pos != '\n')
		++pos;

	if (pos < _ctx.EndPointer && (*(pos + 1) == '\r' || *(pos + 1) == '\n'))
	{
		*pos = '\0';
		++pos;
	}

	// we have reached the end of the buffer
	if (pos >= _ctx.EndPointer)
	{
		// get data from the file
		if (!_ctx.EndOfFile)
		{
			fillBuffer(_ctx);
			// reset line end pointer
			_ctx.LineEndPointer = _ctx.StartPointer - 1;

			if (_ctx.StartPointer != _ctx.EndPointer)
				return getNextLine(_ctx);
			else
				return _ctx.Buffer;
		}
		else
		{
			// EOF
			_ctx.StartPointer = _ctx.EndPointer - 1;
			*_ctx.StartPointer = '\0';
			return _ctx.StartPointer;
		}
	}
	else
	{
		// null terminate the string in place
		*pos = '\0';
		_ctx.LineEndPointer = pos;
		_ctx.WordLength = -1;
		// return pointer to the start of the line
		return _ctx.StartPointer;
	}
}


// null terminate the next word on the previous line and move the next word pointer along
// since we already have a full line in the buffer, we never need to retrieve more data
char* CPLYMeshFileLoader::getNextWord(SContext& _ctx)
{
	// move the start pointer along
	_ctx.StartPointer += _ctx.WordLength + 1;
	if (!*_ctx.StartPointer)
		getNextLine(_ctx);

	if (_ctx.StartPointer == _ctx.LineEndPointer)
	{
		_ctx.WordLength = -1; //
		return _ctx.LineEndPointer;
	}
	// begin at the start of the next word
	char* pos = _ctx.StartPointer;
	while (*pos && pos < _ctx.LineEndPointer && pos < _ctx.EndPointer && *pos != ' ' && *pos != '\t')
		++pos;

	while (*pos && pos < _ctx.LineEndPointer && pos < _ctx.EndPointer && (*pos == ' ' || *pos == '\t'))
	{
		// null terminate the string in place
		*pos = '\0';
		++pos;
	}
	--pos;
    _ctx.WordLength = (int32_t)(pos-_ctx.StartPointer);
	// return pointer to the start of the word
	return _ctx.StartPointer;
}


// read the next float from the file and move the start pointer along
float CPLYMeshFileLoader::getFloat(SContext& _ctx, E_PLY_PROPERTY_TYPE t)
{
	float retVal = 0.0f;

	if (_ctx.IsBinaryFile)
	{
		if (_ctx.EndPointer - _ctx.StartPointer < 8)
			fillBuffer(_ctx);

		if (_ctx.EndPointer - _ctx.StartPointer > 0)
		{
			switch (t)
			{
			case EPLYPT_INT8:
				retVal = *_ctx.StartPointer;
				_ctx.StartPointer++;
				break;
			case EPLYPT_INT16:
				if (_ctx.IsWrongEndian)
					retVal = core::Byteswap::byteswap<int16_t>(*(reinterpret_cast<int16_t*>(_ctx.StartPointer)));
				else
					retVal = *(reinterpret_cast<int16_t*>(_ctx.StartPointer));
				_ctx.StartPointer += 2;
				break;
			case EPLYPT_INT32:
				if (_ctx.IsWrongEndian)
					retVal = float(core::Byteswap::byteswap<int32_t>(*(reinterpret_cast<int32_t*>(_ctx.StartPointer))));
				else
					retVal = float(*(reinterpret_cast<int32_t*>(_ctx.StartPointer)));
				_ctx.StartPointer += 4;
				break;
			case EPLYPT_FLOAT32:
				if (_ctx.IsWrongEndian)
					retVal = core::Byteswap::byteswap<float>(*(reinterpret_cast<float*>(_ctx.StartPointer)));
				else
					retVal = *(reinterpret_cast<float*>(_ctx.StartPointer));
				_ctx.StartPointer += 4;
				break;
			case EPLYPT_FLOAT64:
				char tmp[8];
				memcpy(tmp, _ctx.StartPointer, 8);
				if (_ctx.IsWrongEndian)
					for (size_t i = 0u; i < 4u; ++i)
						std::swap(tmp[i], tmp[7u - i]);
				retVal = float(*(reinterpret_cast<double*>(tmp)));
				_ctx.StartPointer += 8;
				break;
			case EPLYPT_LIST:
			case EPLYPT_UNKNOWN:
			default:
				retVal = 0.0f;
				_ctx.StartPointer++; // ouch!
			}
		}
		else
			retVal = 0.0f;
	}
	else
	{
		char* word = getNextWord(_ctx);
		switch (t)
		{
		case EPLYPT_INT8:
		case EPLYPT_INT16:
		case EPLYPT_INT32:
			retVal = float(atoi(word));
			break;
		case EPLYPT_FLOAT32:
		case EPLYPT_FLOAT64:
			retVal = float(atof(word));
			break;
		case EPLYPT_LIST:
		case EPLYPT_UNKNOWN:
		default:
			retVal = 0.0f;
		}
	}

	return retVal;
}


// read the next int from the file and move the start pointer along
uint32_t CPLYMeshFileLoader::getInt(SContext& _ctx, E_PLY_PROPERTY_TYPE t)
{
	uint32_t retVal = 0;

	if (_ctx.IsBinaryFile)
	{
		if (!_ctx.EndOfFile && _ctx.EndPointer - _ctx.StartPointer < 8)
			fillBuffer(_ctx);

		if (_ctx.EndPointer - _ctx.StartPointer)
		{
			switch (t)
			{
			case EPLYPT_INT8:
				retVal = *_ctx.StartPointer;
				_ctx.StartPointer++;
				break;
			case EPLYPT_INT16:
				if (_ctx.IsWrongEndian)
					retVal = core::Byteswap::byteswap<uint16_t>(*(reinterpret_cast<uint16_t*>(_ctx.StartPointer)));
				else
					retVal = *(reinterpret_cast<uint16_t*>(_ctx.StartPointer));
				_ctx.StartPointer += 2;
				break;
			case EPLYPT_INT32:
				if (_ctx.IsWrongEndian)
					retVal = core::Byteswap::byteswap<int32_t>(*(reinterpret_cast<int32_t*>(_ctx.StartPointer)));
				else
					retVal = *(reinterpret_cast<int32_t*>(_ctx.StartPointer));
				_ctx.StartPointer += 4;
				break;
			case EPLYPT_FLOAT32:
				if (_ctx.IsWrongEndian)
					retVal = (uint32_t)core::Byteswap::byteswap<float>(*(reinterpret_cast<float*>(_ctx.StartPointer)));
				else
					retVal = (uint32_t)(*(reinterpret_cast<float*>(_ctx.StartPointer)));
				_ctx.StartPointer += 4;
				break;
			case EPLYPT_FLOAT64:
				// todo: byteswap 64-bit
				retVal = (uint32_t)(*(reinterpret_cast<double*>(_ctx.StartPointer)));
				_ctx.StartPointer += 8;
				break;
			case EPLYPT_LIST:
			case EPLYPT_UNKNOWN:
			default:
				retVal = 0;
				_ctx.StartPointer++; // ouch!
			}
		}
		else
			retVal = 0;
	}
	else
	{
		char* word = getNextWord(_ctx);
		switch (t)
		{
		case EPLYPT_INT8:
		case EPLYPT_INT16:
		case EPLYPT_INT32:
			retVal = atoi(word);
			break;
		case EPLYPT_FLOAT32:
		case EPLYPT_FLOAT64:
			retVal = uint32_t(atof(word));
			break;
		case EPLYPT_LIST:
		case EPLYPT_UNKNOWN:
		default:
			retVal = 0;
		}
	}
	return retVal;
}


} // end namespace scene
} // end namespace nbl

#endif // _NBL_COMPILE_WITH_PLY_LOADER_
