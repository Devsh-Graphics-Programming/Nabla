// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "BuildConfigOptions.h"
#ifdef _NBL_COMPILE_WITH_PLY_LOADER_

#include <numeric>

#include "CPLYMeshFileLoader.h"
#include "nbl/asset/IMeshManipulator.h"
#include "nbl/video/CGPUMesh.h"

#include "IReadFile.h"
#include "os.h"

namespace nbl
{
namespace asset
{

template<typename AssetType, IAsset::E_TYPE assetType>
static core::smart_refctd_ptr<AssetType> getDefaultAsset(const char* _key, IAssetManager* _assetMgr)
{
	size_t storageSz = 1ull;
	asset::SAssetBundle bundle;
	const IAsset::E_TYPE types[]{ assetType, static_cast<IAsset::E_TYPE>(0u) };

	_assetMgr->findAssets(storageSz, &bundle, _key, types);
	if (bundle.isEmpty())
		return nullptr;
	auto assets = bundle.getContents();

	return core::smart_refctd_ptr_static_cast<AssetType>(assets.begin()[0]);
}

CPLYMeshFileLoader::CPLYMeshFileLoader(IAssetManager* _am) : m_assetMgr{ _am } {}

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

//! creates/loads an animated mesh from the file.
asset::SAssetBundle CPLYMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
		return {};

    SContext ctx;
	ctx.File = _file;

	// attempt to allocate the buffer and fill with data
	if (!allocateBuffer(ctx))
	{
		return {};
	}

	// start with empty mesh
    core::smart_refctd_ptr<asset::CCPUMesh> mesh;
	uint32_t vertCount=0;

	// Currently only supports ASCII meshes
	if (strcmp(getNextLine(ctx), "ply"))
	{
		os::Printer::log("Not a valid PLY file", ctx.File->getFileName().c_str(), ELL_ERROR);
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

			mb->setNormalnAttributeIx(3u);
      
			core::vector<core::vectorSIMDf> attribs[4];
			core::vector<uint32_t> indices;

			bool hasNormals = true;

			// loop through each of the elements
			for (uint32_t i=0; i<ctx.ElementList.size(); ++i)
			{
				// do we want this element type?
				if (ctx.ElementList[i]->Name == "vertex")
				{
					// loop through vertex properties
					for (uint32_t j=0; j<ctx.ElementList[i]->Count; ++j)
						hasNormals &= readVertex(ctx, *ctx.ElementList[i], attribs, _params);
				}
				else if (ctx.ElementList[i]->Name == "face")
				{
					// read faces
					for (uint32_t j=0; j < ctx.ElementList[i]->Count; ++j)
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
				asset::SBufferBinding<ICPUBuffer> indexBinding = {0, core::make_smart_refctd_ptr<asset::ICPUBuffer>(indices.size() * sizeof(uint32_t))};
                memcpy(indexBinding.buffer->getPointer(), indices.data(), indexBinding.buffer->getSize());
				auto DATA = reinterpret_cast<uint32_t*>(indexBinding.buffer->getPointer());
				mb->setIndexCount(indices.size());
				mb->setIndexBufferBinding(std::move(indexBinding));
                mb->setIndexType(asset::EIT_32BIT);

				if (!genVertBuffersForMBuffer(mb.get(), attribs))
					return {};
            }
            else
            {
				mb->setIndexCount(attribs[E_POS].size());
				mb->setIndexType(EIT_UNKNOWN);

				if (!genVertBuffersForMBuffer(mb.get(), attribs))
					return {};
            }

			mb->recalculateBoundingBox();

			mesh = core::make_smart_refctd_ptr<CCPUMesh>();
			mesh->addMeshBuffer(std::move(mb));
			mesh->recalculateBoundingBox(true);
		}
	}

	return SAssetBundle({std::move(mesh)});
}

static void performActionBasedOnOrientationSystem(const asset::IAssetLoader::SAssetLoadParams& _params, std::function<void()> performOnRightHanded, std::function<void()> performOnLeftHanded)
{
	if (_params.loaderFlags & IAssetLoader::ELPF_RIGHT_HANDED_MESHES)
		performOnRightHanded();
	else
		performOnLeftHanded();
}
  
bool CPLYMeshFileLoader::readVertex(SContext& _ctx, const SPLYElement& Element, core::vector<core::vectorSIMDf> _outAttribs[4], const asset::IAssetLoader::SAssetLoadParams& _params)
{
	if (!_ctx.IsBinaryFile)
		getNextLine(_ctx);

	std::pair<bool, core::vectorSIMDf> attribs[4];
	attribs[E_COL].second.W = 1.f;
	attribs[E_NORM].second.Y = 1.f;

	bool result = false;
	for (uint32_t i = 0; i < Element.Properties.size(); ++i)
	{
		E_PLY_PROPERTY_TYPE t = Element.Properties[i].Type;

		if (Element.Properties[i].Name == "x")
		{
			attribs[E_POS].second.X = getFloat(_ctx, t);
			attribs[E_POS].first = true;
		}
		else if (Element.Properties[i].Name == "y")
		{
			attribs[E_POS].second.Y = getFloat(_ctx, t);
			attribs[E_POS].first = true;
		}
		else if (Element.Properties[i].Name == "z")
		{
			attribs[E_POS].second.Z = getFloat(_ctx, t);
			attribs[E_POS].first = true;
		}
		else if (Element.Properties[i].Name == "nx")
		{
			attribs[E_NORM].second.X = getFloat(_ctx, t);
			attribs[E_NORM].first = result = true;
		}
		else if (Element.Properties[i].Name == "ny")
		{
			attribs[E_NORM].second.Y = getFloat(_ctx, t);
			attribs[E_NORM].first = result = true;
		}
		else if (Element.Properties[i].Name == "nz")
		{
			attribs[E_NORM].second.Z = getFloat(_ctx, t);
			attribs[E_NORM].first = result = true;
		}
		// there isn't a single convention for the UV, some softwares like Blender or Assimp use "st" instead of "uv"
		else if (Element.Properties[i].Name == "u" || Element.Properties[i].Name == "s")
		{
			attribs[E_UV].second.X = getFloat(_ctx, t);
			attribs[E_UV].first = true;
		}
		else if (Element.Properties[i].Name == "v" || Element.Properties[i].Name == "t")
		{
			attribs[E_UV].second.Y = getFloat(_ctx, t);
			attribs[E_UV].first = true;
		}
		else if (Element.Properties[i].Name == "red")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(getInt(_ctx, t)) / 255.f;
			attribs[E_COL].second.X = value;
			attribs[E_COL].first = true;
		}
		else if (Element.Properties[i].Name == "green")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(getInt(_ctx, t)) / 255.f;
			attribs[E_COL].second.Y = value;
			attribs[E_COL].first = true;
		}
		else if (Element.Properties[i].Name == "blue")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(getInt(_ctx, t)) / 255.f;
			attribs[E_COL].second.Z = value;
			attribs[E_COL].first = true;
		}
		else if (Element.Properties[i].Name == "alpha")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(getInt(_ctx, t)) / 255.f;
			attribs[E_COL].second.W = value;
			attribs[E_COL].first = true;
		}
		else
			skipProperty(_ctx, Element.Properties[i]);
	}

	for (size_t i = 0u; i < 4u; ++i)
		if (attribs[i].first)
		{
			if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
			{
				if (i == E_POS)
					performActionBasedOnOrientationSystem<float>(attribs[E_POS].second.X, [](float& varToFlip) { varToFlip = -varToFlip; });
				else if (i == E_NORM)
					performActionBasedOnOrientationSystem<float>(attribs[E_NORM].second.X, [](float& varToFlip) { varToFlip = -varToFlip; });
			}

			_outAttribs[i].push_back(attribs[i].second);
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

	if (_ctx.File->getPos() == _ctx.File->getSize())
	{
		_ctx.EndOfFile = true;
	}
	else
	{
		// read data from the file
		uint32_t count = _ctx.File->read(_ctx.EndPointer, PLY_INPUT_BUFFER_SIZE - length);

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

bool CPLYMeshFileLoader::genVertBuffersForMBuffer(asset::ICPUMeshBuffer* _mbuf, const core::vector<core::vectorSIMDf> _attribs[4]) const
{
	core::vector<uint8_t> availableAttributes;
	for (auto i = 0; i < 4; ++i)
		if (!_attribs[i].empty())
			availableAttributes.push_back(i);

	{
		size_t check = _attribs[0].size();
		for (size_t i = 1u; i < 4u; ++i)
		{
			if (_attribs[i].size() != 0u && _attribs[i].size() != check)
				return false;
			else if (_attribs[i].size() != 0u)
				check = _attribs[i].size();
		}
	}
	
	auto putAttr = [&_attribs](asset::ICPUMeshBuffer* _buf, size_t _attr)
	{
		size_t i = 0u;
		for (const core::vectorSIMDf& v : _attribs[_attr])
			_buf->setAttribute(v, _attr, i++);
	};

	auto chooseShaderPath = [&]() -> std::string
	{
		constexpr std::array<std::pair<uint8_t, std::string_view>, 3> avaiableOptionsForShaders
		{ 
			std::make_pair(E_COL, "nbl/builtin/materials/debug/vertex_color_debug_shader/specializedshader"),
			std::make_pair(E_UV, "nbl/builtin/materials/debug/uv_debug_shader/specializedshader"),
			std::make_pair(E_NORM, "nbl/builtin/materials/debug/normal_debug_shader/specializedshader")
		};

		for (auto& it : avaiableOptionsForShaders)
		{
			auto found = std::find(availableAttributes.begin(), availableAttributes.end(), it.first);
			if (found != availableAttributes.end())
				return it.second.data(); 
		}

		return avaiableOptionsForShaders[0].second.data(); // if only positions are present, shaders with debug vertex colors are assumed
	};

	auto mbVertexShader = core::smart_refctd_ptr<ICPUSpecializedShader>();
	auto mbFragmentShader = core::smart_refctd_ptr<ICPUSpecializedShader>();
	{
		const IAsset::E_TYPE types[]{ IAsset::E_TYPE::ET_SPECIALIZED_SHADER, IAsset::E_TYPE::ET_SPECIALIZED_SHADER, static_cast<IAsset::E_TYPE>(0u) };
		auto bundle = m_assetMgr->findAssets(chooseShaderPath(), types);

		auto refCountedBundle =
		{
			core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(bundle->begin()->getContents().begin()[0]),
			core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>((bundle->begin()+1)->getContents().begin()[0])
		};

		for (auto& shader : refCountedBundle)
		{
			if (shader->getStage() == ISpecializedShader::ESS_VERTEX)
				mbVertexShader = std::move(shader);
			else if(shader->getStage() == ISpecializedShader::ESS_FRAGMENT)
				mbFragmentShader = std::move(shader);
		}
	}
	auto mbPipelineLayout = getDefaultAsset<ICPUPipelineLayout, IAsset::ET_PIPELINE_LAYOUT>("nbl/builtin/loaders/PLY/pipelinelayout", m_assetMgr);

	constexpr size_t DS1_METADATA_ENTRY_CNT = 3ull;
	core::smart_refctd_dynamic_array<IPipelineMetadata::ShaderInputSemantic> shaderInputsMetadata = core::make_refctd_dynamic_array<decltype(shaderInputsMetadata)>(DS1_METADATA_ENTRY_CNT);
	{
		ICPUDescriptorSetLayout* ds1layout = mbPipelineLayout->getDescriptorSetLayout(1u);

		constexpr IPipelineMetadata::E_COMMON_SHADER_INPUT types[DS1_METADATA_ENTRY_CNT]{ IPipelineMetadata::ECSI_WORLD_VIEW_PROJ, IPipelineMetadata::ECSI_WORLD_VIEW, IPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE };
		constexpr uint32_t sizes[DS1_METADATA_ENTRY_CNT]{ sizeof(SBasicViewParameters::MVP), sizeof(SBasicViewParameters::MV), sizeof(SBasicViewParameters::NormalMat) };
		constexpr uint32_t relOffsets[DS1_METADATA_ENTRY_CNT]{ offsetof(SBasicViewParameters,MVP), offsetof(SBasicViewParameters,MV), offsetof(SBasicViewParameters,NormalMat) };
		for (uint32_t i = 0u; i < DS1_METADATA_ENTRY_CNT; ++i)
		{
			auto& semantic = (shaderInputsMetadata->end() - i - 1u)[0];
			semantic.type = types[i];
			semantic.descriptorSection.type = IPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER;
			semantic.descriptorSection.uniformBufferObject.binding = ds1layout->getBindings().begin()[0].binding;
			semantic.descriptorSection.uniformBufferObject.set = 1u;
			semantic.descriptorSection.uniformBufferObject.relByteoffset = relOffsets[i];
			semantic.descriptorSection.uniformBufferObject.bytesize = sizes[i];
			semantic.descriptorSection.shaderAccessFlags = ICPUSpecializedShader::ESS_VERTEX;
		}
	}
	
	const std::array<SVertexInputAttribParams, 4> vertexAttribParamsAllOptions =
	{
		SVertexInputAttribParams(0u, EF_R32G32B32_SFLOAT, 0),
		SVertexInputAttribParams(1u, EF_R32G32B32A32_SFLOAT, 0),
		SVertexInputAttribParams(2u, EF_R32G32_SFLOAT, 0),
		SVertexInputAttribParams(3u, EF_R32G32B32_SFLOAT, 0)
	};

	SVertexInputParams inputParams;
	for (auto& attrib : availableAttributes)
	{
		const auto currentBitmask = core::createBitmask({ attrib });
		inputParams.enabledBindingFlags |= currentBitmask;
		inputParams.enabledAttribFlags |= currentBitmask;
		inputParams.bindings[attrib] = { 16, EVIR_PER_VERTEX };
		inputParams.attributes[attrib] = vertexAttribParamsAllOptions[attrib];
	}

	SBlendParams blendParams;
	SPrimitiveAssemblyParams primitiveAssemblyParams;
	if (_mbuf->getIndexBufferBinding()->buffer)
		primitiveAssemblyParams.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST;
	else
		primitiveAssemblyParams.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_POINT_LIST;

	SRasterizationParams rastarizationParmas;

	auto mbPipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(mbPipelineLayout), nullptr, nullptr, inputParams, blendParams, primitiveAssemblyParams, rastarizationParmas);
	{
		mbPipeline->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX, mbVertexShader.get());
		mbPipeline->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX, mbFragmentShader.get());

		auto inputParams = mbPipeline->getVertexInputParams();

		for (auto index = 0; index < 4; ++index)
		{
			auto attribute = _attribs[index];
			if (!attribute.empty())
			{
				const auto bufferByteSize = attribute.size() * 16ull;
				auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(bufferByteSize);
				memcpy(buffer->getPointer(), attribute.data(), bufferByteSize); // TODO refactor input to take SBufferBinding to avoid memcpy

				_mbuf->setVertexBufferBinding({ 0ul, std::move(buffer) }, index);
			}
		}
	}

	m_assetMgr->setAssetMetadata(mbPipeline.get(), core::make_smart_refctd_ptr<CPLYPipelineMetadata>(1, std::move(shaderInputsMetadata)));
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
					retVal = os::Byteswap::byteswap(*(reinterpret_cast<int16_t*>(_ctx.StartPointer)));
				else
					retVal = *(reinterpret_cast<int16_t*>(_ctx.StartPointer));
				_ctx.StartPointer += 2;
				break;
			case EPLYPT_INT32:
				if (_ctx.IsWrongEndian)
					retVal = float(os::Byteswap::byteswap(*(reinterpret_cast<int32_t*>(_ctx.StartPointer))));
				else
					retVal = float(*(reinterpret_cast<int32_t*>(_ctx.StartPointer)));
				_ctx.StartPointer += 4;
				break;
			case EPLYPT_FLOAT32:
				if (_ctx.IsWrongEndian)
					retVal = os::Byteswap::byteswap(*(reinterpret_cast<float*>(_ctx.StartPointer)));
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
					retVal = os::Byteswap::byteswap(*(reinterpret_cast<uint16_t*>(_ctx.StartPointer)));
				else
					retVal = *(reinterpret_cast<uint16_t*>(_ctx.StartPointer));
				_ctx.StartPointer += 2;
				break;
			case EPLYPT_INT32:
				if (_ctx.IsWrongEndian)
					retVal = os::Byteswap::byteswap(*(reinterpret_cast<int32_t*>(_ctx.StartPointer)));
				else
					retVal = *(reinterpret_cast<int32_t*>(_ctx.StartPointer));
				_ctx.StartPointer += 4;
				break;
			case EPLYPT_FLOAT32:
				if (_ctx.IsWrongEndian)
					retVal = (uint32_t)os::Byteswap::byteswap(*(reinterpret_cast<float*>(_ctx.StartPointer)));
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
