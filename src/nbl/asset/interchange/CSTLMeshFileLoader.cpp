// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/asset/IAssetManager.h"

#ifdef _NBL_COMPILE_WITH_STL_LOADER_

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/CQuantNormalCache.h"

#include "CSTLMeshFileLoader.h"

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::io;

constexpr auto POSITION_ATTRIBUTE = 0;
constexpr auto COLOR_ATTRIBUTE = 1;
constexpr auto UV_ATTRIBUTE = 2;
constexpr auto NORMAL_ATTRIBUTE = 3;

CSTLMeshFileLoader::CSTLMeshFileLoader(asset::IAssetManager* _m_assetMgr)
	: IRenderpassIndependentPipelineLoader(_m_assetMgr), m_assetMgr(_m_assetMgr)
{
	
}

void CSTLMeshFileLoader::initialize()
{
	IRenderpassIndependentPipelineLoader::initialize();

	auto precomputeAndCachePipeline = [&](bool withColorAttribute)
	{
		auto getShaderDefaultPaths = [&]() -> std::pair<std::string_view, std::string_view>
		{
			if (withColorAttribute)
				return std::make_pair("nbl/builtin/material/debug/vertex_color/specialized_shader.vert", "nbl/builtin/material/debug/vertex_color/specialized_shader.frag");
			else
				return std::make_pair("nbl/builtin/material/debug/vertex_normal/specialized_shader.vert", "nbl/builtin/material/debug/vertex_normal/specialized_shader.frag");
		 };

		auto defaultOverride = IAssetLoaderOverride(m_assetMgr);
		const std::string pipelineCacheHash = getPipelineCacheKey(withColorAttribute).data();
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
				const auto shaderPaths = getShaderDefaultPaths();

				auto vertexShaderBundle = m_assetMgr->findAssets(shaderPaths.first.data(), types);
				auto fragmentShaderBundle = m_assetMgr->findAssets(shaderPaths.second.data(), types);

				mbVertexShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(vertexShaderBundle->begin()->getContents().begin()[0]);
				mbFragmentShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(fragmentShaderBundle->begin()->getContents().begin()[0]);
			}

			auto defaultOverride = IAssetLoaderOverride(m_assetMgr);

			const IAssetLoader::SAssetLoadContext fakeContext(IAssetLoader::SAssetLoadParams{}, nullptr);
			auto mbBundlePipelineLayout = defaultOverride.findDefaultAsset<ICPUPipelineLayout>("nbl/builtin/pipeline_layout/loader/STL", fakeContext, _hierarchyLevel + ICPURenderpassIndependentPipeline::PIPELINE_LAYOUT_HIERARCHYLEVELS_BELOW);
			auto mbPipelineLayout = mbBundlePipelineLayout.first;

			auto const positionFormatByteSize = getTexelOrBlockBytesize(EF_R32G32B32_SFLOAT);
			auto const colorFormatByteSize = withColorAttribute ? getTexelOrBlockBytesize(EF_B8G8R8A8_UNORM) : 0;
			auto const normalFormatByteSize = getTexelOrBlockBytesize(EF_A2B10G10R10_SNORM_PACK32);

			SVertexInputParams mbInputParams;
			const auto stride = positionFormatByteSize + colorFormatByteSize + normalFormatByteSize;
			mbInputParams.enabledBindingFlags |= core::createBitmask({ 0 });
			mbInputParams.enabledAttribFlags |= core::createBitmask({ POSITION_ATTRIBUTE, NORMAL_ATTRIBUTE, withColorAttribute ? COLOR_ATTRIBUTE : 0 });
			mbInputParams.bindings[0] = { stride, EVIR_PER_VERTEX };

			mbInputParams.attributes[POSITION_ATTRIBUTE].format = EF_R32G32B32_SFLOAT;
			mbInputParams.attributes[POSITION_ATTRIBUTE].relativeOffset = 0;
			mbInputParams.attributes[POSITION_ATTRIBUTE].binding = 0;

			if (withColorAttribute)
			{
				mbInputParams.attributes[COLOR_ATTRIBUTE].format = EF_R32G32B32_SFLOAT;
				mbInputParams.attributes[COLOR_ATTRIBUTE].relativeOffset = positionFormatByteSize;
				mbInputParams.attributes[COLOR_ATTRIBUTE].binding = 0;
			}

			mbInputParams.attributes[NORMAL_ATTRIBUTE].format = EF_R32G32B32_SFLOAT;
			mbInputParams.attributes[NORMAL_ATTRIBUTE].relativeOffset = positionFormatByteSize + colorFormatByteSize;
			mbInputParams.attributes[NORMAL_ATTRIBUTE].binding = 0;

			SBlendParams blendParams;
			SPrimitiveAssemblyParams primitiveAssemblyParams;
			primitiveAssemblyParams.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST;

			SRasterizationParams rastarizationParmas;

			auto mbPipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(mbPipelineLayout), nullptr, nullptr, mbInputParams, blendParams, primitiveAssemblyParams, rastarizationParmas);
			{
				mbPipeline->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX, mbVertexShader.get());
				mbPipeline->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX, mbFragmentShader.get());
			}

			asset::SAssetBundle newPipelineBundle(nullptr, {core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(mbPipeline)});
			defaultOverride.insertAssetIntoCache(newPipelineBundle, pipelineCacheHash, fakeContext, _hierarchyLevel + ICPURenderpassIndependentPipeline::DESC_SET_HIERARCHYLEVELS_BELOW);
		}
		else
			return;
	};

	/*
		Pipeline permutations are cached
	*/

	precomputeAndCachePipeline(true);
	precomputeAndCachePipeline(false);
}

SAssetBundle CSTLMeshFileLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
		return {};

	SContext context = {
		asset::IAssetLoader::SAssetLoadContext{
			_params,
			_file
		},
		_hierarchyLevel,
		_override
	};

	if (_params.meshManipulatorOverride == nullptr)
	{
		_NBL_DEBUG_BREAK_IF(true);
		assert(false);
	}

	CQuantNormalCache* const quantNormalCache = _params.meshManipulatorOverride->getQuantNormalCache();

	const size_t filesize = context.inner.mainFile->getSize();
	if (filesize < 6ull) // we need a header
		return {};

	bool hasColor = false;

	auto mesh = core::make_smart_refctd_ptr<ICPUMesh>();
	auto meshbuffer = core::make_smart_refctd_ptr<ICPUMeshBuffer>();
	meshbuffer->setPositionAttributeIx(POSITION_ATTRIBUTE);
	meshbuffer->setNormalAttributeIx(NORMAL_ATTRIBUTE);

	bool binary = false;
	std::string token;
	if (getNextToken(&context, token) != "solid")
		binary = hasColor = true;

	core::vector<core::vectorSIMDf> positions, normals;
	core::vector<uint32_t> colors;
	if (binary)
	{
		if (_file->getSize() < 80)
			return {};

		constexpr size_t headerOffset = 80; //! skip header
		context.fileOffset += headerOffset;

		uint32_t vertexCount = 0u;

		system::future<size_t> future;
		context.inner.mainFile->read(future, &vertexCount, context.fileOffset, sizeof(vertexCount));
		const auto bytesRead = future.get();
		context.fileOffset += bytesRead;

		positions.reserve(3 * vertexCount);
		normals.reserve(vertexCount);
		colors.reserve(vertexCount);
	}
	else
		goNextLine(&context); // skip header

	uint16_t attrib = 0u;
	token.reserve(32);
	while (context.fileOffset < filesize) // TODO: check it
	{
		if (!binary)
		{
			if (getNextToken(&context, token) != "facet")
			{
				if (token == "endsolid")
					break;
				return {};
			}
			if (getNextToken(&context, token) != "normal")
			{
				return {};
			}
		}

		{
			core::vectorSIMDf n;
			getNextVector(&context, n, binary);
			if(_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
				performActionBasedOnOrientationSystem<float>(n.x, [](float& varToFlip) {varToFlip = -varToFlip;});
			normals.push_back(core::normalize(n));
		}

		if (!binary)
		{
			if (getNextToken(&context, token) != "outer" || getNextToken(&context, token) != "loop")
				return {};
		}

		{
			core::vectorSIMDf p[3];
			for (uint32_t i = 0u; i < 3u; ++i)
			{
				if (!binary)
				{
					if (getNextToken(&context, token) != "vertex")
						return {};
				}
				getNextVector(&context, p[i], binary);
				if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
					performActionBasedOnOrientationSystem<float>(p[i].x, [](float& varToFlip){varToFlip = -varToFlip; });
			}
			for (uint32_t i = 0u; i < 3u; ++i) // seems like in STL format vertices are ordered in clockwise manner...
				positions.push_back(p[2u - i]);
		}

		if (!binary)
		{
			if (getNextToken(&context, token) != "endloop" || getNextToken(&context, token) != "endfacet")
				return {};
		}
		else
		{
			system::future<size_t> future;
			context.inner.mainFile->read(future, &attrib, context.fileOffset, sizeof(attrib));
			const auto bytesRead = future.get();
			context.fileOffset += bytesRead;
		}

		if (hasColor && (attrib & 0x8000u)) // assuming VisCam/SolidView non-standard trick to store color in 2 bytes of extra attribute
		{
			const void* srcColor[1]{ &attrib };
			uint32_t color{};
			convertColor<EF_A1R5G5B5_UNORM_PACK16, EF_B8G8R8A8_UNORM>(srcColor, &color, 0u, 0u);
			colors.push_back(color);
		}
		else
		{
			hasColor = false;
			colors.clear();
		}

		if ((normals.back() == core::vectorSIMDf()).all())
		{
			normals.back().set(
				core::plane3dSIMDf(
					*(positions.rbegin() + 2),
					*(positions.rbegin() + 1),
					*(positions.rbegin() + 0)).getNormal()
			);
		}
	} // end while (_file->getPos() < filesize)

	const size_t vtxSize = hasColor ? (3 * sizeof(float) + 4 + 4) : (3 * sizeof(float) + 4);
	auto vertexBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vtxSize * positions.size());

	using quant_normal_t = CQuantNormalCache::value_type_t<EF_A2B10G10R10_SNORM_PACK32>;

	quant_normal_t normal;
	for (size_t i = 0u; i < positions.size(); ++i)
	{
		if (i % 3 == 0)
			normal = quantNormalCache->quantize<EF_A2B10G10R10_SNORM_PACK32>(normals[i / 3]);
		uint8_t* ptr = ((uint8_t*)(vertexBuf->getPointer())) + i * vtxSize;
		memcpy(ptr, positions[i].pointer, 3 * 4);

		*reinterpret_cast<quant_normal_t*>(ptr + 12) = normal;

		if (hasColor)
			memcpy(ptr + 16, colors.data() + i / 3, 4);
	}

	const IAssetLoader::SAssetLoadContext fakeContext(IAssetLoader::SAssetLoadParams{}, nullptr);
	const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_RENDERPASS_INDEPENDENT_PIPELINE, (asset::IAsset::E_TYPE)0u };
	auto pipelineBundle = _override->findCachedAsset(getPipelineCacheKey(hasColor).data(), types, fakeContext, _hierarchyLevel + ICPURenderpassIndependentPipeline::DESC_SET_HIERARCHYLEVELS_BELOW);
	{
		bool status = !pipelineBundle.getContents().empty();
		assert(status);
	}

	auto mbPipeline = core::smart_refctd_ptr_static_cast<asset::ICPURenderpassIndependentPipeline>(pipelineBundle.getContents().begin()[0]);

	auto meta = core::make_smart_refctd_ptr<CSTLMetadata>(1u, std::move(m_basicViewParamsSemantics));
	meta->placeMeta(0u, mbPipeline.get());

	meshbuffer->setPipeline(std::move(mbPipeline));
	meshbuffer->setIndexCount(positions.size());
	meshbuffer->setIndexType(asset::EIT_UNKNOWN);

	meshbuffer->setVertexBufferBinding({ 0ul, vertexBuf }, 0);
	mesh->getMeshBufferVector().emplace_back(std::move(meshbuffer));
	
	return SAssetBundle(std::move(meta), { std::move(mesh) });
}

bool CSTLMeshFileLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
{
	if (!_file || _file->getSize() <= 6u)
		return false;

	char header[6];
	
	system::future<size_t> future;
	_file->read(future, header, 0, sizeof(header));
	future.get();

	if (strncmp(header, "solid ", 6u) == 0)
		return true;
	else
	{
		if (_file->getSize() < 84u)
			return false;

		uint32_t triangleCount;

		constexpr size_t readOffset = 80;
		_file->read(future, &triangleCount, readOffset, sizeof(triangleCount));
		future.get();

		constexpr size_t STL_TRI_SZ = 50u;
		return _file->getSize() == (STL_TRI_SZ * triangleCount + 84u);
	}
}

//! Read 3d vector of floats
void CSTLMeshFileLoader::getNextVector(SContext* context, core::vectorSIMDf& vec, bool binary) const
{
	system::future<size_t> future;

	if (binary)
	{
		context->inner.mainFile->read(future, &vec.X, context->fileOffset, sizeof(vec.X));
		{
			const auto bytesRead = future.get();
			context->fileOffset += bytesRead;
		}

		context->inner.mainFile->read(future, &vec.Y, context->fileOffset, sizeof(vec.Y));
		{
			const auto bytesRead = future.get();
			context->fileOffset += bytesRead;
		}

		context->inner.mainFile->read(future, &vec.Z, context->fileOffset, sizeof(vec.Z));
		{
			const auto bytesRead = future.get();
			context->fileOffset += bytesRead;
		}
	}
	else
	{
		goNextWord(context);
		std::string tmp;

		getNextToken(context, tmp);
		sscanf(tmp.c_str(), "%f", &vec.X);
		getNextToken(context, tmp);
		sscanf(tmp.c_str(), "%f", &vec.Y);
		getNextToken(context, tmp);
		sscanf(tmp.c_str(), "%f", &vec.Z);
	}
	vec.X = -vec.X;
}

//! Read next word
const std::string& CSTLMeshFileLoader::getNextToken(SContext* context, std::string& token) const
{
	goNextWord(context);
	char c;

	system::future<size_t> future;

	while (context->fileOffset < context->inner.mainFile->getSize())
	{
		context->inner.mainFile->read(future, &c, context->fileOffset, sizeof(c));
		const auto bytesRead = future.get();
		context->fileOffset += bytesRead;

		// found it, so leave
		if (core::isspace(c))
			break;
		token.append(&c);
	}
	return token;
}

//! skip to next word
void CSTLMeshFileLoader::goNextWord(SContext* context) const
{
	uint8_t c;
	while (context->fileOffset < context->inner.mainFile->getSize()) // TODO: check it
	{
		system::future<size_t> future;
		context->inner.mainFile->read(future, &c, context->fileOffset, sizeof(c));
		const auto bytesRead = future.get();

		if (core::isspace(c))
			context->fileOffset += bytesRead;
		else
			break; // found it, so leave
	}
}

//! Read until line break is reached and stop at the next non-space character
void CSTLMeshFileLoader::goNextLine(SContext* context) const
{
	uint8_t c;
	// look for newline characters
	while (context->fileOffset < context->inner.mainFile->getSize()) // TODO: check it
	{
		system::future<size_t> future;
		context->inner.mainFile->read(future, &c, context->fileOffset, sizeof(c));
		future.get();

		// found it, so leave
		if (c == '\n' || c == '\r')
			break;
	}
}


#endif // _NBL_COMPILE_WITH_STL_LOADER_
