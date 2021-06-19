// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "BuildConfigOptions.h"

#ifdef _NBL_COMPILE_WITH_STL_LOADER_

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/CQuantNormalCache.h"

#include "CSTLMeshFileLoader.h"

#include "IReadFile.h"
#include "nbl_os.h"

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

SAssetBundle CSTLMeshFileLoader::loadAsset(IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (_params.meshManipulatorOverride == nullptr)
	{
		_NBL_DEBUG_BREAK_IF(true);
		assert(false);
	}

	CQuantNormalCache* const quantNormalCache = _params.meshManipulatorOverride->getQuantNormalCache();

	const size_t filesize = _file->getSize();
	if (filesize < 6ull) // we need a header
		return {};

	bool hasColor = false;

	auto mesh = core::make_smart_refctd_ptr<ICPUMesh>();
	auto meshbuffer = core::make_smart_refctd_ptr<ICPUMeshBuffer>();
	meshbuffer->setPositionAttributeIx(POSITION_ATTRIBUTE);
	meshbuffer->setNormalAttributeIx(NORMAL_ATTRIBUTE);

	bool binary = false;
	core::stringc token;
	if (getNextToken(_file, token) != "solid")
		binary = hasColor = true;

	core::vector<core::vectorSIMDf> positions, normals;
	core::vector<uint32_t> colors;
	if (binary)
	{
		if (_file->getSize() < 80)
			return {};

		_file->seek(80); // skip header
		uint32_t vtxCnt = 0u;
		_file->read(&vtxCnt, 4);
		positions.reserve(3 * vtxCnt);
		normals.reserve(vtxCnt);
		colors.reserve(vtxCnt);
	}
	else
		goNextLine(_file); // skip header


	uint16_t attrib = 0u;
	token.reserve(32);
	while (_file->getPos() < filesize)
	{
		if (!binary)
		{
			if (getNextToken(_file, token) != "facet")
			{
				if (token == "endsolid")
					break;
				return {};
			}
			if (getNextToken(_file, token) != "normal")
			{
				return {};
			}
		}

		{
			core::vectorSIMDf n;
			getNextVector(_file, n, binary);
			if(_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
				performActionBasedOnOrientationSystem<float>(n.x, [](float& varToFlip) {varToFlip = -varToFlip;});
			normals.push_back(core::normalize(n));
		}

		if (!binary)
		{
			if (getNextToken(_file, token) != "outer" || getNextToken(_file, token) != "loop")
				return {};
		}

		{
			core::vectorSIMDf p[3];
			for (uint32_t i = 0u; i < 3u; ++i)
			{
				if (!binary)
				{
					if (getNextToken(_file, token) != "vertex")
						return {};
				}
				getNextVector(_file, p[i], binary);
				if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
					performActionBasedOnOrientationSystem<float>(p[i].x, [](float& varToFlip){varToFlip = -varToFlip; });
			}
			for (uint32_t i = 0u; i < 3u; ++i) // seems like in STL format vertices are ordered in clockwise manner...
				positions.push_back(p[2u - i]);
		}

		if (!binary)
		{
			if (getNextToken(_file, token) != "endloop" || getNextToken(_file, token) != "endfacet")
				return {};
		}
		else
		{
			_file->read(&attrib, 2);
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


bool CSTLMeshFileLoader::isALoadableFileFormat(io::IReadFile* _file) const
{
	if (!_file || _file->getSize() <= 6u)
		return false;

	char header[6];
	const size_t prevPos = _file->getPos();
	_file->seek(0u);
	_file->read(header, 6u);
	_file->seek(prevPos);

	if (strncmp(header, "solid ", 6u) == 0)
		return true;
	else
	{
		if (_file->getSize() < 84u)
		{
			_file->seek(prevPos);
			return false;
		}
		_file->seek(80u);
		uint32_t triCnt;
		_file->read(&triCnt, 4u);
		_file->seek(prevPos);
		const size_t STL_TRI_SZ = 50u;
		return _file->getSize() == (STL_TRI_SZ * triCnt + 84u);
	}
}

//! Read 3d vector of floats
void CSTLMeshFileLoader::getNextVector(io::IReadFile* file, core::vectorSIMDf& vec, bool binary) const
{
	if (binary)
	{
		file->read(&vec.X, 4);
		file->read(&vec.Y, 4);
		file->read(&vec.Z, 4);
	}
	else
	{
		goNextWord(file);
		core::stringc tmp;

		getNextToken(file, tmp);
		sscanf(tmp.c_str(), "%f", &vec.X);
		getNextToken(file, tmp);
		sscanf(tmp.c_str(), "%f", &vec.Y);
		getNextToken(file, tmp);
		sscanf(tmp.c_str(), "%f", &vec.Z);
	}
	vec.X = -vec.X;
}


//! Read next word
const core::stringc& CSTLMeshFileLoader::getNextToken(io::IReadFile* file, core::stringc& token) const
{
	goNextWord(file);
	uint8_t c;
	token = "";
	while (file->getPos() != file->getSize())
	{
		file->read(&c, 1);
		// found it, so leave
		if (core::isspace(c))
			break;
		token.append(c);
	}
	return token;
}

//! skip to next word
void CSTLMeshFileLoader::goNextWord(io::IReadFile* file) const
{
	uint8_t c;
	while (file->getPos() != file->getSize())
	{
		file->read(&c, 1);
		// found it, so leave
		if (!core::isspace(c))
		{
			file->seek(-1, true);
			break;
		}
	}
}


//! Read until line break is reached and stop at the next non-space character
void CSTLMeshFileLoader::goNextLine(io::IReadFile* file) const
{
	uint8_t c;
	// look for newline characters
	while (file->getPos() != file->getSize())
	{
		file->read(&c, 1);
		// found it, so leave
		if (c == '\n' || c == '\r')
			break;
	}
}


#endif // _NBL_COMPILE_WITH_STL_LOADER_
