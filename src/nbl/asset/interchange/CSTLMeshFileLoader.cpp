// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CSTLMeshFileLoader.h"

#ifdef _NBL_COMPILE_WITH_STL_LOADER_

#include "nbl/asset/asset.h"

#include "nbl/asset/IAssetManager.h"

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"
#include <cmath>

using namespace nbl;
using namespace nbl::asset;

CSTLMeshFileLoader::CSTLMeshFileLoader(asset::IAssetManager* _m_assetMgr)
	: m_assetMgr(_m_assetMgr)
{
}

void CSTLMeshFileLoader::initialize()
{
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


	const size_t filesize = context.inner.mainFile->getSize();
	if (filesize < 6ull) // we need a header
		return {};

	bool binary = false;
	std::string token;
	if (getNextToken(&context, token) != "solid")
		binary = true;
	const bool rightHanded = (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES) != 0;

	core::vector<core::vectorSIMDf> positions, normals;
	if (binary)
	{
		if (_file->getSize() < 80)
			return {};

		constexpr size_t headerOffset = 80; 
		context.fileOffset = headerOffset; //! skip header

		uint32_t vertexCount = 0u;

		system::IFile::success_t success;
		context.inner.mainFile->read(success, &vertexCount, context.fileOffset, sizeof(vertexCount));
		if (!success)
			return {};
		context.fileOffset += sizeof(vertexCount);

		positions.reserve(3 * vertexCount);
		normals.reserve(vertexCount);
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
			if (rightHanded)
				n.x = -n.x;
			const float len2 = core::dot(n, n).X;
			if (len2 > 0.f && std::abs(len2 - 1.f) < 1e-4f)
				normals.push_back(n);
			else
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
				if (rightHanded)
					p[i].x = -p[i].x;
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
			system::IFile::success_t success;
			context.inner.mainFile->read(success, &attrib, context.fileOffset, sizeof(attrib));
			if (!success)
				return {};
			context.fileOffset += sizeof(attrib);
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

	if (positions.empty())
		return {};

	core::vector<float> posData(positions.size() * 3u);
	core::vector<float> normalData(positions.size() * 3u);
	for (size_t i = 0u; i < positions.size(); ++i)
	{
		const auto& pos = positions[i];
		const auto& nrm = normals[i / 3u];
		const size_t base = i * 3u;
		posData[base + 0u] = pos.pointer[0];
		posData[base + 1u] = pos.pointer[1];
		posData[base + 2u] = pos.pointer[2];
		normalData[base + 0u] = nrm.pointer[0];
		normalData[base + 1u] = nrm.pointer[1];
		normalData[base + 2u] = nrm.pointer[2];
	}

	auto geometry = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	geometry->setIndexing(IPolygonGeometryBase::TriangleList());
	auto posView = createView(EF_R32G32B32_SFLOAT, positions.size(), posData.data());
	auto normalView = createView(EF_R32G32B32_SFLOAT, positions.size(), normalData.data());
	geometry->setPositionView(std::move(posView));
	geometry->setNormalView(std::move(normalView));
	CPolygonGeometryManipulator::recomputeContentHashes(geometry.get());
	CPolygonGeometryManipulator::recomputeRanges(geometry.get());
	CPolygonGeometryManipulator::recomputeAABB(geometry.get());

	auto meta = core::make_smart_refctd_ptr<CSTLMetadata>();
	return SAssetBundle(std::move(meta), { std::move(geometry) });
}

bool CSTLMeshFileLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
{
	if (!_file || _file->getSize() <= 6u)
		return false;

	char header[6];
	{
		system::IFile::success_t success;
		_file->read(success, header, 0, sizeof(header));
		if (!success)
			return false;
	}

	if (strncmp(header, "solid ", 6u) == 0)
		return true;
	else
	{
		if (_file->getSize() < 84u)
			return false;

		uint32_t triangleCount;

		constexpr size_t readOffset = 80;
		system::IFile::success_t success;
		_file->read(success, &triangleCount, readOffset, sizeof(triangleCount));
		if (!success)
			return false;

		constexpr size_t STL_TRI_SZ = 50u;
		return _file->getSize() == (STL_TRI_SZ * triangleCount + 84u);
	}
}

//! Read 3d vector of floats
void CSTLMeshFileLoader::getNextVector(SContext* context, core::vectorSIMDf& vec, bool binary) const
{
	if (binary)
	{
		{
			system::IFile::success_t success;
			context->inner.mainFile->read(success, &vec.X, context->fileOffset, 4);
			context->fileOffset += success.getBytesProcessed();
		}
		
		{
			system::IFile::success_t success;
			context->inner.mainFile->read(success, &vec.Y, context->fileOffset, 4);
			context->fileOffset += success.getBytesProcessed();
		}

		{
			system::IFile::success_t success;
			context->inner.mainFile->read(success, &vec.Z, context->fileOffset, 4);
			context->fileOffset += success.getBytesProcessed();
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
}

//! Read next word
const std::string& CSTLMeshFileLoader::getNextToken(SContext* context, std::string& token) const
{
	goNextWord(context);
	char c;
	token = "";

	while (context->fileOffset != context->inner.mainFile->getSize())
	{
		system::IFile::success_t success;
		context->inner.mainFile->read(success, &c, context->fileOffset, sizeof(c));
		context->fileOffset += success.getBytesProcessed();

		// found it, so leave
		if (core::isspace(c))
			break;
		token += c;
	}
	return token;
}

//! skip to next word
void CSTLMeshFileLoader::goNextWord(SContext* context) const
{
	uint8_t c;
	while (context->fileOffset != context->inner.mainFile->getSize()) // TODO: check it
	{
		system::IFile::success_t success;
		context->inner.mainFile->read(success, &c, context->fileOffset, sizeof(c));
		context->fileOffset += success.getBytesProcessed();

		// found it, so leave
		if (!core::isspace(c))
		{
			context->fileOffset -= success.getBytesProcessed();
			break;
		}
	}
}

//! Read until line break is reached and stop at the next non-space character
void CSTLMeshFileLoader::goNextLine(SContext* context) const
{
	uint8_t c;
	// look for newline characters
	while (context->fileOffset != context->inner.mainFile->getSize()) // TODO: check it
	{
		system::IFile::success_t success;
		context->inner.mainFile->read(success, &c, context->fileOffset, sizeof(c));
		context->fileOffset += success.getBytesProcessed();

		// found it, so leave
		if (c == '\n' || c == '\r')
			break;
	}
}


#endif // _NBL_COMPILE_WITH_STL_LOADER_
