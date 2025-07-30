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

using namespace nbl;
using namespace nbl::asset;

constexpr auto POSITION_ATTRIBUTE = 0;
constexpr auto COLOR_ATTRIBUTE = 1;
constexpr auto UV_ATTRIBUTE = 2;
constexpr auto NORMAL_ATTRIBUTE = 3;

struct SContext
{
	IAssetLoader::SAssetLoadContext inner;
	uint32_t topHierarchyLevel;
	IAssetLoader::IAssetLoaderOverride* loaderOverride;

	size_t fileOffset = {};

	// skips to the first non-space character available
	void goNextWord()
	{
		uint8_t c;
		while (fileOffset != inner.mainFile->getSize()) // TODO: check it
		{
			system::IFile::success_t success;
			inner.mainFile->read(success, &c, fileOffset, sizeof(c));
			fileOffset += success.getBytesProcessed();

			// found it, so leave
			if (!core::isspace(c))
			{
				fileOffset -= success.getBytesProcessed();
				break;
			}
		}
	}

	// returns the next word
	const std::string& getNextToken(std::string& token)
	{
		goNextWord();
		char c;
		token = "";

		while (fileOffset != inner.mainFile->getSize())
		{
			system::IFile::success_t success;
			inner.mainFile->read(success, &c, fileOffset, sizeof(c));
			fileOffset += success.getBytesProcessed();

			// found it, so leave
			if (core::isspace(c))
				break;
			token += c;
		}
		return token;
	}

	// skip to next printable character after the first line break
	void goNextLine()
	{
		uint8_t c;
		// look for newline characters
		while (fileOffset != inner.mainFile->getSize()) // TODO: check it
		{
			system::IFile::success_t success;
			inner.mainFile->read(success, &c, fileOffset, sizeof(c));
			fileOffset += success.getBytesProcessed();

			// found it, so leave
			if (c == '\n' || c == '\r')
				break;
		}
	}

	//! Read 3d vector of floats
	void getNextVector(hlsl::float32_t3& vec, bool binary)
	{
		if (binary)
		{
			{
				system::IFile::success_t success;
				inner.mainFile->read(success, &vec.x, fileOffset, 4);
				fileOffset += success.getBytesProcessed();
			}

			{
				system::IFile::success_t success;
				inner.mainFile->read(success, &vec.y, fileOffset, 4);
				fileOffset += success.getBytesProcessed();
			}

			{
				system::IFile::success_t success;
				inner.mainFile->read(success, &vec.z, fileOffset, 4);
				fileOffset += success.getBytesProcessed();
			}
		}
		else
		{
			goNextWord();
			std::string tmp;

			getNextToken(tmp);
			sscanf(tmp.c_str(), "%f", &vec.x);
			getNextToken(tmp);
			sscanf(tmp.c_str(), "%f", &vec.y);
			getNextToken(tmp);
			sscanf(tmp.c_str(), "%f", &vec.z);
		}
		vec.x = -vec.x;
	}
};

SAssetBundle CSTLMeshFileLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	using namespace nbl::core;
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

	auto geometry = make_smart_refctd_ptr<ICPUPolygonGeometry>();

	const size_t filesize = context.inner.mainFile->getSize();
	if (filesize < 6ull) // we need a header
		return {};

	bool hasColor = false;

	bool binary = false;
	std::string token;
	if (context.getNextToken(token) != "solid")
		binary = hasColor = true;

	core::vector<hlsl::float32_t3> positions, normals;
	core::vector<uint32_t> colors;
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
		colors.reserve(vertexCount);
	}
	else
		context.goNextLine(); // skip header

	uint16_t attrib = 0u;
	token.reserve(32);
	while (context.fileOffset < filesize) // TODO: check it
	{
		if (!binary)
		{
			if (context.getNextToken(token) != "facet")
			{
				if (token == "endsolid")
					break;
				return {};
			}
			if (context.getNextToken(token) != "normal")
			{
				return {};
			}
		}

		{
			hlsl::float32_t3 n;
			context.getNextVector(n, binary);
			if(_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
				performActionBasedOnOrientationSystem<float>(n.x, [](float& varToFlip) {varToFlip = -varToFlip;});
			normals.push_back(hlsl::normalize(n));
		}

		if (!binary)
		{
			if (context.getNextToken(token) != "outer" || context.getNextToken(token) != "loop")
				return {};
		}

		{
			hlsl::float32_t3 p[3];
			for (uint32_t i = 0u; i < 3u; ++i)
			{
				if (!binary)
				{
					if (context.getNextToken(token) != "vertex")
						return {};
				}
				context.getNextVector(p[i], binary);
				if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
					performActionBasedOnOrientationSystem<float>(p[i].x, [](float& varToFlip){varToFlip = -varToFlip; });
			}
			for (uint32_t i = 0u; i < 3u; ++i) // seems like in STL format vertices are ordered in clockwise manner...
				positions.push_back(p[2u - i]);


		}

		if (!binary)
		{
			if (context.getNextToken(token) != "endloop" || context.getNextToken(token) != "endfacet")
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

		if (normals.back() == hlsl::float32_t3{})
		{
			assert(false);
			static auto computeNormal = [](const hlsl::float32_t3& v1, const hlsl::float32_t3& v2, const hlsl::float32_t3& v3)
				{
					return hlsl::normalize(hlsl::cross(v2 - v1, v3 - v1));
				};

			auto& pos1 = *(positions.rbegin() + 2);
			auto& pos2 = *(positions.rbegin() + 1);
			auto& pos3 = *(positions.rbegin() + 0);
			normals.back() = computeNormal(pos1, pos2, pos3);
		}
	} // end while (_file->getPos() < filesize)

	geometry->setPositionView(createView(E_FORMAT::EF_R32G32B32_SFLOAT, positions.size(), positions.data()));
	geometry->setNormalView(createView(E_FORMAT::EF_R32G32B32_SFLOAT, normals.size(), normals.data()));

	// TODO: Vertex colors

	CPolygonGeometryManipulator::recomputeContentHashes(geometry.get());
	CPolygonGeometryManipulator::recomputeRanges(geometry.get());

	geometry->setIndexing(IPolygonGeometryBase::TriangleList());

	CPolygonGeometryManipulator::recomputeAABB(geometry.get());

	auto meta = make_smart_refctd_ptr<CSTLMetadata>();
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

#endif // _NBL_COMPILE_WITH_STL_LOADER_
