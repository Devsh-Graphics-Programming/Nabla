// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CSTLMeshFileLoader.h"

#ifdef _NBL_COMPILE_WITH_STL_LOADER_

#include "nbl/asset/asset.h"
#include "nbl/asset/metadata/CSTLMetadata.h"
#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
#include "nbl/system/IFile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>

namespace nbl::asset
{

struct SSTLContext
{
	IAssetLoader::SAssetLoadContext inner;
	size_t fileOffset = 0ull;
};

static bool stlReadExact(system::IFile* file, void* dst, const size_t offset, const size_t bytes)
{
	if (!file || (!dst && bytes != 0ull))
		return false;
	if (bytes == 0ull)
		return true;

	system::IFile::success_t success;
	file->read(success, dst, offset, bytes);
	return success && success.getBytesProcessed() == bytes;
}

static bool stlReadWithPolicy(system::IFile* file, uint8_t* dst, const size_t offset, const size_t bytes, const SResolvedFileIOPolicy& ioPlan)
{
	if (!file || (!dst && bytes != 0ull))
		return false;
	if (bytes == 0ull)
		return true;

	size_t bytesRead = 0ull;
	switch (ioPlan.strategy)
	{
		case SResolvedFileIOPolicy::Strategy::WholeFile:
			return stlReadExact(file, dst, offset, bytes);
		case SResolvedFileIOPolicy::Strategy::Chunked:
		default:
			while (bytesRead < bytes)
			{
				const size_t chunk = static_cast<size_t>(std::min<uint64_t>(ioPlan.chunkSizeBytes, bytes - bytesRead));
				system::IFile::success_t success;
				file->read(success, dst + bytesRead, offset + bytesRead, chunk);
				if (!success)
					return false;
				const size_t processed = success.getBytesProcessed();
				if (processed == 0ull)
					return false;
				bytesRead += processed;
			}
			return true;
	}
}

static bool stlReadU8(SSTLContext* context, uint8_t& out)
{
	if (!context)
		return false;

	system::IFile::success_t success;
	context->inner.mainFile->read(success, &out, context->fileOffset, sizeof(out));
	if (!success || success.getBytesProcessed() != sizeof(out))
		return false;
	context->fileOffset += sizeof(out);
	return true;
}

static bool stlReadF32(SSTLContext* context, float& out)
{
	if (!context)
		return false;

	system::IFile::success_t success;
	context->inner.mainFile->read(success, &out, context->fileOffset, sizeof(out));
	if (!success || success.getBytesProcessed() != sizeof(out))
		return false;
	context->fileOffset += sizeof(out);
	return true;
}

static void stlGoNextWord(SSTLContext* context)
{
	if (!context)
		return;

	uint8_t c = 0u;
	while (context->fileOffset < context->inner.mainFile->getSize())
	{
		const size_t before = context->fileOffset;
		if (!stlReadU8(context, c))
			break;
		if (!core::isspace(c))
		{
			context->fileOffset = before;
			break;
		}
	}
}

static const std::string& stlGetNextToken(SSTLContext* context, std::string& token)
{
	stlGoNextWord(context);
	token.clear();

	char c = 0;
	while (context->fileOffset < context->inner.mainFile->getSize())
	{
		system::IFile::success_t success;
		context->inner.mainFile->read(success, &c, context->fileOffset, sizeof(c));
		if (!success || success.getBytesProcessed() != sizeof(c))
			break;
		context->fileOffset += sizeof(c);
		if (core::isspace(c))
			break;
		token += c;
	}

	return token;
}

static void stlGoNextLine(SSTLContext* context)
{
	if (!context)
		return;

	uint8_t c = 0u;
	while (context->fileOffset < context->inner.mainFile->getSize())
	{
		if (!stlReadU8(context, c))
			break;
		if (c == '\n' || c == '\r')
			break;
	}
}

static bool stlGetNextVector(SSTLContext* context, core::vectorSIMDf& vec, const bool binary)
{
	if (!context)
		return false;

	if (binary)
	{
		float x = 0.f;
		float y = 0.f;
		float z = 0.f;
		if (!stlReadF32(context, x) || !stlReadF32(context, y) || !stlReadF32(context, z))
			return false;
		vec.set(x, y, z, 0.f);
		return true;
	}

	stlGoNextWord(context);
	std::string tmp;
	if (stlGetNextToken(context, tmp).empty())
		return false;
	std::sscanf(tmp.c_str(), "%f", &vec.X);
	if (stlGetNextToken(context, tmp).empty())
		return false;
	std::sscanf(tmp.c_str(), "%f", &vec.Y);
	if (stlGetNextToken(context, tmp).empty())
		return false;
	std::sscanf(tmp.c_str(), "%f", &vec.Z);
	vec.W = 0.f;
	return true;
}

static bool stlReadFloatFromPayload(const uint8_t*& cursor, const uint8_t* const end, float& out)
{
	if (cursor + sizeof(float) > end)
		return false;
	std::memcpy(&out, cursor, sizeof(float));
	cursor += sizeof(float);
	return true;
}

CSTLMeshFileLoader::CSTLMeshFileLoader(asset::IAssetManager* _assetManager)
{
	(void)_assetManager;
}

const char** CSTLMeshFileLoader::getAssociatedFileExtensions() const
{
	static const char* ext[] = { "stl", nullptr };
	return ext;
}

SAssetBundle CSTLMeshFileLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	(void)_override;
	(void)_hierarchyLevel;

	if (!_file)
		return {};

	using clock_t = std::chrono::high_resolution_clock;
	const auto totalStart = clock_t::now();
	double detectMs = 0.0;
	double ioMs = 0.0;
	double parseMs = 0.0;
	double buildMs = 0.0;
	double hashMs = 0.0;
	double aabbMs = 0.0;
	uint64_t triangleCount = 0u;

	SSTLContext context = {
		asset::IAssetLoader::SAssetLoadContext{
			_params,
			_file
		},
		0ull
	};

	const size_t filesize = context.inner.mainFile->getSize();
	if (filesize < 6ull)
		return {};

	const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(filesize), true);
	if (!ioPlan.valid)
	{
		_params.logger.log("STL loader: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str(), ioPlan.reason);
		return {};
	}

	bool binary = false;
	std::string token;
	{
		const auto detectStart = clock_t::now();
		char header[6] = {};
		if (!stlReadExact(context.inner.mainFile, header, 0ull, sizeof(header)))
			return {};

		const bool startsWithSolid = (std::strncmp(header, "solid ", 6u) == 0);
		bool binaryBySize = false;
		if (filesize >= 84ull)
		{
			uint32_t triCount = 0u;
			if (stlReadExact(context.inner.mainFile, &triCount, 80ull, sizeof(triCount)))
			{
				const uint64_t expectedSize = 84ull + static_cast<uint64_t>(triCount) * 50ull;
				binaryBySize = (expectedSize == filesize);
			}
		}

		if (binaryBySize)
			binary = true;
		else if (!startsWithSolid)
			binary = true;
		else
			binary = (stlGetNextToken(&context, token) != "solid");

		if (binary)
			context.fileOffset = 0ull;
		detectMs = std::chrono::duration<double, std::milli>(clock_t::now() - detectStart).count();
	}

	core::vector<core::vectorSIMDf> positions;
	core::vector<core::vectorSIMDf> normals;

	if (binary)
	{
		if (filesize < 84ull)
			return {};

		uint32_t triangleCount32 = 0u;
		if (!stlReadExact(context.inner.mainFile, &triangleCount32, 80ull, sizeof(triangleCount32)))
			return {};

		triangleCount = triangleCount32;
		const size_t dataSize = static_cast<size_t>(triangleCount) * 50ull;
		const size_t expectedSize = 84ull + dataSize;
		if (filesize < expectedSize)
			return {};

		core::vector<uint8_t> payload;
		payload.resize(dataSize);

		const auto ioStart = clock_t::now();
		if (!stlReadWithPolicy(context.inner.mainFile, payload.data(), 84ull, dataSize, ioPlan))
			return {};
		ioMs = std::chrono::duration<double, std::milli>(clock_t::now() - ioStart).count();

		positions.reserve(static_cast<size_t>(triangleCount) * 3ull);
		normals.reserve(static_cast<size_t>(triangleCount));

		const auto parseStart = clock_t::now();
		const uint8_t* cursor = payload.data();
		const uint8_t* const end = cursor + payload.size();
		for (uint64_t tri = 0ull; tri < triangleCount; ++tri)
		{
			float nx = 0.f;
			float ny = 0.f;
			float nz = 0.f;
			if (!stlReadFloatFromPayload(cursor, end, nx) || !stlReadFloatFromPayload(cursor, end, ny) || !stlReadFloatFromPayload(cursor, end, nz))
				return {};

			core::vectorSIMDf fileNormal;
			fileNormal.set(nx, ny, nz, 0.f);
			const float fileLen2 = core::dot(fileNormal, fileNormal).X;
			if (fileLen2 > 0.f && std::abs(fileLen2 - 1.f) < 1e-4f)
				normals.push_back(fileNormal);
			else
				normals.push_back(core::normalize(fileNormal));

			core::vectorSIMDf p[3] = {};
			for (uint32_t i = 0u; i < 3u; ++i)
			{
				float x = 0.f;
				float y = 0.f;
				float z = 0.f;
				if (!stlReadFloatFromPayload(cursor, end, x) || !stlReadFloatFromPayload(cursor, end, y) || !stlReadFloatFromPayload(cursor, end, z))
					return {};
				p[i].set(x, y, z, 0.f);
			}

			positions.push_back(p[2u]);
			positions.push_back(p[1u]);
			positions.push_back(p[0u]);

			if ((normals.back() == core::vectorSIMDf()).all())
			{
				normals.back().set(core::plane3dSIMDf(
					*(positions.rbegin() + 2),
					*(positions.rbegin() + 1),
					*(positions.rbegin() + 0)).getNormal());
			}

			if (cursor + sizeof(uint16_t) > end)
				return {};
			cursor += sizeof(uint16_t);
		}
		parseMs = std::chrono::duration<double, std::milli>(clock_t::now() - parseStart).count();
	}
	else
	{
		stlGoNextLine(&context);
		token.reserve(32);

		const auto parseStart = clock_t::now();
		while (context.fileOffset < filesize)
		{
			if (stlGetNextToken(&context, token) != "facet")
			{
				if (token == "endsolid")
					break;
				return {};
			}
			if (stlGetNextToken(&context, token) != "normal")
				return {};

			core::vectorSIMDf fileNormal;
			if (!stlGetNextVector(&context, fileNormal, false))
				return {};

			const float fileLen2 = core::dot(fileNormal, fileNormal).X;
			if (fileLen2 > 0.f && std::abs(fileLen2 - 1.f) < 1e-4f)
				normals.push_back(fileNormal);
			else
				normals.push_back(core::normalize(fileNormal));

			if (stlGetNextToken(&context, token) != "outer" || stlGetNextToken(&context, token) != "loop")
				return {};

			core::vectorSIMDf p[3] = {};
			for (uint32_t i = 0u; i < 3u; ++i)
			{
				if (stlGetNextToken(&context, token) != "vertex")
					return {};
				if (!stlGetNextVector(&context, p[i], false))
					return {};
			}

			positions.push_back(p[2u]);
			positions.push_back(p[1u]);
			positions.push_back(p[0u]);

			if (stlGetNextToken(&context, token) != "endloop" || stlGetNextToken(&context, token) != "endfacet")
				return {};

			if ((normals.back() == core::vectorSIMDf()).all())
			{
				normals.back().set(core::plane3dSIMDf(
					*(positions.rbegin() + 2),
					*(positions.rbegin() + 1),
					*(positions.rbegin() + 0)).getNormal());
			}
		}
		parseMs = std::chrono::duration<double, std::milli>(clock_t::now() - parseStart).count();
	}

	if (positions.empty())
		return {};

	triangleCount = positions.size() / 3ull;
	const uint64_t vertexCount = positions.size();

	const auto buildStart = clock_t::now();
	auto geometry = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	geometry->setIndexing(IPolygonGeometryBase::TriangleList());

	auto posView = createView(EF_R32G32B32_SFLOAT, positions.size());
	auto normalView = createView(EF_R32G32B32_SFLOAT, positions.size());
	if (!posView || !normalView)
		return {};

	auto* posOut = reinterpret_cast<hlsl::float32_t3*>(posView.getPointer());
	auto* normalOut = reinterpret_cast<hlsl::float32_t3*>(normalView.getPointer());
	if (!posOut || !normalOut)
		return {};

	hlsl::shapes::AABB<3, hlsl::float32_t> parsedAABB = hlsl::shapes::AABB<3, hlsl::float32_t>::create();
	bool hasParsedAABB = false;
	auto addAABBPoint = [&parsedAABB, &hasParsedAABB](const hlsl::float32_t3& p)->void
	{
		if (!hasParsedAABB)
		{
			parsedAABB.minVx = p;
			parsedAABB.maxVx = p;
			hasParsedAABB = true;
			return;
		}
		if (p.x < parsedAABB.minVx.x) parsedAABB.minVx.x = p.x;
		if (p.y < parsedAABB.minVx.y) parsedAABB.minVx.y = p.y;
		if (p.z < parsedAABB.minVx.z) parsedAABB.minVx.z = p.z;
		if (p.x > parsedAABB.maxVx.x) parsedAABB.maxVx.x = p.x;
		if (p.y > parsedAABB.maxVx.y) parsedAABB.maxVx.y = p.y;
		if (p.z > parsedAABB.maxVx.z) parsedAABB.maxVx.z = p.z;
	};

	for (size_t i = 0u; i < positions.size(); ++i)
	{
		const auto& pos = positions[i];
		const auto& nrm = normals[i / 3u];
		posOut[i] = { pos.X, pos.Y, pos.Z };
		normalOut[i] = { nrm.X, nrm.Y, nrm.Z };
		addAABBPoint(posOut[i]);
	}

	geometry->setPositionView(std::move(posView));
	geometry->setNormalView(std::move(normalView));
	buildMs = std::chrono::duration<double, std::milli>(clock_t::now() - buildStart).count();

	const auto hashStart = clock_t::now();
	CPolygonGeometryManipulator::recomputeContentHashes(geometry.get());
	hashMs = std::chrono::duration<double, std::milli>(clock_t::now() - hashStart).count();

	const auto aabbStart = clock_t::now();
	if (hasParsedAABB)
	{
		geometry->visitAABB([&parsedAABB](auto& ref)->void
		{
			ref = std::remove_reference_t<decltype(ref)>::create();
			ref.minVx.x = parsedAABB.minVx.x;
			ref.minVx.y = parsedAABB.minVx.y;
			ref.minVx.z = parsedAABB.minVx.z;
			ref.minVx.w = 0.0;
			ref.maxVx.x = parsedAABB.maxVx.x;
			ref.maxVx.y = parsedAABB.maxVx.y;
			ref.maxVx.z = parsedAABB.maxVx.z;
			ref.maxVx.w = 0.0;
		});
	}
	else
	{
		CPolygonGeometryManipulator::recomputeAABB(geometry.get());
	}
	aabbMs = std::chrono::duration<double, std::milli>(clock_t::now() - aabbStart).count();

	const auto totalMs = std::chrono::duration<double, std::milli>(clock_t::now() - totalStart).count();
	_params.logger.log(
		"STL loader perf: file=%s total=%.3f ms detect=%.3f io=%.3f parse=%.3f build=%.3f hash=%.3f aabb=%.3f binary=%d triangles=%llu vertices=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
		system::ILogger::ELL_PERFORMANCE,
		_file->getFileName().string().c_str(),
		totalMs,
		detectMs,
		ioMs,
		parseMs,
		buildMs,
		hashMs,
		aabbMs,
		binary ? 1 : 0,
		static_cast<unsigned long long>(triangleCount),
		static_cast<unsigned long long>(vertexCount),
		toString(_params.ioPolicy.strategy),
		toString(ioPlan.strategy),
		static_cast<unsigned long long>(ioPlan.chunkSizeBytes),
		ioPlan.reason);

	auto meta = core::make_smart_refctd_ptr<CSTLMetadata>();
	return SAssetBundle(std::move(meta), { std::move(geometry) });
}

bool CSTLMeshFileLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
{
	(void)logger;
	if (!_file || _file->getSize() <= 6u)
		return false;

	char header[6] = {};
	if (!stlReadExact(_file, header, 0ull, sizeof(header)))
		return false;

	if (std::strncmp(header, "solid ", 6u) == 0)
		return true;

	if (_file->getSize() < 84u)
		return false;

	uint32_t triangleCount = 0u;
	if (!stlReadExact(_file, &triangleCount, 80ull, sizeof(triangleCount)))
		return false;

	constexpr size_t STL_TRI_SZ = sizeof(float) * 12ull + sizeof(uint16_t);
	return _file->getSize() == (STL_TRI_SZ * triangleCount + 84u);
}

}

#endif // _NBL_COMPILE_WITH_STL_LOADER_
