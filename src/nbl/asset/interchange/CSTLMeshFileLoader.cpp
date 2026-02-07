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

constexpr size_t StlTextProbeBytes = 6ull;
constexpr size_t StlBinaryHeaderBytes = 80ull;
constexpr size_t StlTriangleCountBytes = sizeof(uint32_t);
constexpr size_t StlBinaryPrefixBytes = StlBinaryHeaderBytes + StlTriangleCountBytes;
constexpr size_t StlTriangleFloatCount = 12ull;
constexpr size_t StlTriangleFloatBytes = sizeof(float) * StlTriangleFloatCount;
constexpr size_t StlTriangleAttributeBytes = sizeof(uint16_t);
constexpr size_t StlTriangleRecordBytes = StlTriangleFloatBytes + StlTriangleAttributeBytes;
constexpr size_t StlVerticesPerTriangle = 3ull;
constexpr size_t StlFloatChannelsPerVertex = 3ull;
constexpr size_t StlFloatsPerTriangleVertices = StlVerticesPerTriangle * StlFloatChannelsPerVertex;
constexpr size_t StlFloatsPerTriangleOutput = StlFloatsPerTriangleVertices;

bool stlReadExact(system::IFile* file, void* dst, const size_t offset, const size_t bytes)
{
	if (!file || (!dst && bytes != 0ull))
		return false;
	if (bytes == 0ull)
		return true;

	system::IFile::success_t success;
	file->read(success, dst, offset, bytes);
	return success && success.getBytesProcessed() == bytes;
}

bool stlReadWithPolicy(system::IFile* file, uint8_t* dst, const size_t offset, const size_t bytes, const SResolvedFileIOPolicy& ioPlan)
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

bool stlReadU8(SSTLContext* context, uint8_t& out)
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

bool stlReadF32(SSTLContext* context, float& out)
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

void stlGoNextWord(SSTLContext* context)
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

const std::string& stlGetNextToken(SSTLContext* context, std::string& token)
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

void stlGoNextLine(SSTLContext* context)
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

bool stlGetNextVector(SSTLContext* context, hlsl::float32_t3& vec, const bool binary)
{
	if (!context)
		return false;

	if (binary)
	{
		if (!stlReadF32(context, vec.x) || !stlReadF32(context, vec.y) || !stlReadF32(context, vec.z))
			return false;
		return true;
	}

	stlGoNextWord(context);
	std::string tmp;
	if (stlGetNextToken(context, tmp).empty())
		return false;
	std::sscanf(tmp.c_str(), "%f", &vec.x);
	if (stlGetNextToken(context, tmp).empty())
		return false;
	std::sscanf(tmp.c_str(), "%f", &vec.y);
	if (stlGetNextToken(context, tmp).empty())
		return false;
	std::sscanf(tmp.c_str(), "%f", &vec.z);
	return true;
}

hlsl::float32_t3 stlNormalizeOrZero(const hlsl::float32_t3& v)
{
	const float len2 = hlsl::dot(v, v);
	if (len2 <= 0.f)
		return hlsl::float32_t3(0.f, 0.f, 0.f);
	return hlsl::normalize(v);
}

hlsl::float32_t3 stlComputeFaceNormal(const hlsl::float32_t3& a, const hlsl::float32_t3& b, const hlsl::float32_t3& c)
{
	return stlNormalizeOrZero(hlsl::cross(b - a, c - a));
}

hlsl::float32_t3 stlResolveStoredNormal(const hlsl::float32_t3& fileNormal)
{
	const float fileLen2 = hlsl::dot(fileNormal, fileNormal);
	if (fileLen2 > 0.f && std::abs(fileLen2 - 1.f) < 1e-4f)
		return fileNormal;
	return stlNormalizeOrZero(fileNormal);
}

void stlPushTriangleReversed(const hlsl::float32_t3 (&p)[3], core::vector<hlsl::float32_t3>& positions)
{
	positions.push_back(p[2u]);
	positions.push_back(p[1u]);
	positions.push_back(p[0u]);
}

void stlFixLastFaceNormal(core::vector<hlsl::float32_t3>& normals, const core::vector<hlsl::float32_t3>& positions)
{
	if (normals.empty() || positions.size() < 3ull)
		return;

	const auto& lastNormal = normals.back();
	if (hlsl::dot(lastNormal, lastNormal) > 0.f)
		return;

	normals.back() = stlComputeFaceNormal(*(positions.rbegin() + 2), *(positions.rbegin() + 1), *(positions.rbegin() + 0));
}

void stlExtendAABB(hlsl::shapes::AABB<3, hlsl::float32_t>& aabb, bool& hasAABB, const hlsl::float32_t3& p)
{
	if (!hasAABB)
	{
		aabb.minVx = p;
		aabb.maxVx = p;
		hasAABB = true;
		return;
	}

	if (p.x < aabb.minVx.x) aabb.minVx.x = p.x;
	if (p.y < aabb.minVx.y) aabb.minVx.y = p.y;
	if (p.z < aabb.minVx.z) aabb.minVx.z = p.z;
	if (p.x > aabb.maxVx.x) aabb.maxVx.x = p.x;
	if (p.y > aabb.maxVx.y) aabb.maxVx.y = p.y;
	if (p.z > aabb.maxVx.z) aabb.maxVx.z = p.z;
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
	if (filesize < StlTextProbeBytes)
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
		char header[StlTextProbeBytes] = {};
		if (!stlReadExact(context.inner.mainFile, header, 0ull, sizeof(header)))
			return {};

		const bool startsWithSolid = (std::strncmp(header, "solid ", StlTextProbeBytes) == 0);
		bool binaryBySize = false;
		if (filesize >= StlBinaryPrefixBytes)
		{
			uint32_t triCount = 0u;
			if (stlReadExact(context.inner.mainFile, &triCount, StlBinaryHeaderBytes, sizeof(triCount)))
			{
				const uint64_t expectedSize = StlBinaryPrefixBytes + static_cast<uint64_t>(triCount) * StlTriangleRecordBytes;
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

	auto geometry = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	geometry->setIndexing(IPolygonGeometryBase::TriangleList());
	hlsl::shapes::AABB<3, hlsl::float32_t> parsedAABB = hlsl::shapes::AABB<3, hlsl::float32_t>::create();
	bool hasParsedAABB = false;
	uint64_t vertexCount = 0ull;

	if (binary)
	{
		if (filesize < StlBinaryPrefixBytes)
			return {};

		uint32_t triangleCount32 = 0u;
		if (!stlReadExact(context.inner.mainFile, &triangleCount32, StlBinaryHeaderBytes, sizeof(triangleCount32)))
			return {};

		triangleCount = triangleCount32;
		const size_t dataSize = static_cast<size_t>(triangleCount) * StlTriangleRecordBytes;
		const size_t expectedSize = StlBinaryPrefixBytes + dataSize;
		if (filesize < expectedSize)
			return {};

		core::vector<uint8_t> payload;
		payload.resize(dataSize);

		const auto ioStart = clock_t::now();
		if (!stlReadWithPolicy(context.inner.mainFile, payload.data(), StlBinaryPrefixBytes, dataSize, ioPlan))
			return {};
		ioMs = std::chrono::duration<double, std::milli>(clock_t::now() - ioStart).count();

		vertexCount = triangleCount * StlVerticesPerTriangle;
		const auto buildPrepStart = clock_t::now();
		auto posView = createView(EF_R32G32B32_SFLOAT, static_cast<size_t>(vertexCount));
		auto normalView = createView(EF_R32G32B32_SFLOAT, static_cast<size_t>(vertexCount));
		if (!posView || !normalView)
			return {};

		auto* posOut = reinterpret_cast<hlsl::float32_t3*>(posView.getPointer());
		auto* normalOut = reinterpret_cast<hlsl::float32_t3*>(normalView.getPointer());
		if (!posOut || !normalOut)
			return {};
		buildMs += std::chrono::duration<double, std::milli>(clock_t::now() - buildPrepStart).count();

		const auto parseStart = clock_t::now();
		const uint8_t* cursor = payload.data();
		const uint8_t* const end = cursor + payload.size();
		auto* posOutFloat = reinterpret_cast<float*>(posOut);
		auto* normalOutFloat = reinterpret_cast<float*>(normalOut);
		for (uint64_t tri = 0ull; tri < triangleCount; ++tri)
		{
			if (cursor + StlTriangleRecordBytes > end)
				return {};

			float triData[StlTriangleFloatCount] = {};
			std::memcpy(triData, cursor, StlTriangleFloatBytes);
			cursor += StlTriangleFloatBytes;
			cursor += StlTriangleAttributeBytes;

			const float vertex0x = triData[9];
			const float vertex0y = triData[10];
			const float vertex0z = triData[11];
			const float vertex1x = triData[6];
			const float vertex1y = triData[7];
			const float vertex1z = triData[8];
			const float vertex2x = triData[3];
			const float vertex2y = triData[4];
			const float vertex2z = triData[5];

			float normalX = triData[0];
			float normalY = triData[1];
			float normalZ = triData[2];
			const float normalLen2 = normalX * normalX + normalY * normalY + normalZ * normalZ;
			if (normalLen2 <= 0.f)
			{
				const float edge10x = vertex1x - vertex0x;
				const float edge10y = vertex1y - vertex0y;
				const float edge10z = vertex1z - vertex0z;
				const float edge20x = vertex2x - vertex0x;
				const float edge20y = vertex2y - vertex0y;
				const float edge20z = vertex2z - vertex0z;

				normalX = edge10y * edge20z - edge10z * edge20y;
				normalY = edge10z * edge20x - edge10x * edge20z;
				normalZ = edge10x * edge20y - edge10y * edge20x;
				const float planeLen2 = normalX * normalX + normalY * normalY + normalZ * normalZ;
				if (planeLen2 > 0.f)
				{
					const float invLen = 1.f / std::sqrt(planeLen2);
					normalX *= invLen;
					normalY *= invLen;
					normalZ *= invLen;
				}
				else
				{
					normalX = 0.f;
					normalY = 0.f;
					normalZ = 0.f;
				}
			}
			else if (std::abs(normalLen2 - 1.f) >= 1e-4f)
			{
				const float invLen = 1.f / std::sqrt(normalLen2);
				normalX *= invLen;
				normalY *= invLen;
				normalZ *= invLen;
			}

			const size_t base = static_cast<size_t>(tri) * StlFloatsPerTriangleOutput;
			posOutFloat[base + 0ull] = vertex0x;
			posOutFloat[base + 1ull] = vertex0y;
			posOutFloat[base + 2ull] = vertex0z;
			posOutFloat[base + 3ull] = vertex1x;
			posOutFloat[base + 4ull] = vertex1y;
			posOutFloat[base + 5ull] = vertex1z;
			posOutFloat[base + 6ull] = vertex2x;
			posOutFloat[base + 7ull] = vertex2y;
			posOutFloat[base + 8ull] = vertex2z;

			normalOutFloat[base + 0ull] = normalX;
			normalOutFloat[base + 1ull] = normalY;
			normalOutFloat[base + 2ull] = normalZ;
			normalOutFloat[base + 3ull] = normalX;
			normalOutFloat[base + 4ull] = normalY;
			normalOutFloat[base + 5ull] = normalZ;
			normalOutFloat[base + 6ull] = normalX;
			normalOutFloat[base + 7ull] = normalY;
			normalOutFloat[base + 8ull] = normalZ;

			if (!hasParsedAABB)
			{
				hasParsedAABB = true;
				parsedAABB.minVx.x = vertex0x;
				parsedAABB.minVx.y = vertex0y;
				parsedAABB.minVx.z = vertex0z;
				parsedAABB.maxVx.x = vertex0x;
				parsedAABB.maxVx.y = vertex0y;
				parsedAABB.maxVx.z = vertex0z;
			}

			if (vertex0x < parsedAABB.minVx.x) parsedAABB.minVx.x = vertex0x;
			if (vertex0y < parsedAABB.minVx.y) parsedAABB.minVx.y = vertex0y;
			if (vertex0z < parsedAABB.minVx.z) parsedAABB.minVx.z = vertex0z;
			if (vertex1x < parsedAABB.minVx.x) parsedAABB.minVx.x = vertex1x;
			if (vertex1y < parsedAABB.minVx.y) parsedAABB.minVx.y = vertex1y;
			if (vertex1z < parsedAABB.minVx.z) parsedAABB.minVx.z = vertex1z;
			if (vertex2x < parsedAABB.minVx.x) parsedAABB.minVx.x = vertex2x;
			if (vertex2y < parsedAABB.minVx.y) parsedAABB.minVx.y = vertex2y;
			if (vertex2z < parsedAABB.minVx.z) parsedAABB.minVx.z = vertex2z;

			if (vertex0x > parsedAABB.maxVx.x) parsedAABB.maxVx.x = vertex0x;
			if (vertex0y > parsedAABB.maxVx.y) parsedAABB.maxVx.y = vertex0y;
			if (vertex0z > parsedAABB.maxVx.z) parsedAABB.maxVx.z = vertex0z;
			if (vertex1x > parsedAABB.maxVx.x) parsedAABB.maxVx.x = vertex1x;
			if (vertex1y > parsedAABB.maxVx.y) parsedAABB.maxVx.y = vertex1y;
			if (vertex1z > parsedAABB.maxVx.z) parsedAABB.maxVx.z = vertex1z;
			if (vertex2x > parsedAABB.maxVx.x) parsedAABB.maxVx.x = vertex2x;
			if (vertex2y > parsedAABB.maxVx.y) parsedAABB.maxVx.y = vertex2y;
			if (vertex2z > parsedAABB.maxVx.z) parsedAABB.maxVx.z = vertex2z;
		}
		parseMs = std::chrono::duration<double, std::milli>(clock_t::now() - parseStart).count();

		const auto buildFinalizeStart = clock_t::now();
		geometry->setPositionView(std::move(posView));
		geometry->setNormalView(std::move(normalView));
		buildMs += std::chrono::duration<double, std::milli>(clock_t::now() - buildFinalizeStart).count();
	}
	else
	{
		core::vector<hlsl::float32_t3> positions;
		core::vector<hlsl::float32_t3> normals;
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

			hlsl::float32_t3 fileNormal = {};
			if (!stlGetNextVector(&context, fileNormal, false))
				return {};

			normals.push_back(stlResolveStoredNormal(fileNormal));

			if (stlGetNextToken(&context, token) != "outer" || stlGetNextToken(&context, token) != "loop")
				return {};

			hlsl::float32_t3 p[3] = {};
			for (uint32_t i = 0u; i < 3u; ++i)
			{
				if (stlGetNextToken(&context, token) != "vertex")
					return {};
				if (!stlGetNextVector(&context, p[i], false))
					return {};
			}

			stlPushTriangleReversed(p, positions);

			if (stlGetNextToken(&context, token) != "endloop" || stlGetNextToken(&context, token) != "endfacet")
				return {};

			stlFixLastFaceNormal(normals, positions);
		}
		parseMs = std::chrono::duration<double, std::milli>(clock_t::now() - parseStart).count();
		if (positions.empty())
			return {};

		triangleCount = positions.size() / StlVerticesPerTriangle;
		vertexCount = positions.size();

		const auto buildStart = clock_t::now();
		auto posView = createView(EF_R32G32B32_SFLOAT, positions.size());
		auto normalView = createView(EF_R32G32B32_SFLOAT, positions.size());
		if (!posView || !normalView)
			return {};

		auto* posOut = reinterpret_cast<hlsl::float32_t3*>(posView.getPointer());
		auto* normalOut = reinterpret_cast<hlsl::float32_t3*>(normalView.getPointer());
		if (!posOut || !normalOut)
			return {};

		for (size_t i = 0u; i < positions.size(); ++i)
		{
			const auto& pos = positions[i];
			const auto& nrm = normals[i / 3u];
			posOut[i] = { pos.x, pos.y, pos.z };
			normalOut[i] = { nrm.x, nrm.y, nrm.z };
			stlExtendAABB(parsedAABB, hasParsedAABB, posOut[i]);
		}

		geometry->setPositionView(std::move(posView));
		geometry->setNormalView(std::move(normalView));
		buildMs = std::chrono::duration<double, std::milli>(clock_t::now() - buildStart).count();
	}

	if (vertexCount == 0ull)
		return {};

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
	if (!_file || _file->getSize() <= StlTextProbeBytes)
		return false;

	char header[StlTextProbeBytes] = {};
	if (!stlReadExact(_file, header, 0ull, sizeof(header)))
		return false;

	if (std::strncmp(header, "solid ", StlTextProbeBytes) == 0)
		return true;

	if (_file->getSize() < StlBinaryPrefixBytes)
		return false;

	uint32_t triangleCount = 0u;
	if (!stlReadExact(_file, &triangleCount, StlBinaryHeaderBytes, sizeof(triangleCount)))
		return false;

	return _file->getSize() == (StlTriangleRecordBytes * triangleCount + StlBinaryPrefixBytes);
}

}

#endif // _NBL_COMPILE_WITH_STL_LOADER_
