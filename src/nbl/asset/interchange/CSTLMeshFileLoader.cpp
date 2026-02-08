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

#include <array>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fast_float/fast_float.h>
#include <limits>
#include <string_view>
#include <type_traits>

namespace nbl::asset
{

struct SFileReadTelemetry
{
	uint64_t callCount = 0ull;
	uint64_t totalBytes = 0ull;
	uint64_t minBytes = std::numeric_limits<uint64_t>::max();

	void account(const uint64_t bytes)
	{
		++callCount;
		totalBytes += bytes;
		if (bytes < minBytes)
			minBytes = bytes;
	}

	uint64_t getMinOrZero() const
	{
		return callCount ? minBytes : 0ull;
	}

	uint64_t getAvgOrZero() const
	{
		return callCount ? (totalBytes / callCount) : 0ull;
	}
};

struct SSTLContext
{
	IAssetLoader::SAssetLoadContext inner;
	SFileReadTelemetry ioTelemetry = {};
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

bool stlReadExact(system::IFile* file, void* dst, const size_t offset, const size_t bytes, SFileReadTelemetry* ioTelemetry = nullptr)
{
	if (!file || (!dst && bytes != 0ull))
		return false;
	if (bytes == 0ull)
		return true;

	system::IFile::success_t success;
	file->read(success, dst, offset, bytes);
	if (success && ioTelemetry)
		ioTelemetry->account(success.getBytesProcessed());
	return success && success.getBytesProcessed() == bytes;
}

bool stlReadWithPolicy(system::IFile* file, uint8_t* dst, const size_t offset, const size_t bytes, const SResolvedFileIOPolicy& ioPlan, SFileReadTelemetry* ioTelemetry = nullptr)
{
	if (!file || (!dst && bytes != 0ull))
		return false;
	if (bytes == 0ull)
		return true;

	size_t bytesRead = 0ull;
	switch (ioPlan.strategy)
	{
		case SResolvedFileIOPolicy::Strategy::WholeFile:
			return stlReadExact(file, dst, offset, bytes, ioTelemetry);
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
				if (ioTelemetry)
					ioTelemetry->account(processed);
				bytesRead += processed;
			}
			return true;
	}
}

const char* stlSkipWhitespace(const char* ptr, const char* const end)
{
	while (ptr < end && core::isspace(*ptr))
		++ptr;
	return ptr;
}

bool stlReadTextToken(const char*& ptr, const char* const end, std::string_view& outToken)
{
	ptr = stlSkipWhitespace(ptr, end);
	if (ptr >= end)
	{
		outToken = {};
		return false;
	}

	const char* tokenEnd = ptr;
	while (tokenEnd < end && !core::isspace(*tokenEnd))
		++tokenEnd;

	outToken = std::string_view(ptr, static_cast<size_t>(tokenEnd - ptr));
	ptr = tokenEnd;
	return true;
}

bool stlReadTextFloat(const char*& ptr, const char* const end, float& outValue)
{
	ptr = stlSkipWhitespace(ptr, end);
	if (ptr >= end)
		return false;

	const auto parseResult = fast_float::from_chars(ptr, end, outValue);
	if (parseResult.ec == std::errc() && parseResult.ptr != ptr)
	{
		ptr = parseResult.ptr;
		return true;
	}

	char* fallbackEnd = nullptr;
	outValue = std::strtof(ptr, &fallbackEnd);
	if (!fallbackEnd || fallbackEnd == ptr)
		return false;
	ptr = fallbackEnd <= end ? fallbackEnd : end;
	return true;
}

bool stlReadTextVec3(const char*& ptr, const char* const end, hlsl::float32_t3& outVec)
{
	return
		stlReadTextFloat(ptr, end, outVec.x) &&
		stlReadTextFloat(ptr, end, outVec.y) &&
		stlReadTextFloat(ptr, end, outVec.z);
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

ICPUPolygonGeometry::SDataView stlCreateAdoptedFloat3View(core::vector<hlsl::float32_t3>&& values)
{
	if (values.empty())
		return {};

	auto backer = core::make_smart_refctd_ptr<core::adoption_memory_resource<core::vector<hlsl::float32_t3>>>(std::move(values));
	auto& payload = backer->getBacker();
	auto* const payloadPtr = payload.data();
	const size_t byteCount = payload.size() * sizeof(hlsl::float32_t3);
	auto buffer = ICPUBuffer::create(
		{ { byteCount }, payloadPtr, core::smart_refctd_ptr<core::refctd_memory_resource>(std::move(backer)), alignof(hlsl::float32_t3) },
		core::adopt_memory);
	if (!buffer)
		return {};

	ICPUPolygonGeometry::SDataView view = {};
	view.composed = {
		.stride = sizeof(hlsl::float32_t3),
		.format = EF_R32G32B32_SFLOAT,
		.rangeFormat = IGeometryBase::getMatchingAABBFormat(EF_R32G32B32_SFLOAT)
	};
	view.src = {
		.offset = 0u,
		.size = byteCount,
		.buffer = std::move(buffer)
	};
	return view;
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
	double buildAllocViewsMs = 0.0;
	double buildSetViewsMs = 0.0;
	double buildMiscMs = 0.0;
	double hashMs = 0.0;
	double aabbMs = 0.0;
	uint64_t triangleCount = 0u;
	const char* parsePath = "unknown";

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
	bool hasBinaryTriCountFromDetect = false;
	uint32_t binaryTriCountFromDetect = 0u;
	{
		const auto detectStart = clock_t::now();
		std::array<uint8_t, StlBinaryPrefixBytes> prefix = {};
		const bool hasPrefix = filesize >= StlBinaryPrefixBytes && stlReadExact(context.inner.mainFile, prefix.data(), 0ull, StlBinaryPrefixBytes, &context.ioTelemetry);
		bool startsWithSolid = false;
		if (hasPrefix)
		{
			startsWithSolid = (std::memcmp(prefix.data(), "solid ", StlTextProbeBytes) == 0);
		}
		else
		{
			char header[StlTextProbeBytes] = {};
			if (!stlReadExact(context.inner.mainFile, header, 0ull, sizeof(header), &context.ioTelemetry))
				return {};
			startsWithSolid = (std::strncmp(header, "solid ", StlTextProbeBytes) == 0);
		}

		bool binaryBySize = false;
		if (hasPrefix)
		{
			uint32_t triCount = 0u;
			std::memcpy(&triCount, prefix.data() + StlBinaryHeaderBytes, sizeof(triCount));
			binaryTriCountFromDetect = triCount;
			hasBinaryTriCountFromDetect = true;
			const uint64_t expectedSize = StlBinaryPrefixBytes + static_cast<uint64_t>(triCount) * StlTriangleRecordBytes;
			binaryBySize = (expectedSize == filesize);
		}

		if (binaryBySize)
			binary = true;
		else if (!startsWithSolid)
			binary = true;
		else
			binary = false;

		detectMs = std::chrono::duration<double, std::milli>(clock_t::now() - detectStart).count();
	}

	auto geometry = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	geometry->setIndexing(IPolygonGeometryBase::TriangleList());
	hlsl::shapes::AABB<3, hlsl::float32_t> parsedAABB = hlsl::shapes::AABB<3, hlsl::float32_t>::create();
	bool hasParsedAABB = false;
	uint64_t vertexCount = 0ull;

	if (binary)
	{
		parsePath = "binary_fast";
		if (filesize < StlBinaryPrefixBytes)
			return {};

		uint32_t triangleCount32 = binaryTriCountFromDetect;
		if (!hasBinaryTriCountFromDetect)
		{
			if (!stlReadExact(context.inner.mainFile, &triangleCount32, StlBinaryHeaderBytes, sizeof(triangleCount32), &context.ioTelemetry))
				return {};
		}

		triangleCount = triangleCount32;
		const size_t dataSize = static_cast<size_t>(triangleCount) * StlTriangleRecordBytes;
		const size_t expectedSize = StlBinaryPrefixBytes + dataSize;
		if (filesize < expectedSize)
			return {};

		core::vector<uint8_t> payload;
		payload.resize(dataSize);
		const auto ioStart = clock_t::now();
		if (!stlReadWithPolicy(context.inner.mainFile, payload.data(), StlBinaryPrefixBytes, dataSize, ioPlan, &context.ioTelemetry))
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
		auto* posOutFloat = reinterpret_cast<float*>(posOut);
		auto* normalOutFloat = reinterpret_cast<float*>(normalOut);
		const double buildPrepMs = std::chrono::duration<double, std::milli>(clock_t::now() - buildPrepStart).count();
		buildAllocViewsMs += buildPrepMs;
		buildMs += buildPrepMs;

		const auto parseStart = clock_t::now();
		const uint8_t* cursor = payload.data();
		const uint8_t* const end = cursor + payload.size();
		if (end < cursor || static_cast<size_t>(end - cursor) < static_cast<size_t>(triangleCount) * StlTriangleRecordBytes)
			return {};
		for (uint64_t tri = 0ull; tri < triangleCount; ++tri)
		{
			const uint8_t* const triRecord = cursor;
			cursor += StlTriangleRecordBytes;

			float normalData[StlFloatChannelsPerVertex] = {};
			std::memcpy(normalData, triRecord, sizeof(normalData));
			float normalX = normalData[0];
			float normalY = normalData[1];
			float normalZ = normalData[2];

			const size_t base = static_cast<size_t>(tri) * StlVerticesPerTriangle * StlFloatChannelsPerVertex;
			std::memcpy(posOutFloat + base + 0ull, triRecord + 9ull * sizeof(float), sizeof(normalData));
			std::memcpy(posOutFloat + base + 3ull, triRecord + 6ull * sizeof(float), sizeof(normalData));
			std::memcpy(posOutFloat + base + 6ull, triRecord + 3ull * sizeof(float), sizeof(normalData));

			const float vertex0x = posOutFloat[base + 0ull];
			const float vertex0y = posOutFloat[base + 1ull];
			const float vertex0z = posOutFloat[base + 2ull];
			const float vertex1x = posOutFloat[base + 3ull];
			const float vertex1y = posOutFloat[base + 4ull];
			const float vertex1z = posOutFloat[base + 5ull];
			const float vertex2x = posOutFloat[base + 6ull];
			const float vertex2y = posOutFloat[base + 7ull];
			const float vertex2z = posOutFloat[base + 8ull];
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
		const double buildFinalizeMs = std::chrono::duration<double, std::milli>(clock_t::now() - buildFinalizeStart).count();
		buildSetViewsMs += buildFinalizeMs;
		buildMs += buildFinalizeMs;
	}
	else
	{
		parsePath = "ascii_fallback";
		core::vector<uint8_t> asciiPayload;
		asciiPayload.resize(filesize + 1ull);
		const auto ioStart = clock_t::now();
		if (!stlReadWithPolicy(context.inner.mainFile, asciiPayload.data(), 0ull, filesize, ioPlan, &context.ioTelemetry))
			return {};
		ioMs = std::chrono::duration<double, std::milli>(clock_t::now() - ioStart).count();
		asciiPayload[filesize] = 0u;

		const char* cursor = reinterpret_cast<const char*>(asciiPayload.data());
		const char* const end = cursor + filesize;
		core::vector<hlsl::float32_t3> positions;
		core::vector<hlsl::float32_t3> normals;
		std::string_view textToken = {};
		if (!stlReadTextToken(cursor, end, textToken) || textToken != std::string_view("solid"))
			return {};

		const auto parseStart = clock_t::now();
		while (stlReadTextToken(cursor, end, textToken))
		{
			if (textToken == std::string_view("endsolid"))
				break;
			if (textToken != std::string_view("facet"))
			{
				continue;
			}
			if (!stlReadTextToken(cursor, end, textToken) || textToken != std::string_view("normal"))
				return {};

			hlsl::float32_t3 fileNormal = {};
			if (!stlReadTextVec3(cursor, end, fileNormal))
				return {};

			normals.push_back(stlResolveStoredNormal(fileNormal));

			if (!stlReadTextToken(cursor, end, textToken) || textToken != std::string_view("outer"))
				return {};
			if (!stlReadTextToken(cursor, end, textToken) || textToken != std::string_view("loop"))
				return {};

			hlsl::float32_t3 p[3] = {};
			for (uint32_t i = 0u; i < 3u; ++i)
			{
				if (!stlReadTextToken(cursor, end, textToken) || textToken != std::string_view("vertex"))
					return {};
				if (!stlReadTextVec3(cursor, end, p[i]))
					return {};
			}

			stlPushTriangleReversed(p, positions);
			hlsl::float32_t3 faceNormal = stlResolveStoredNormal(fileNormal);
			if (hlsl::dot(faceNormal, faceNormal) <= 0.f)
				faceNormal = stlComputeFaceNormal(p[2u], p[1u], p[0u]);
			normals.push_back(faceNormal);
			normals.push_back(faceNormal);
			normals.push_back(faceNormal);
			stlExtendAABB(parsedAABB, hasParsedAABB, p[2u]);
			stlExtendAABB(parsedAABB, hasParsedAABB, p[1u]);
			stlExtendAABB(parsedAABB, hasParsedAABB, p[0u]);

			if (!stlReadTextToken(cursor, end, textToken) || textToken != std::string_view("endloop"))
				return {};
			if (!stlReadTextToken(cursor, end, textToken) || textToken != std::string_view("endfacet"))
				return {};
		}
		parseMs = std::chrono::duration<double, std::milli>(clock_t::now() - parseStart).count();
		if (positions.empty())
			return {};

		triangleCount = positions.size() / StlVerticesPerTriangle;
		vertexCount = positions.size();

		const auto buildStart = clock_t::now();
		const auto allocStart = clock_t::now();
		auto posView = stlCreateAdoptedFloat3View(std::move(positions));
		auto normalView = stlCreateAdoptedFloat3View(std::move(normals));
		if (!posView || !normalView)
			return {};
		buildAllocViewsMs += std::chrono::duration<double, std::milli>(clock_t::now() - allocStart).count();

		const auto setStart = clock_t::now();
		geometry->setPositionView(std::move(posView));
		geometry->setNormalView(std::move(normalView));
		buildSetViewsMs += std::chrono::duration<double, std::milli>(clock_t::now() - setStart).count();
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

	buildMiscMs = std::max(0.0, buildMs - (buildAllocViewsMs + buildSetViewsMs));

	const auto totalMs = std::chrono::duration<double, std::milli>(clock_t::now() - totalStart).count();
	const uint64_t ioMinRead = context.ioTelemetry.getMinOrZero();
	const uint64_t ioAvgRead = context.ioTelemetry.getAvgOrZero();
	if (
		static_cast<uint64_t>(filesize) > (1ull << 20) &&
		(
			ioAvgRead < 1024ull ||
			(ioMinRead < 64ull && context.ioTelemetry.callCount > 1024ull)
		)
	)
	{
		_params.logger.log(
			"STL loader tiny-io guard: file=%s reads=%llu min=%llu avg=%llu",
			system::ILogger::ELL_WARNING,
			_file->getFileName().string().c_str(),
			static_cast<unsigned long long>(context.ioTelemetry.callCount),
			static_cast<unsigned long long>(ioMinRead),
			static_cast<unsigned long long>(ioAvgRead));
	}
	_params.logger.log(
		"STL loader perf: file=%s total=%.3f ms detect=%.3f io=%.3f parse=%.3f build=%.3f build_alloc_views=%.3f build_set_views=%.3f build_misc=%.3f hash=%.3f aabb=%.3f binary=%d parse_path=%s triangles=%llu vertices=%llu io_reads=%llu io_min_read=%llu io_avg_read=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
		system::ILogger::ELL_PERFORMANCE,
		_file->getFileName().string().c_str(),
		totalMs,
		detectMs,
		ioMs,
		parseMs,
		buildMs,
		buildAllocViewsMs,
		buildSetViewsMs,
		buildMiscMs,
		hashMs,
		aabbMs,
		binary ? 1 : 0,
		parsePath,
		static_cast<unsigned long long>(triangleCount),
		static_cast<unsigned long long>(vertexCount),
		static_cast<unsigned long long>(context.ioTelemetry.callCount),
		static_cast<unsigned long long>(ioMinRead),
		static_cast<unsigned long long>(ioAvgRead),
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
