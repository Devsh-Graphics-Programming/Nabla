// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CSTLMeshFileLoader.h"

#ifdef _NBL_COMPILE_WITH_STL_LOADER_

#include "nbl/asset/interchange/SLoaderRuntimeTuning.h"
#include "nbl/asset/asset.h"
#include "nbl/asset/metadata/CSTLMetadata.h"
#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
#include "nbl/core/hash/blake.h"
#include "nbl/system/IFile.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <execution>
#include <fast_float/fast_float.h>
#include <limits>
#include <numeric>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>
#include <ranges>

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

template<typename Fn>
void stlRunParallelWorkers(const size_t workerCount, Fn&& fn)
{
	if (workerCount <= 1ull)
	{
		fn(0ull);
		return;
	}
	auto workerIds = std::views::iota(size_t{0ull}, workerCount);
	std::for_each(std::execution::par, workerIds.begin(), workerIds.end(), [&fn](const size_t workerIx)
	{
		fn(workerIx);
	});
}

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

class CStlSplitBlockMemoryResource final : public core::refctd_memory_resource
{
	public:
		inline CStlSplitBlockMemoryResource(
			core::smart_refctd_ptr<core::refctd_memory_resource>&& upstream,
			void* block,
			const size_t blockBytes,
			const size_t alignment
		) : m_upstream(std::move(upstream)), m_block(block), m_blockBytes(blockBytes), m_alignment(alignment)
		{
		}

		inline void* allocate(std::size_t bytes, std::size_t alignment) override
		{
			assert(false);
			return nullptr;
		}

		inline void deallocate(void* p, std::size_t bytes, std::size_t alignment) override
		{
			(void)alignment;
			const auto* const begin = reinterpret_cast<const uint8_t*>(m_block);
			const auto* const end = begin + m_blockBytes;
			const auto* const ptr = reinterpret_cast<const uint8_t*>(p);
			assert(ptr >= begin && ptr <= end);
			assert(ptr + bytes <= end);
		}

	protected:
		inline ~CStlSplitBlockMemoryResource() override
		{
			if (m_upstream && m_block)
				m_upstream->deallocate(m_block, m_blockBytes, m_alignment);
		}

	private:
		core::smart_refctd_ptr<core::refctd_memory_resource> m_upstream;
		void* m_block = nullptr;
		size_t m_blockBytes = 0ull;
		size_t m_alignment = 1ull;
};

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

void stlRecomputeContentHashesParallel(ICPUPolygonGeometry* geometry, const SFileIOPolicy& ioPolicy)
{
	if (!geometry)
		return;

	core::vector<core::smart_refctd_ptr<ICPUBuffer>> buffers;
	auto appendViewBuffer = [&buffers](const IGeometry<ICPUBuffer>::SDataView& view) -> void
	{
		if (!view || !view.src.buffer)
			return;
		for (const auto& existing : buffers)
		{
			if (existing.get() == view.src.buffer.get())
				return;
		}
		buffers.push_back(core::smart_refctd_ptr<ICPUBuffer>(view.src.buffer));
	};

	appendViewBuffer(geometry->getPositionView());
	appendViewBuffer(geometry->getIndexView());
	appendViewBuffer(geometry->getNormalView());
	for (const auto& view : *geometry->getAuxAttributeViews())
		appendViewBuffer(view);
	for (const auto& view : *geometry->getJointWeightViews())
	{
		appendViewBuffer(view.indices);
		appendViewBuffer(view.weights);
	}
	if (auto jointOBB = geometry->getJointOBBView(); jointOBB)
		appendViewBuffer(*jointOBB);

	if (buffers.empty())
		return;

	uint64_t totalBytes = 0ull;
	for (const auto& buffer : buffers)
		totalBytes += static_cast<uint64_t>(buffer->getSize());

	const size_t hw = resolveLoaderHardwareThreads();
	const uint8_t* hashSampleData = nullptr;
	uint64_t hashSampleBytes = 0ull;
	for (const auto& buffer : buffers)
	{
		const auto* ptr = reinterpret_cast<const uint8_t*>(buffer->getPointer());
		if (!ptr)
			continue;
		hashSampleData = ptr;
		hashSampleBytes = std::min<uint64_t>(static_cast<uint64_t>(buffer->getSize()), 128ull << 10);
		if (hashSampleBytes > 0ull)
			break;
	}
	SLoaderRuntimeTuningRequest tuningRequest = {};
	tuningRequest.inputBytes = totalBytes;
	tuningRequest.totalWorkUnits = buffers.size();
	tuningRequest.minBytesPerWorker = std::max<uint64_t>(1ull, buffers.empty() ? 1ull : loaderRuntimeCeilDiv(totalBytes, static_cast<uint64_t>(buffers.size())));
	tuningRequest.hardwareThreads = static_cast<uint32_t>(hw);
	tuningRequest.hardMaxWorkers = static_cast<uint32_t>(std::min(hw, buffers.size()));
	tuningRequest.targetChunksPerWorker = 1u;
	tuningRequest.sampleData = hashSampleData;
	tuningRequest.sampleBytes = hashSampleBytes;
	const auto tuning = tuneLoaderRuntime(ioPolicy, tuningRequest);
	const size_t workerCount = std::min(tuning.workerCount, buffers.size());
	if (workerCount > 1ull)
	{
		stlRunParallelWorkers(workerCount, [&buffers, workerCount](const size_t workerIx)
		{
			const size_t beginIx = (buffers.size() * workerIx) / workerCount;
			const size_t endIx = (buffers.size() * (workerIx + 1ull)) / workerCount;
			for (size_t i = beginIx; i < endIx; ++i)
			{
				auto& buffer = buffers[i];
				if (buffer->getContentHash() != IPreHashed::INVALID_HASH)
					continue;
				buffer->setContentHash(buffer->computeContentHash());
			}
		});
		return;
	}

	for (auto& buffer : buffers)
	{
		if (buffer->getContentHash() != IPreHashed::INVALID_HASH)
			continue;
		buffer->setContentHash(buffer->computeContentHash());
	}
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
	const bool computeContentHashes = (_params.loaderFlags & IAssetLoader::ELPF_COMPUTE_CONTENT_HASHES) != 0;
	bool contentHashesAssigned = false;

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

	core::vector<uint8_t> wholeFilePayload;
	const uint8_t* wholeFileData = nullptr;
	bool wholeFileDataIsMapped = false;
	if (ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile)
	{
		const auto* constFile = static_cast<const system::IFile*>(context.inner.mainFile);
		const auto* mapped = reinterpret_cast<const uint8_t*>(constFile->getMappedPointer());
		if (mapped)
		{
			wholeFileData = mapped;
			wholeFileDataIsMapped = true;
			context.ioTelemetry.account(filesize);
		}
		else
		{
			const auto ioStart = clock_t::now();
			wholeFilePayload.resize(filesize + 1ull);
			if (!stlReadExact(context.inner.mainFile, wholeFilePayload.data(), 0ull, filesize, &context.ioTelemetry))
				return {};
			wholeFilePayload[filesize] = 0u;
			wholeFileData = wholeFilePayload.data();
			ioMs = std::chrono::duration<double, std::milli>(clock_t::now() - ioStart).count();
		}
	}

	bool binary = false;
	bool hasBinaryTriCountFromDetect = false;
	uint32_t binaryTriCountFromDetect = 0u;
	{
		const auto detectStart = clock_t::now();
		std::array<uint8_t, StlBinaryPrefixBytes> prefix = {};
		bool hasPrefix = false;
		if (wholeFileData && filesize >= StlBinaryPrefixBytes)
		{
			std::memcpy(prefix.data(), wholeFileData, StlBinaryPrefixBytes);
			hasPrefix = true;
		}
		else
		{
			hasPrefix = filesize >= StlBinaryPrefixBytes && stlReadExact(context.inner.mainFile, prefix.data(), 0ull, StlBinaryPrefixBytes, &context.ioTelemetry);
		}
		bool startsWithSolid = false;
		if (hasPrefix)
		{
			startsWithSolid = (std::memcmp(prefix.data(), "solid ", StlTextProbeBytes) == 0);
		}
		else
		{
			char header[StlTextProbeBytes] = {};
			if (wholeFileData)
				std::memcpy(header, wholeFileData, sizeof(header));
			else if (!stlReadExact(context.inner.mainFile, header, 0ull, sizeof(header), &context.ioTelemetry))
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

	if (!binary && wholeFileDataIsMapped)
	{
		wholeFilePayload.resize(filesize + 1ull);
		std::memcpy(wholeFilePayload.data(), wholeFileData, filesize);
		wholeFilePayload[filesize] = 0u;
		wholeFileData = wholeFilePayload.data();
		wholeFileDataIsMapped = false;
	}

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

		const uint8_t* payloadData = nullptr;
		if (wholeFileData)
		{
			payloadData = wholeFileData + StlBinaryPrefixBytes;
		}
		else
		{
			core::vector<uint8_t> payload;
			payload.resize(dataSize);
			const auto ioStart = clock_t::now();
			if (!stlReadWithPolicy(context.inner.mainFile, payload.data(), StlBinaryPrefixBytes, dataSize, ioPlan, &context.ioTelemetry))
				return {};
			ioMs = std::chrono::duration<double, std::milli>(clock_t::now() - ioStart).count();
			wholeFilePayload = std::move(payload);
			payloadData = wholeFilePayload.data();
		}

		vertexCount = triangleCount * StlVerticesPerTriangle;
		const auto buildPrepStart = clock_t::now();
		const size_t vertexCountSizeT = static_cast<size_t>(vertexCount);
		if (vertexCountSizeT > (std::numeric_limits<size_t>::max() / sizeof(hlsl::float32_t3)))
			return {};
		const size_t viewByteSize = vertexCountSizeT * sizeof(hlsl::float32_t3);
		if (viewByteSize > (std::numeric_limits<size_t>::max() - viewByteSize))
			return {};
		const size_t blockBytes = viewByteSize * 2ull;
		auto upstream = core::getDefaultMemoryResource();
		if (!upstream)
			return {};
		void* block = upstream->allocate(blockBytes, alignof(float));
		if (!block)
			return {};
		auto blockResource = core::make_smart_refctd_ptr<CStlSplitBlockMemoryResource>(
			core::smart_refctd_ptr<core::refctd_memory_resource>(std::move(upstream)),
			block,
			blockBytes,
			alignof(float));
		auto posBuffer = ICPUBuffer::create({
			{ viewByteSize },
			block,
			core::smart_refctd_ptr<core::refctd_memory_resource>(blockResource),
			alignof(float)
		}, core::adopt_memory);
		auto normalBuffer = ICPUBuffer::create({
			{ viewByteSize },
			reinterpret_cast<uint8_t*>(block) + viewByteSize,
			core::smart_refctd_ptr<core::refctd_memory_resource>(blockResource),
			alignof(float)
		}, core::adopt_memory);
		if (!posBuffer || !normalBuffer)
			return {};
		ICPUPolygonGeometry::SDataView posView = {};
		posView.composed = {
			.stride = sizeof(hlsl::float32_t3),
			.format = EF_R32G32B32_SFLOAT,
			.rangeFormat = IGeometryBase::getMatchingAABBFormat(EF_R32G32B32_SFLOAT)
		};
		posView.src = {
			.offset = 0ull,
			.size = viewByteSize,
			.buffer = std::move(posBuffer)
		};
		ICPUPolygonGeometry::SDataView normalView = {};
		normalView.composed = {
			.stride = sizeof(hlsl::float32_t3),
			.format = EF_R32G32B32_SFLOAT,
			.rangeFormat = IGeometryBase::getMatchingAABBFormat(EF_R32G32B32_SFLOAT)
		};
		normalView.src = {
			.offset = 0ull,
			.size = viewByteSize,
			.buffer = std::move(normalBuffer)
		};
		auto* posOutFloat = reinterpret_cast<float*>(posView.getPointer());
		auto* normalOutFloat = reinterpret_cast<float*>(normalView.getPointer());
		if (!posOutFloat || !normalOutFloat)
			return {};
		const double buildPrepMs = std::chrono::duration<double, std::milli>(clock_t::now() - buildPrepStart).count();
		buildAllocViewsMs += buildPrepMs;
		buildMs += buildPrepMs;

		const auto parseStart = clock_t::now();
		const uint8_t* cursor = payloadData;
		const uint8_t* const end = cursor + dataSize;
		if (end < cursor || static_cast<size_t>(end - cursor) < static_cast<size_t>(triangleCount) * StlTriangleRecordBytes)
			return {};
		const size_t hw = resolveLoaderHardwareThreads();
		SLoaderRuntimeTuningRequest parseTuningRequest = {};
		parseTuningRequest.inputBytes = dataSize;
		parseTuningRequest.totalWorkUnits = triangleCount;
		parseTuningRequest.minBytesPerWorker = StlTriangleRecordBytes;
		parseTuningRequest.hardwareThreads = static_cast<uint32_t>(hw);
		parseTuningRequest.hardMaxWorkers = static_cast<uint32_t>(hw);
		parseTuningRequest.targetChunksPerWorker = 4u;
		parseTuningRequest.minChunkWorkUnits = 1ull;
		parseTuningRequest.maxChunkWorkUnits = std::max<uint64_t>(1ull, triangleCount);
		parseTuningRequest.sampleData = payloadData;
		parseTuningRequest.sampleBytes = std::min<uint64_t>(dataSize, 128ull << 10);
		const auto parseTuning = tuneLoaderRuntime(_params.ioPolicy, parseTuningRequest);
		const size_t workerCount = std::max<size_t>(1ull, std::min(parseTuning.workerCount, static_cast<size_t>(std::max<uint64_t>(1ull, triangleCount))));
		static constexpr bool ComputeAABBInParse = true;
		struct SThreadAABB
		{
			bool has = false;
			float minX = 0.f;
			float minY = 0.f;
			float minZ = 0.f;
			float maxX = 0.f;
			float maxY = 0.f;
			float maxZ = 0.f;
		};
		std::vector<SThreadAABB> threadAABBs(ComputeAABBInParse ? workerCount : 0ull);
		const uint64_t parseChunkTriangles = std::max<uint64_t>(1ull, parseTuning.chunkWorkUnits);
		const size_t parseChunkCount = static_cast<size_t>(loaderRuntimeCeilDiv(triangleCount, parseChunkTriangles));
		const bool hashInParsePipeline = computeContentHashes;
		std::vector<uint8_t> hashChunkReady(hashInParsePipeline ? parseChunkCount : 0ull, 0u);
		double positionHashPipelineMs = 0.0;
		double normalHashPipelineMs = 0.0;
		std::atomic_bool hashPipelineOk = true;
		core::blake3_hash_t parsedPositionHash = static_cast<core::blake3_hash_t>(core::blake3_hasher{});
		core::blake3_hash_t parsedNormalHash = static_cast<core::blake3_hash_t>(core::blake3_hasher{});
		auto parseRange = [&](const uint64_t beginTri, const uint64_t endTri, SThreadAABB& localAABB) -> void
		{
			const uint8_t* localCursor = payloadData + beginTri * StlTriangleRecordBytes;
			float* posCursor = posOutFloat + beginTri * StlVerticesPerTriangle * StlFloatChannelsPerVertex;
			float* normalCursor = normalOutFloat + beginTri * StlVerticesPerTriangle * StlFloatChannelsPerVertex;
			for (uint64_t tri = beginTri; tri < endTri; ++tri)
			{
				const uint8_t* const triRecord = localCursor;
				localCursor += StlTriangleRecordBytes;
				float triValues[StlTriangleFloatCount];
				std::memcpy(triValues, triRecord, sizeof(triValues));

				float normalX = triValues[0ull];
				float normalY = triValues[1ull];
				float normalZ = triValues[2ull];

				const float vertex0x = triValues[9ull];
				const float vertex0y = triValues[10ull];
				const float vertex0z = triValues[11ull];
				const float vertex1x = triValues[6ull];
				const float vertex1y = triValues[7ull];
				const float vertex1z = triValues[8ull];
				const float vertex2x = triValues[3ull];
				const float vertex2y = triValues[4ull];
				const float vertex2z = triValues[5ull];

				posCursor[0ull] = vertex0x;
				posCursor[1ull] = vertex0y;
				posCursor[2ull] = vertex0z;
				posCursor[3ull] = vertex1x;
				posCursor[4ull] = vertex1y;
				posCursor[5ull] = vertex1z;
				posCursor[6ull] = vertex2x;
				posCursor[7ull] = vertex2y;
				posCursor[8ull] = vertex2z;
				if constexpr (ComputeAABBInParse)
				{
					if (!localAABB.has)
					{
						localAABB.has = true;
						localAABB.minX = vertex0x;
						localAABB.minY = vertex0y;
						localAABB.minZ = vertex0z;
						localAABB.maxX = vertex0x;
						localAABB.maxY = vertex0y;
						localAABB.maxZ = vertex0z;
					}
					if (vertex0x < localAABB.minX) localAABB.minX = vertex0x;
					if (vertex0y < localAABB.minY) localAABB.minY = vertex0y;
					if (vertex0z < localAABB.minZ) localAABB.minZ = vertex0z;
					if (vertex0x > localAABB.maxX) localAABB.maxX = vertex0x;
					if (vertex0y > localAABB.maxY) localAABB.maxY = vertex0y;
					if (vertex0z > localAABB.maxZ) localAABB.maxZ = vertex0z;
					if (vertex1x < localAABB.minX) localAABB.minX = vertex1x;
					if (vertex1y < localAABB.minY) localAABB.minY = vertex1y;
					if (vertex1z < localAABB.minZ) localAABB.minZ = vertex1z;
					if (vertex1x > localAABB.maxX) localAABB.maxX = vertex1x;
					if (vertex1y > localAABB.maxY) localAABB.maxY = vertex1y;
					if (vertex1z > localAABB.maxZ) localAABB.maxZ = vertex1z;
					if (vertex2x < localAABB.minX) localAABB.minX = vertex2x;
					if (vertex2y < localAABB.minY) localAABB.minY = vertex2y;
					if (vertex2z < localAABB.minZ) localAABB.minZ = vertex2z;
					if (vertex2x > localAABB.maxX) localAABB.maxX = vertex2x;
					if (vertex2y > localAABB.maxY) localAABB.maxY = vertex2y;
					if (vertex2z > localAABB.maxZ) localAABB.maxZ = vertex2z;
				}
				if (normalX == 0.f && normalY == 0.f && normalZ == 0.f)
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
				normalCursor[0ull] = normalX;
				normalCursor[1ull] = normalY;
				normalCursor[2ull] = normalZ;
				normalCursor[3ull] = normalX;
				normalCursor[4ull] = normalY;
				normalCursor[5ull] = normalZ;
				normalCursor[6ull] = normalX;
				normalCursor[7ull] = normalY;
				normalCursor[8ull] = normalZ;
				posCursor += StlVerticesPerTriangle * StlFloatChannelsPerVertex;
				normalCursor += StlVerticesPerTriangle * StlFloatChannelsPerVertex;
			}
		};
		std::jthread positionHashThread;
		std::jthread normalHashThread;
		if (hashInParsePipeline)
		{
			positionHashThread = std::jthread([&]()
			{
				try
				{
					core::blake3_hasher positionHasher;
					const auto hashThreadStart = clock_t::now();
					size_t chunkIx = 0ull;
					while (chunkIx < parseChunkCount)
					{
						auto ready = std::atomic_ref<uint8_t>(hashChunkReady[chunkIx]);
						while (ready.load(std::memory_order_acquire) == 0u)
							ready.wait(0u, std::memory_order_acquire);

						size_t runEnd = chunkIx + 1ull;
						while (runEnd < parseChunkCount)
						{
							const auto runReady = std::atomic_ref<uint8_t>(hashChunkReady[runEnd]).load(std::memory_order_acquire);
							if (runReady == 0u)
								break;
							++runEnd;
						}

						const uint64_t begin = static_cast<uint64_t>(chunkIx) * parseChunkTriangles;
						const uint64_t endTri = std::min<uint64_t>(static_cast<uint64_t>(runEnd) * parseChunkTriangles, triangleCount);
						const size_t runTriangles = static_cast<size_t>(endTri - begin);
						const size_t runBytes = runTriangles * StlVerticesPerTriangle * StlFloatChannelsPerVertex * sizeof(float);
						positionHasher.update(posOutFloat + begin * StlVerticesPerTriangle * StlFloatChannelsPerVertex, runBytes);
						chunkIx = runEnd;
					}
					positionHashPipelineMs = std::chrono::duration<double, std::milli>(clock_t::now() - hashThreadStart).count();
					parsedPositionHash = static_cast<core::blake3_hash_t>(positionHasher);
				}
				catch (...)
				{
					hashPipelineOk.store(false, std::memory_order_relaxed);
				}
			});
			normalHashThread = std::jthread([&]()
			{
				try
				{
					core::blake3_hasher normalHasher;
					const auto hashThreadStart = clock_t::now();
					size_t chunkIx = 0ull;
					while (chunkIx < parseChunkCount)
					{
						auto ready = std::atomic_ref<uint8_t>(hashChunkReady[chunkIx]);
						while (ready.load(std::memory_order_acquire) == 0u)
							ready.wait(0u, std::memory_order_acquire);

						size_t runEnd = chunkIx + 1ull;
						while (runEnd < parseChunkCount)
						{
							const auto runReady = std::atomic_ref<uint8_t>(hashChunkReady[runEnd]).load(std::memory_order_acquire);
							if (runReady == 0u)
								break;
							++runEnd;
						}

						const uint64_t begin = static_cast<uint64_t>(chunkIx) * parseChunkTriangles;
						const uint64_t endTri = std::min<uint64_t>(static_cast<uint64_t>(runEnd) * parseChunkTriangles, triangleCount);
						const size_t runTriangles = static_cast<size_t>(endTri - begin);
						const size_t runBytes = runTriangles * StlVerticesPerTriangle * StlFloatChannelsPerVertex * sizeof(float);
						normalHasher.update(normalOutFloat + begin * StlVerticesPerTriangle * StlFloatChannelsPerVertex, runBytes);
						chunkIx = runEnd;
					}
					normalHashPipelineMs = std::chrono::duration<double, std::milli>(clock_t::now() - hashThreadStart).count();
					parsedNormalHash = static_cast<core::blake3_hash_t>(normalHasher);
				}
				catch (...)
				{
					hashPipelineOk.store(false, std::memory_order_relaxed);
				}
			});
		}
		std::atomic_size_t nextChunkIx = 0ull;
		auto parseWorker = [&](const size_t workerIx) -> void
		{
			SThreadAABB localAABB = {};
			while (true)
			{
				const size_t chunkIx = nextChunkIx.fetch_add(1ull, std::memory_order_relaxed);
				if (chunkIx >= parseChunkCount)
					break;
				const uint64_t begin = static_cast<uint64_t>(chunkIx) * parseChunkTriangles;
				const uint64_t endTri = std::min<uint64_t>(begin + parseChunkTriangles, triangleCount);
				parseRange(begin, endTri, localAABB);
				if (hashInParsePipeline)
				{
					auto ready = std::atomic_ref<uint8_t>(hashChunkReady[chunkIx]);
					ready.store(1u, std::memory_order_release);
					ready.notify_all();
				}
			}
			if constexpr (ComputeAABBInParse)
				threadAABBs[workerIx] = localAABB;
		};

		if (workerCount > 1ull)
		{
			stlRunParallelWorkers(workerCount, parseWorker);
		}
		else
		{
			parseWorker(0ull);
		}
		if (positionHashThread.joinable())
			positionHashThread.join();
		if (normalHashThread.joinable())
			normalHashThread.join();
		if (hashInParsePipeline)
		{
			if (!hashPipelineOk.load(std::memory_order_relaxed))
				return {};
			hashMs += positionHashPipelineMs + normalHashPipelineMs;
			posView.src.buffer->setContentHash(parsedPositionHash);
			normalView.src.buffer->setContentHash(parsedNormalHash);
			contentHashesAssigned = true;
		}
		if constexpr (ComputeAABBInParse)
		{
			for (const auto& localAABB : threadAABBs)
			{
				if (!localAABB.has)
					continue;
				if (!hasParsedAABB)
				{
					hasParsedAABB = true;
					parsedAABB = hlsl::shapes::AABB<3, hlsl::float32_t>::create();
					parsedAABB.minVx.x = localAABB.minX;
					parsedAABB.minVx.y = localAABB.minY;
					parsedAABB.minVx.z = localAABB.minZ;
					parsedAABB.maxVx.x = localAABB.maxX;
					parsedAABB.maxVx.y = localAABB.maxY;
					parsedAABB.maxVx.z = localAABB.maxZ;
					continue;
				}
				if (localAABB.minX < parsedAABB.minVx.x) parsedAABB.minVx.x = localAABB.minX;
				if (localAABB.minY < parsedAABB.minVx.y) parsedAABB.minVx.y = localAABB.minY;
				if (localAABB.minZ < parsedAABB.minVx.z) parsedAABB.minVx.z = localAABB.minZ;
				if (localAABB.maxX > parsedAABB.maxVx.x) parsedAABB.maxVx.x = localAABB.maxX;
				if (localAABB.maxY > parsedAABB.maxVx.y) parsedAABB.maxVx.y = localAABB.maxY;
				if (localAABB.maxZ > parsedAABB.maxVx.z) parsedAABB.maxVx.z = localAABB.maxZ;
			}
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
		if (!wholeFileData)
		{
			const auto ioStart = clock_t::now();
			wholeFilePayload.resize(filesize + 1ull);
			if (!stlReadWithPolicy(context.inner.mainFile, wholeFilePayload.data(), 0ull, filesize, ioPlan, &context.ioTelemetry))
				return {};
			ioMs = std::chrono::duration<double, std::milli>(clock_t::now() - ioStart).count();
			wholeFilePayload[filesize] = 0u;
			wholeFileData = wholeFilePayload.data();
		}

		const char* cursor = reinterpret_cast<const char*>(wholeFileData);
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

	if (computeContentHashes && !contentHashesAssigned)
	{
		const auto hashStart = clock_t::now();
		stlRecomputeContentHashesParallel(geometry.get(), _params.ioPolicy);
		hashMs += std::chrono::duration<double, std::milli>(clock_t::now() - hashStart).count();
	}

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

	const size_t fileSize = _file->getSize();
	if (fileSize < StlBinaryPrefixBytes)
	{
		char header[StlTextProbeBytes] = {};
		if (!stlReadExact(_file, header, 0ull, sizeof(header)))
			return false;
		return std::strncmp(header, "solid ", StlTextProbeBytes) == 0;
	}

	std::array<uint8_t, StlBinaryPrefixBytes> prefix = {};
	if (!stlReadExact(_file, prefix.data(), 0ull, prefix.size()))
		return false;

	uint32_t triangleCount = 0u;
	std::memcpy(&triangleCount, prefix.data() + StlBinaryHeaderBytes, sizeof(triangleCount));
	if (std::memcmp(prefix.data(), "solid ", StlTextProbeBytes) == 0)
		return true;

	return fileSize == (StlTriangleRecordBytes * triangleCount + StlBinaryPrefixBytes);
}

}

#endif // _NBL_COMPILE_WITH_STL_LOADER_


