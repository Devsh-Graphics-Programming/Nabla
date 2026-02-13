// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CSTLMeshFileLoader.h"

#ifdef _NBL_COMPILE_WITH_STL_LOADER_

#include "nbl/asset/interchange/SGeometryContentHashCommon.h"
#include "nbl/asset/interchange/SInterchangeIOCommon.h"
#include "nbl/asset/interchange/SLoaderRuntimeTuning.h"
#include "nbl/asset/asset.h"
#include "nbl/asset/metadata/CSTLMetadata.h"
#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
#include "nbl/core/hash/blake.h"
#include "nbl/system/IFile.h"

#include <fast_float/fast_float.h>

namespace nbl::asset
{

struct SSTLContext
{
	IAssetLoader::SAssetLoadContext inner;
	SFileReadTelemetry ioTelemetry = {};
	static constexpr size_t TextProbeBytes = 6ull;
	static constexpr size_t BinaryHeaderBytes = 80ull;
	static constexpr size_t TriangleCountBytes = sizeof(uint32_t);
	static constexpr size_t BinaryPrefixBytes = BinaryHeaderBytes + TriangleCountBytes;
	static constexpr size_t TriangleFloatCount = 12ull;
	static constexpr size_t TriangleFloatBytes = sizeof(float) * TriangleFloatCount;
	static constexpr size_t TriangleAttributeBytes = sizeof(uint16_t);
	static constexpr size_t TriangleRecordBytes = TriangleFloatBytes + TriangleAttributeBytes;
	static constexpr size_t VerticesPerTriangle = 3ull;
	static constexpr size_t FloatChannelsPerVertex = 3ull;
};

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
	return stlReadTextFloat(ptr, end, outVec.x) && stlReadTextFloat(ptr, end, outVec.y) && stlReadTextFloat(ptr, end, outVec.z);
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

		inline void* allocate(std::size_t, std::size_t) override
		{
			assert(false);
			return nullptr;
		}

		inline void deallocate(void* p, std::size_t bytes, std::size_t) override
		{
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
	auto buffer = ICPUBuffer::create({ { byteCount }, payloadPtr, core::smart_refctd_ptr<core::refctd_memory_resource>(std::move(backer)), alignof(hlsl::float32_t3) }, core::adopt_memory);
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

CSTLMeshFileLoader::CSTLMeshFileLoader(asset::IAssetManager*)
{
}

const char** CSTLMeshFileLoader::getAssociatedFileExtensions() const
{
	static const char* ext[] = { "stl", nullptr };
	return ext;
}

SAssetBundle CSTLMeshFileLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride*, uint32_t)
{
	if (!_file)
		return {};

	uint64_t triangleCount = 0u;
	const char* parsePath = "unknown";
	const bool computeContentHashes = (_params.loaderFlags & IAssetLoader::ELPF_DONT_COMPUTE_CONTENT_HASHES) == 0;
	bool contentHashesAssigned = false;

	SSTLContext context = { asset::IAssetLoader::SAssetLoadContext{ _params,_file },0ull };

	const size_t filesize = context.inner.mainFile->getSize();
	if (filesize < SSTLContext::TextProbeBytes)
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
			wholeFilePayload.resize(filesize + 1ull);
			if (!readFileExact(context.inner.mainFile, wholeFilePayload.data(), 0ull, filesize, &context.ioTelemetry))
				return {};
			wholeFilePayload[filesize] = 0u;
			wholeFileData = wholeFilePayload.data();
		}
	}

	bool binary = false;
	bool hasBinaryTriCountFromDetect = false;
	uint32_t binaryTriCountFromDetect = 0u;
	{
		std::array<uint8_t, SSTLContext::BinaryPrefixBytes> prefix = {};
		bool hasPrefix = false;
		if (wholeFileData && filesize >= SSTLContext::BinaryPrefixBytes)
		{
			std::memcpy(prefix.data(), wholeFileData, SSTLContext::BinaryPrefixBytes);
			hasPrefix = true;
		}
		else
		{
			hasPrefix = filesize >= SSTLContext::BinaryPrefixBytes && readFileExact(context.inner.mainFile, prefix.data(), 0ull, SSTLContext::BinaryPrefixBytes, &context.ioTelemetry);
		}
		bool startsWithSolid = false;
		if (hasPrefix)
		{
			startsWithSolid = (std::memcmp(prefix.data(), "solid ", SSTLContext::TextProbeBytes) == 0);
		}
		else
		{
			char header[SSTLContext::TextProbeBytes] = {};
			if (wholeFileData)
				std::memcpy(header, wholeFileData, sizeof(header));
			else if (!readFileExact(context.inner.mainFile, header, 0ull, sizeof(header), &context.ioTelemetry))
				return {};
			startsWithSolid = (std::strncmp(header, "solid ", SSTLContext::TextProbeBytes) == 0);
		}

		bool binaryBySize = false;
		if (hasPrefix)
		{
			uint32_t triCount = 0u;
			std::memcpy(&triCount, prefix.data() + SSTLContext::BinaryHeaderBytes, sizeof(triCount));
			binaryTriCountFromDetect = triCount;
			hasBinaryTriCountFromDetect = true;
			const uint64_t expectedSize = SSTLContext::BinaryPrefixBytes + static_cast<uint64_t>(triCount) * SSTLContext::TriangleRecordBytes;
			binaryBySize = (expectedSize == filesize);
		}

		if (binaryBySize)
			binary = true;
		else if (!startsWithSolid)
			binary = true;
		else
			binary = false;

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
		if (filesize < SSTLContext::BinaryPrefixBytes)
			return {};

		uint32_t triangleCount32 = binaryTriCountFromDetect;
		if (!hasBinaryTriCountFromDetect)
		{
			if (!readFileExact(context.inner.mainFile, &triangleCount32, SSTLContext::BinaryHeaderBytes, sizeof(triangleCount32), &context.ioTelemetry))
				return {};
		}

		triangleCount = triangleCount32;
		const size_t dataSize = static_cast<size_t>(triangleCount) * SSTLContext::TriangleRecordBytes;
		const size_t expectedSize = SSTLContext::BinaryPrefixBytes + dataSize;
		if (filesize < expectedSize)
			return {};

		const uint8_t* payloadData = nullptr;
		if (wholeFileData)
		{
			payloadData = wholeFileData + SSTLContext::BinaryPrefixBytes;
		}
		else
		{
			core::vector<uint8_t> payload;
			payload.resize(dataSize);
			if (!readFileWithPolicy(context.inner.mainFile, payload.data(), SSTLContext::BinaryPrefixBytes, dataSize, ioPlan, &context.ioTelemetry))
				return {};
			wholeFilePayload = std::move(payload);
			payloadData = wholeFilePayload.data();
		}

		vertexCount = triangleCount * SSTLContext::VerticesPerTriangle;
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
		auto posBuffer = ICPUBuffer::create({ { viewByteSize },block,core::smart_refctd_ptr<core::refctd_memory_resource>(blockResource),alignof(float) }, core::adopt_memory);
		auto normalBuffer = ICPUBuffer::create({ { viewByteSize },reinterpret_cast<uint8_t*>(block) + viewByteSize,core::smart_refctd_ptr<core::refctd_memory_resource>(blockResource),alignof(float) }, core::adopt_memory);
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

		const uint8_t* cursor = payloadData;
		const uint8_t* const end = cursor + dataSize;
		if (end < cursor || static_cast<size_t>(end - cursor) < static_cast<size_t>(triangleCount) * SSTLContext::TriangleRecordBytes)
			return {};
		const size_t hw = resolveLoaderHardwareThreads();
		SLoaderRuntimeTuningRequest parseTuningRequest = {};
		parseTuningRequest.inputBytes = dataSize;
		parseTuningRequest.totalWorkUnits = triangleCount;
		parseTuningRequest.minBytesPerWorker = SSTLContext::TriangleRecordBytes;
		parseTuningRequest.hardwareThreads = static_cast<uint32_t>(hw);
		parseTuningRequest.hardMaxWorkers = static_cast<uint32_t>(std::max<size_t>(1ull, hw > 2ull ? (hw - 2ull) : hw));
		parseTuningRequest.targetChunksPerWorker = 2u;
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
		std::atomic_bool hashPipelineOk = true;
		core::blake3_hash_t parsedPositionHash = static_cast<core::blake3_hash_t>(core::blake3_hasher{});
		core::blake3_hash_t parsedNormalHash = static_cast<core::blake3_hash_t>(core::blake3_hasher{});
		auto parseRange = [&](const uint64_t beginTri, const uint64_t endTri, SThreadAABB& localAABB) -> void
		{
			const uint8_t* localCursor = payloadData + beginTri * SSTLContext::TriangleRecordBytes;
			float* posCursor = posOutFloat + beginTri * SSTLContext::VerticesPerTriangle * SSTLContext::FloatChannelsPerVertex;
			float* normalCursor = normalOutFloat + beginTri * SSTLContext::VerticesPerTriangle * SSTLContext::FloatChannelsPerVertex;
			for (uint64_t tri = beginTri; tri < endTri; ++tri)
			{
				const uint8_t* const triRecord = localCursor;
				localCursor += SSTLContext::TriangleRecordBytes;
				float triValues[SSTLContext::TriangleFloatCount];
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
				posCursor += SSTLContext::VerticesPerTriangle * SSTLContext::FloatChannelsPerVertex;
				normalCursor += SSTLContext::VerticesPerTriangle * SSTLContext::FloatChannelsPerVertex;
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
						const size_t runBytes = runTriangles * SSTLContext::VerticesPerTriangle * SSTLContext::FloatChannelsPerVertex * sizeof(float);
						positionHasher.update(posOutFloat + begin * SSTLContext::VerticesPerTriangle * SSTLContext::FloatChannelsPerVertex, runBytes);
						chunkIx = runEnd;
					}
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
						const size_t runBytes = runTriangles * SSTLContext::VerticesPerTriangle * SSTLContext::FloatChannelsPerVertex * sizeof(float);
						normalHasher.update(normalOutFloat + begin * SSTLContext::VerticesPerTriangle * SSTLContext::FloatChannelsPerVertex, runBytes);
						chunkIx = runEnd;
					}
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
		const auto runParallelWorkers = [](const size_t localWorkerCount, const auto& fn) -> void
		{
			if (localWorkerCount <= 1ull) { fn(0ull); return; }
			core::vector<std::jthread> workers;
			workers.reserve(localWorkerCount - 1ull);
			for (size_t workerIx = 1ull; workerIx < localWorkerCount; ++workerIx)
				workers.emplace_back([&fn, workerIx]() { fn(workerIx); });
			fn(0ull);
		};
		runParallelWorkers(workerCount, parseWorker);
		if (positionHashThread.joinable())
			positionHashThread.join();
		if (normalHashThread.joinable())
			normalHashThread.join();
		if (hashInParsePipeline)
		{
			if (!hashPipelineOk.load(std::memory_order_relaxed))
				return {};
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
		geometry->setPositionView(std::move(posView));
		geometry->setNormalView(std::move(normalView));
	}
	else
	{
		parsePath = "ascii_fallback";
		if (!wholeFileData)
		{
			wholeFilePayload.resize(filesize + 1ull);
			if (!readFileWithPolicy(context.inner.mainFile, wholeFilePayload.data(), 0ull, filesize, ioPlan, &context.ioTelemetry))
				return {};
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
		if (positions.empty())
			return {};

		triangleCount = positions.size() / SSTLContext::VerticesPerTriangle;
		vertexCount = positions.size();

		auto posView = stlCreateAdoptedFloat3View(std::move(positions));
		auto normalView = stlCreateAdoptedFloat3View(std::move(normals));
		if (!posView || !normalView)
			return {};
		geometry->setPositionView(std::move(posView));
		geometry->setNormalView(std::move(normalView));
	}

	if (vertexCount == 0ull)
		return {};

	if (computeContentHashes && !contentHashesAssigned)
	{
		recomputeGeometryContentHashesParallel(geometry.get(), _params.ioPolicy);
	}

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
	const uint64_t ioMinRead = context.ioTelemetry.getMinOrZero();
	const uint64_t ioAvgRead = context.ioTelemetry.getAvgOrZero();
	if (isTinyIOTelemetryLikely(context.ioTelemetry, static_cast<uint64_t>(filesize)))
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
		"STL loader stats: file=%s binary=%d parse_path=%s triangles=%llu vertices=%llu io_reads=%llu io_min_read=%llu io_avg_read=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
		system::ILogger::ELL_PERFORMANCE,
		_file->getFileName().string().c_str(),
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

bool CSTLMeshFileLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr) const
{
	if (!_file || _file->getSize() <= SSTLContext::TextProbeBytes)
		return false;

	const size_t fileSize = _file->getSize();
	if (fileSize < SSTLContext::BinaryPrefixBytes)
	{
		char header[SSTLContext::TextProbeBytes] = {};
		if (!readFileExact(_file, header, 0ull, sizeof(header)))
			return false;
		return std::strncmp(header, "solid ", SSTLContext::TextProbeBytes) == 0;
	}

	std::array<uint8_t, SSTLContext::BinaryPrefixBytes> prefix = {};
	if (!readFileExact(_file, prefix.data(), 0ull, prefix.size()))
		return false;

	uint32_t triangleCount = 0u;
	std::memcpy(&triangleCount, prefix.data() + SSTLContext::BinaryHeaderBytes, sizeof(triangleCount));
	if (std::memcmp(prefix.data(), "solid ", SSTLContext::TextProbeBytes) == 0)
		return true;

	return fileSize == (SSTLContext::TriangleRecordBytes * triangleCount + SSTLContext::BinaryPrefixBytes);
}

}

#endif // _NBL_COMPILE_WITH_STL_LOADER_

