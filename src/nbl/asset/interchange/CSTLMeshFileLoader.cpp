#ifdef _NBL_COMPILE_WITH_STL_LOADER_
// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CSTLMeshFileLoader.h"

#include "SSTLPolygonGeometryAuxLayout.h"
#include "impl/SLoadSession.h"
#include "impl/STextParse.h"
#include "nbl/asset/asset.h"
#include "nbl/asset/format/convertColor.h"
#include "nbl/asset/interchange/SGeometryContentHash.h"
#include "nbl/asset/interchange/SGeometryLoaderCommon.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "nbl/asset/interchange/SLoaderRuntimeTuning.h"
#include "nbl/asset/metadata/CSTLMetadata.h"
#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
#include "nbl/asset/utils/SGeometryNormalCommon.h"
#include "nbl/builtin/hlsl/shapes/AABBAccumulator.hlsl"
#include "nbl/core/hash/blake.h"
#include "nbl/system/IFile.h"

#include <optional>

namespace nbl::asset
{

namespace
{

struct Parse
{
	using Common = impl::TextParse;

	static hlsl::float32_t3 resolveStoredNormal(const hlsl::float32_t3& fileNormal)
	{
		const float fileLen2 = hlsl::dot(fileNormal, fileNormal);
		if (fileLen2 > 0.f && std::abs(fileLen2 - 1.f) < 1e-4f)
			return fileNormal;
		return SGeometryNormalCommon::normalizeOrZero(fileNormal);
	}

	static void pushTriangleReversed(const std::array<hlsl::float32_t3, 3>& p, core::vector<hlsl::float32_t3>& positions)
	{
		positions.push_back(p[2u]);
		positions.push_back(p[1u]);
		positions.push_back(p[0u]);
	}

	static uint32_t decodeViscamColorToB8G8R8A8(const uint16_t packedColor)
	{
		std::array<const void*, 4> src = {&packedColor};
		uint32_t outColor = 0u;
		convertColor<EF_A1R5G5B5_UNORM_PACK16, EF_B8G8R8A8_UNORM>(src.data(), &outColor, 0u, 0u);
		return outColor;
	}

	struct Context
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

	class AsciiParser
	{
		public:
			inline AsciiParser(const char* begin, const char* end) : m_cursor(begin), m_end(end) {}

			inline std::optional<std::string_view> readToken()
			{
				return Common::readToken(m_cursor, m_end);
			}

			inline std::optional<float> readFloat()
			{
				float value = 0.f;
				if (!Common::parseNumber(m_cursor, m_end, value))
					return std::nullopt;
				return value;
			}

			inline std::optional<hlsl::float32_t3> readVec3()
			{
				const auto x = readFloat();
				const auto y = readFloat();
				const auto z = readFloat();
				if (!x.has_value() || !y.has_value() || !z.has_value())
					return std::nullopt;
				return hlsl::float32_t3(*x, *y, *z);
			}

		private:
			const char* m_cursor = nullptr;
			const char* m_end = nullptr;
	};

	class SplitBlockMemoryResource final : public core::refctd_memory_resource
	{
		public:
			inline SplitBlockMemoryResource(core::smart_refctd_ptr<core::refctd_memory_resource>&& upstream, void* block, const size_t blockBytes, const size_t alignment)
				: m_upstream(std::move(upstream)), m_block(block), m_blockBytes(blockBytes), m_alignment(alignment) {}

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
			inline ~SplitBlockMemoryResource() override
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
};

}

CSTLMeshFileLoader::CSTLMeshFileLoader(asset::IAssetManager*)
{
}

const char** CSTLMeshFileLoader::getAssociatedFileExtensions() const
{
	static const char* ext[] = { "stl", nullptr };
	return ext;
}

SAssetBundle CSTLMeshFileLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override [[maybe_unused]], uint32_t _hierarchyLevel [[maybe_unused]])
{
	using Context = Parse::Context;
	using AsciiParser = Parse::AsciiParser;
	using SplitBlockMemoryResource = Parse::SplitBlockMemoryResource;

	if (!_file)
		return {};

	uint64_t triangleCount = 0u;
	const char* parsePath = "unknown";
	const bool computeContentHashes = !_params.loaderFlags.hasAnyFlag(IAssetLoader::ELPF_DONT_COMPUTE_CONTENT_HASHES);
	bool hasTriangleColors = false;

	Context context = {asset::IAssetLoader::SAssetLoadContext{_params, _file}, 0ull};
	const size_t filesize = context.inner.mainFile->getSize();
	if (filesize < Context::TextProbeBytes)
		return {};

	impl::SLoadSession loadSession = {};
	if (!impl::SLoadSession::begin(_params.logger, "STL loader", _file, _params.ioPolicy, static_cast<uint64_t>(filesize), true, loadSession))
		return {};

	core::vector<uint8_t> wholeFilePayload;
	const uint8_t* wholeFileData = nullptr;
	if (loadSession.isWholeFile())
	{
		wholeFileData = loadSession.mapOrReadWholeFile(wholeFilePayload, &context.ioTelemetry);
		if (!wholeFileData)
			return {};
	}

    bool binary = false;
    bool hasBinaryTriCountFromDetect = false;
    uint32_t binaryTriCountFromDetect = 0u;
    {
        std::array<uint8_t, Context::BinaryPrefixBytes> prefix = {};
        bool hasPrefix = false;
        if (wholeFileData && filesize >= Context::BinaryPrefixBytes) {
            std::memcpy(prefix.data(), wholeFileData, Context::BinaryPrefixBytes);
            hasPrefix = true;
        } else {
            hasPrefix = filesize >= Context::BinaryPrefixBytes &&
                        SInterchangeIO::readFileExact(
                            context.inner.mainFile, prefix.data(), 0ull,
                            Context::BinaryPrefixBytes, &context.ioTelemetry);
        }
        bool startsWithSolid = false;
        if (hasPrefix) {
            startsWithSolid =
                (std::memcmp(prefix.data(), "solid ", Context::TextProbeBytes) == 0);
        } else {
            char header[Context::TextProbeBytes] = {};
            if (wholeFileData)
                std::memcpy(header, wholeFileData, sizeof(header));
            else if (!SInterchangeIO::readFileExact(context.inner.mainFile, header,
                                                    0ull, sizeof(header),
                                                    &context.ioTelemetry))
                return {};
            startsWithSolid =
                (std::strncmp(header, "solid ", Context::TextProbeBytes) == 0);
        }

        bool binaryBySize = false;
        if (hasPrefix) {
            uint32_t triCount = 0u;
            std::memcpy(&triCount, prefix.data() + Context::BinaryHeaderBytes,
                        sizeof(triCount));
            binaryTriCountFromDetect = triCount;
            hasBinaryTriCountFromDetect = true;
            const uint64_t expectedSize =
                Context::BinaryPrefixBytes +
                static_cast<uint64_t>(triCount) * Context::TriangleRecordBytes;
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
    hlsl::shapes::util::AABBAccumulator3<float> parsedAABB =
        hlsl::shapes::util::createAABBAccumulator<float>();
    uint64_t vertexCount = 0ull;

    if (binary) {
        parsePath = "binary_fast";
        if (filesize < Context::BinaryPrefixBytes)
            return {};

        uint32_t triangleCount32 = binaryTriCountFromDetect;
        if (!hasBinaryTriCountFromDetect) {
            if (!SInterchangeIO::readFileExact(
                    context.inner.mainFile, &triangleCount32,
                    Context::BinaryHeaderBytes, sizeof(triangleCount32),
                    &context.ioTelemetry))
                return {};
        }

        triangleCount = triangleCount32;
        const size_t dataSize =
            static_cast<size_t>(triangleCount) * Context::TriangleRecordBytes;
        const size_t expectedSize = Context::BinaryPrefixBytes + dataSize;
        if (filesize < expectedSize)
            return {};

        const uint8_t* payloadData = wholeFileData ? (wholeFileData + Context::BinaryPrefixBytes) : loadSession.readRange(Context::BinaryPrefixBytes, dataSize, wholeFilePayload, &context.ioTelemetry);
        if (!payloadData)
            return {};

        vertexCount = triangleCount * Context::VerticesPerTriangle;
        const size_t vertexCountSizeT = static_cast<size_t>(vertexCount);
        if (vertexCountSizeT >
            (std::numeric_limits<size_t>::max() / sizeof(hlsl::float32_t3)))
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
        auto blockResource = core::make_smart_refctd_ptr<SplitBlockMemoryResource>(
            core::smart_refctd_ptr<core::refctd_memory_resource>(
                std::move(upstream)),
            block, blockBytes, alignof(float));
        auto posBuffer = ICPUBuffer::create(
            {{viewByteSize},
             block,
             core::smart_refctd_ptr<core::refctd_memory_resource>(blockResource),
             alignof(float)},
            core::adopt_memory);
        auto normalBuffer = ICPUBuffer::create(
            {{viewByteSize},
             reinterpret_cast<uint8_t*>(block) + viewByteSize,
             core::smart_refctd_ptr<core::refctd_memory_resource>(blockResource),
             alignof(float)},
            core::adopt_memory);
        if (!posBuffer || !normalBuffer)
            return {};
        ICPUPolygonGeometry::SDataView posView = {};
        posView.composed = {.stride = sizeof(hlsl::float32_t3),
                            .format = EF_R32G32B32_SFLOAT,
                            .rangeFormat = IGeometryBase::getMatchingAABBFormat(
                                EF_R32G32B32_SFLOAT)};
        posView.src = {
            .offset = 0ull,
            .size = viewByteSize,
            .buffer = std::move(posBuffer)};
        ICPUPolygonGeometry::SDataView normalView = {};
        normalView.composed = {.stride = sizeof(hlsl::float32_t3),
                               .format = EF_R32G32B32_SFLOAT,
                               .rangeFormat = IGeometryBase::getMatchingAABBFormat(
                                   EF_R32G32B32_SFLOAT)};
        normalView.src = {.offset = 0ull,
                          .size = viewByteSize,
                          .buffer = std::move(normalBuffer)};
        auto* posOutFloat = reinterpret_cast<float*>(posView.getPointer());
        auto* normalOutFloat = reinterpret_cast<float*>(normalView.getPointer());
        if (!posOutFloat || !normalOutFloat)
            return {};

        const uint8_t* cursor = payloadData;
        const uint8_t* const end = cursor + dataSize;
        if (end < cursor ||
            static_cast<size_t>(end - cursor) <
                static_cast<size_t>(triangleCount) * Context::TriangleRecordBytes)
            return {};
        core::vector<uint32_t> faceColors(static_cast<size_t>(triangleCount), 0u);
        std::atomic_bool colorValidForAllFaces = true;
        const size_t hw = SLoaderRuntimeTuner::resolveHardwareThreads();
        const size_t hardMaxWorkers = SLoaderRuntimeTuner::resolveHardMaxWorkers(
            hw, _params.ioPolicy.runtimeTuning.workerHeadroom);
        SLoaderRuntimeTuningRequest parseTuningRequest = {};
        parseTuningRequest.inputBytes = dataSize;
        parseTuningRequest.totalWorkUnits = triangleCount;
        parseTuningRequest.minBytesPerWorker = Context::TriangleRecordBytes;
        parseTuningRequest.hardwareThreads = static_cast<uint32_t>(hw);
        parseTuningRequest.hardMaxWorkers = static_cast<uint32_t>(hardMaxWorkers);
        parseTuningRequest.targetChunksPerWorker =
            _params.ioPolicy.runtimeTuning.targetChunksPerWorker;
        parseTuningRequest.minChunkWorkUnits = 1ull;
        parseTuningRequest.maxChunkWorkUnits =
            std::max<uint64_t>(1ull, triangleCount);
        parseTuningRequest.sampleData = payloadData;
        parseTuningRequest.sampleBytes =
            SLoaderRuntimeTuner::resolveSampleBytes(_params.ioPolicy, dataSize);
        const auto parseTuning =
            SLoaderRuntimeTuner::tune(_params.ioPolicy, parseTuningRequest);
        const size_t workerCount = std::max<size_t>(
            1ull,
            std::min(parseTuning.workerCount,
                     static_cast<size_t>(std::max<uint64_t>(1ull, triangleCount))));
        static constexpr bool ComputeAABBInParse = true;
        struct SThreadAABB {
            bool has = false;
            float minX = 0.f;
            float minY = 0.f;
            float minZ = 0.f;
            float maxX = 0.f;
            float maxY = 0.f;
            float maxZ = 0.f;
        };
        std::vector<SThreadAABB> threadAABBs(ComputeAABBInParse ? workerCount
                                                                : 0ull);
        const uint64_t parseChunkTriangles =
            std::max<uint64_t>(1ull, parseTuning.chunkWorkUnits);
        const size_t parseChunkCount = static_cast<size_t>(
            SLoaderRuntimeTuner::ceilDiv(triangleCount, parseChunkTriangles));
        const bool hashInParsePipeline = computeContentHashes;
        std::vector<uint8_t> hashChunkReady(
            hashInParsePipeline ? parseChunkCount : 0ull, 0u);
        std::atomic_bool hashPipelineOk = true;
        core::blake3_hash_t parsedPositionHash =
            static_cast<core::blake3_hash_t>(core::blake3_hasher{});
        core::blake3_hash_t parsedNormalHash =
            static_cast<core::blake3_hash_t>(core::blake3_hasher{});
        auto parseRange = [&](const uint64_t beginTri, const uint64_t endTri,
                              SThreadAABB& localAABB) -> void {
            const uint8_t* localCursor =
                payloadData + beginTri * Context::TriangleRecordBytes;
            float* posCursor = posOutFloat + beginTri * Context::VerticesPerTriangle *
                                                 Context::FloatChannelsPerVertex;
            float* normalCursor =
                normalOutFloat + beginTri * Context::VerticesPerTriangle *
                                     Context::FloatChannelsPerVertex;
            for (uint64_t tri = beginTri; tri < endTri; ++tri) {
                const uint8_t* const triRecord = localCursor;
                localCursor += Context::TriangleRecordBytes;
                std::array<float, Context::TriangleFloatCount> triValues = {};
                std::memcpy(triValues.data(), triRecord, sizeof(triValues));
                uint16_t packedColor = 0u;
                std::memcpy(&packedColor, triRecord + Context::TriangleFloatBytes,
                            sizeof(packedColor));
                if (packedColor & 0x8000u)
                    faceColors[static_cast<size_t>(tri)] =
                        Parse::decodeViscamColorToB8G8R8A8(packedColor);
                else
                    colorValidForAllFaces.store(false, std::memory_order_relaxed);

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
                if constexpr (ComputeAABBInParse) {
                    if (!localAABB.has) {
                        localAABB.has = true;
                        localAABB.minX = vertex0x;
                        localAABB.minY = vertex0y;
                        localAABB.minZ = vertex0z;
                        localAABB.maxX = vertex0x;
                        localAABB.maxY = vertex0y;
                        localAABB.maxZ = vertex0z;
                    }
                    if (vertex0x < localAABB.minX)
                        localAABB.minX = vertex0x;
                    if (vertex0y < localAABB.minY)
                        localAABB.minY = vertex0y;
                    if (vertex0z < localAABB.minZ)
                        localAABB.minZ = vertex0z;
                    if (vertex0x > localAABB.maxX)
                        localAABB.maxX = vertex0x;
                    if (vertex0y > localAABB.maxY)
                        localAABB.maxY = vertex0y;
                    if (vertex0z > localAABB.maxZ)
                        localAABB.maxZ = vertex0z;
                    if (vertex1x < localAABB.minX)
                        localAABB.minX = vertex1x;
                    if (vertex1y < localAABB.minY)
                        localAABB.minY = vertex1y;
                    if (vertex1z < localAABB.minZ)
                        localAABB.minZ = vertex1z;
                    if (vertex1x > localAABB.maxX)
                        localAABB.maxX = vertex1x;
                    if (vertex1y > localAABB.maxY)
                        localAABB.maxY = vertex1y;
                    if (vertex1z > localAABB.maxZ)
                        localAABB.maxZ = vertex1z;
                    if (vertex2x < localAABB.minX)
                        localAABB.minX = vertex2x;
                    if (vertex2y < localAABB.minY)
                        localAABB.minY = vertex2y;
                    if (vertex2z < localAABB.minZ)
                        localAABB.minZ = vertex2z;
                    if (vertex2x > localAABB.maxX)
                        localAABB.maxX = vertex2x;
                    if (vertex2y > localAABB.maxY)
                        localAABB.maxY = vertex2y;
                    if (vertex2z > localAABB.maxZ)
                        localAABB.maxZ = vertex2z;
                }
                if (normalX == 0.f && normalY == 0.f && normalZ == 0.f) {
                    const float edge10x = vertex1x - vertex0x;
                    const float edge10y = vertex1y - vertex0y;
                    const float edge10z = vertex1z - vertex0z;
                    const float edge20x = vertex2x - vertex0x;
                    const float edge20y = vertex2y - vertex0y;
                    const float edge20z = vertex2z - vertex0z;

                    normalX = edge10y * edge20z - edge10z * edge20y;
                    normalY = edge10z * edge20x - edge10x * edge20z;
                    normalZ = edge10x * edge20y - edge10y * edge20x;
                    const float planeLen2 =
                        normalX * normalX + normalY * normalY + normalZ * normalZ;
                    if (planeLen2 > 0.f) {
                        const float invLen = 1.f / std::sqrt(planeLen2);
                        normalX *= invLen;
                        normalY *= invLen;
                        normalZ *= invLen;
                    } else {
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
                posCursor +=
                    Context::VerticesPerTriangle * Context::FloatChannelsPerVertex;
                normalCursor +=
                    Context::VerticesPerTriangle * Context::FloatChannelsPerVertex;
            }
        };
        std::jthread positionHashThread;
        std::jthread normalHashThread;
        if (hashInParsePipeline) {
            auto launchHashThread =
                [&](const float* srcFloat,
                    core::blake3_hash_t& outHash) -> std::jthread {
                return std::jthread([&, srcFloat, outHashPtr = &outHash]() {
                    try {
                        core::blake3_hasher hasher;
                        size_t chunkIx = 0ull;
                        while (chunkIx < parseChunkCount) {
                            auto ready = std::atomic_ref<uint8_t>(hashChunkReady[chunkIx]);
                            while (ready.load(std::memory_order_acquire) == 0u)
                                ready.wait(0u, std::memory_order_acquire);

                            size_t runEnd = chunkIx + 1ull;
                            while (runEnd < parseChunkCount) {
                                const auto runReady =
                                    std::atomic_ref<uint8_t>(hashChunkReady[runEnd])
                                        .load(std::memory_order_acquire);
                                if (runReady == 0u)
                                    break;
                                ++runEnd;
                            }

                            const uint64_t begin =
                                static_cast<uint64_t>(chunkIx) * parseChunkTriangles;
                            const uint64_t endTri = std::min<uint64_t>(
                                static_cast<uint64_t>(runEnd) * parseChunkTriangles,
                                triangleCount);
                            const size_t runTriangles = static_cast<size_t>(endTri - begin);
                            const size_t runBytes =
                                runTriangles * Context::VerticesPerTriangle *
                                Context::FloatChannelsPerVertex * sizeof(float);
                            hasher.update(srcFloat + begin * Context::VerticesPerTriangle *
                                                         Context::FloatChannelsPerVertex,
                                          runBytes);
                            chunkIx = runEnd;
                        }
                        *outHashPtr = static_cast<core::blake3_hash_t>(hasher);
                    } catch (...) {
                        hashPipelineOk.store(false, std::memory_order_relaxed);
                    }
                });
            };
            positionHashThread = launchHashThread(posOutFloat, parsedPositionHash);
            normalHashThread = launchHashThread(normalOutFloat, parsedNormalHash);
        }
        std::atomic_size_t nextChunkIx = 0ull;
        auto parseWorker = [&](const size_t workerIx) -> void {
            SThreadAABB localAABB = {};
            while (true) {
                const size_t chunkIx =
                    nextChunkIx.fetch_add(1ull, std::memory_order_relaxed);
                if (chunkIx >= parseChunkCount)
                    break;
                const uint64_t begin =
                    static_cast<uint64_t>(chunkIx) * parseChunkTriangles;
                const uint64_t endTri =
                    std::min<uint64_t>(begin + parseChunkTriangles, triangleCount);
                parseRange(begin, endTri, localAABB);
                if (hashInParsePipeline) {
                    auto ready = std::atomic_ref<uint8_t>(hashChunkReady[chunkIx]);
                    ready.store(1u, std::memory_order_release);
                    ready.notify_all();
                }
            }
            if constexpr (ComputeAABBInParse)
                threadAABBs[workerIx] = localAABB;
        };
        SLoaderRuntimeTuner::dispatchWorkers(workerCount, parseWorker);
        if (positionHashThread.joinable())
            positionHashThread.join();
        if (normalHashThread.joinable())
            normalHashThread.join();
        if (hashInParsePipeline) {
            if (!hashPipelineOk.load(std::memory_order_relaxed))
                return {};
            posView.src.buffer->setContentHash(parsedPositionHash);
            normalView.src.buffer->setContentHash(parsedNormalHash);
        }
        if constexpr (ComputeAABBInParse) {
            for (const auto& localAABB : threadAABBs) {
                if (!localAABB.has)
                    continue;
                hlsl::shapes::util::extendAABBAccumulator(
                    parsedAABB, localAABB.minX, localAABB.minY, localAABB.minZ);
                hlsl::shapes::util::extendAABBAccumulator(
                    parsedAABB, localAABB.maxX, localAABB.maxY, localAABB.maxZ);
            }
        }
        geometry->setPositionView(std::move(posView));
        geometry->setNormalView(std::move(normalView));
        if (colorValidForAllFaces.load(std::memory_order_relaxed)) {
            core::vector<uint32_t> vertexColors(vertexCountSizeT);
            for (size_t triIx = 0ull; triIx < static_cast<size_t>(triangleCount);
                 ++triIx) {
                const uint32_t triColor = faceColors[triIx];
                const size_t baseIx = triIx * Context::VerticesPerTriangle;
                vertexColors[baseIx + 0ull] = triColor;
                vertexColors[baseIx + 1ull] = triColor;
                vertexColors[baseIx + 2ull] = triColor;
            }
            auto colorView =
                SGeometryLoaderCommon::createAdoptedView<EF_B8G8R8A8_UNORM>(
                    std::move(vertexColors));
            if (!colorView)
                return {};
            auto* const auxViews = geometry->getAuxAttributeViews();
            auxViews->resize(SSTLPolygonGeometryAuxLayout::COLOR0 + 1u);
            auxViews->operator[](SSTLPolygonGeometryAuxLayout::COLOR0) =
                std::move(colorView);
            hasTriangleColors = true;
        }
    } else {
        parsePath = "ascii_fallback";
        if (!wholeFileData)
        {
            wholeFileData = loadSession.mapOrReadWholeFile(wholeFilePayload, &context.ioTelemetry);
            if (!wholeFileData)
                return {};
        }

        const char* const begin = reinterpret_cast<const char*>(wholeFileData);
        const char* const end = begin + filesize;
        AsciiParser parser(begin, end);
        core::vector<hlsl::float32_t3> positions;
        core::vector<hlsl::float32_t3> normals;
        const auto firstToken = parser.readToken();
        if (!firstToken.has_value() || *firstToken != std::string_view("solid"))
            return {};

        for (;;) {
            const auto maybeToken = parser.readToken();
            if (!maybeToken.has_value())
                break;
            const std::string_view textToken = *maybeToken;
            if (textToken == std::string_view("endsolid"))
                break;
            if (textToken != std::string_view("facet"))
                continue;

            const auto normalKeyword = parser.readToken();
            if (!normalKeyword.has_value() ||
                *normalKeyword != std::string_view("normal"))
                return {};

            const auto fileNormal = parser.readVec3();
            if (!fileNormal.has_value())
                return {};

            const auto outerKeyword = parser.readToken();
            if (!outerKeyword.has_value() ||
                *outerKeyword != std::string_view("outer"))
                return {};
            const auto loopKeyword = parser.readToken();
            if (!loopKeyword.has_value() || *loopKeyword != std::string_view("loop"))
                return {};

            std::array<hlsl::float32_t3, 3> p = {};
            for (uint32_t i = 0u; i < 3u; ++i) {
                const auto vertexKeyword = parser.readToken();
                if (!vertexKeyword.has_value() ||
                    *vertexKeyword != std::string_view("vertex"))
                    return {};
                const auto vertex = parser.readVec3();
                if (!vertex.has_value())
                    return {};
                p[i] = *vertex;
            }

            Parse::pushTriangleReversed(p, positions);
            hlsl::float32_t3 faceNormal = Parse::resolveStoredNormal(*fileNormal);
            if (hlsl::dot(faceNormal, faceNormal) <= 0.f)
                faceNormal =
                    SGeometryNormalCommon::computeFaceNormal(p[2u], p[1u], p[0u]);
            normals.push_back(faceNormal);
            normals.push_back(faceNormal);
            normals.push_back(faceNormal);
            hlsl::shapes::util::extendAABBAccumulator(parsedAABB, p[2u]);
            hlsl::shapes::util::extendAABBAccumulator(parsedAABB, p[1u]);
            hlsl::shapes::util::extendAABBAccumulator(parsedAABB, p[0u]);

            const auto endLoopKeyword = parser.readToken();
            if (!endLoopKeyword.has_value() ||
                *endLoopKeyword != std::string_view("endloop"))
                return {};
            const auto endFacetKeyword = parser.readToken();
            if (!endFacetKeyword.has_value() ||
                *endFacetKeyword != std::string_view("endfacet"))
                return {};
        }
        if (positions.empty())
            return {};

        triangleCount = positions.size() / Context::VerticesPerTriangle;
        vertexCount = positions.size();

        auto posView =
            SGeometryLoaderCommon::createAdoptedView<EF_R32G32B32_SFLOAT>(
                std::move(positions));
        auto normalView =
            SGeometryLoaderCommon::createAdoptedView<EF_R32G32B32_SFLOAT>(
                std::move(normals));
        if (!posView || !normalView)
            return {};
        geometry->setPositionView(std::move(posView));
        geometry->setNormalView(std::move(normalView));
    }

    if (vertexCount == 0ull)
        return {};

    if (computeContentHashes) {
        SPolygonGeometryContentHash::computeMissing(geometry.get(),
                                                    _params.ioPolicy);
    }

    if (!parsedAABB.empty())
        geometry->applyAABB(parsedAABB.value);
    else {
        CPolygonGeometryManipulator::recomputeAABB(geometry.get());
    }
    const uint64_t ioMinRead = context.ioTelemetry.getMinOrZero();
    const uint64_t ioAvgRead = context.ioTelemetry.getAvgOrZero();
    loadSession.logTinyIO(_params.logger, context.ioTelemetry);
    _params.logger.log(
        "STL loader stats: file=%s binary=%d parse_path=%s triangles=%llu "
        "vertices=%llu colors=%d io_reads=%llu io_min_read=%llu io_avg_read=%llu "
        "io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
        system::ILogger::ELL_PERFORMANCE, _file->getFileName().string().c_str(),
        binary ? 1 : 0, parsePath, static_cast<unsigned long long>(triangleCount),
        static_cast<unsigned long long>(vertexCount), hasTriangleColors ? 1 : 0,
        static_cast<unsigned long long>(context.ioTelemetry.callCount),
        static_cast<unsigned long long>(ioMinRead),
        static_cast<unsigned long long>(ioAvgRead),
        system::to_string(_params.ioPolicy.strategy).c_str(),
        system::to_string(loadSession.ioPlan.strategy).c_str(),
        static_cast<unsigned long long>(loadSession.ioPlan.chunkSizeBytes()), loadSession.ioPlan.reason);
    auto meta = core::make_smart_refctd_ptr<CSTLMetadata>();
    return SAssetBundle(std::move(meta), {std::move(geometry)});
}

bool CSTLMeshFileLoader::isALoadableFileFormat(
    system::IFile* _file, const system::logger_opt_ptr) const {
    using Context = Parse::Context;

    if (!_file || _file->getSize() <= Context::TextProbeBytes)
        return false;

    const size_t fileSize = _file->getSize();
    if (fileSize < Context::BinaryPrefixBytes) {
        char header[Context::TextProbeBytes] = {};
        if (!SInterchangeIO::readFileExact(_file, header, 0ull, sizeof(header)))
            return false;
        return std::strncmp(header, "solid ", Context::TextProbeBytes) == 0;
    }

    std::array<uint8_t, Context::BinaryPrefixBytes> prefix = {};
    if (!SInterchangeIO::readFileExact(_file, prefix.data(), 0ull, prefix.size()))
        return false;

    uint32_t triangleCount = 0u;
    std::memcpy(&triangleCount, prefix.data() + Context::BinaryHeaderBytes,
                sizeof(triangleCount));
    if (std::memcmp(prefix.data(), "solid ", Context::TextProbeBytes) == 0)
        return true;

    return fileSize == (Context::TriangleRecordBytes * triangleCount +
                        Context::BinaryPrefixBytes);
}

}

#endif // _NBL_COMPILE_WITH_STL_LOADER_
