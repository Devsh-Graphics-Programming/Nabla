#ifdef _NBL_COMPILE_WITH_OBJ_LOADER_
// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#include "nbl/core/declarations.h"
#include "nbl/asset/IAssetManager.h"
#include "nbl/asset/ICPUGeometryCollection.h"
#include "nbl/asset/interchange/SGeometryContentHash.h"
#include "nbl/asset/interchange/SGeometryLoaderCommon.h"
#include "nbl/asset/interchange/SOBJPolygonGeometryAuxLayout.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "nbl/asset/interchange/SLoaderRuntimeTuning.h"
#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
#include "nbl/builtin/hlsl/shapes/AABBAccumulator.hlsl"
#include "nbl/system/IFile.h"
#include "COBJMeshFileLoader.h"
#include "impl/SFileAccess.h"
#include "impl/STextParse.h"
#include <array>
#include <bit>
#include <cctype>
#include <optional>
#include <span>
#include <string_view>
namespace nbl::asset
{
namespace
{
struct Parse
{
	using Common = impl::TextParse;
	struct VertexDedupNode { int32_t uv = -1; int32_t normal = -1; uint32_t smoothingGroup = 0u; uint32_t outIndex = 0u; int32_t next = -1; };
	static bool resolveIndex(const int32_t rawIndex, const size_t elementCount, int32_t& resolved)
	{
		if (rawIndex > 0)
		{
			const uint64_t oneBased = static_cast<uint64_t>(rawIndex);
			if (oneBased == 0ull)
				return false;
			const uint64_t zeroBased = oneBased - 1ull;
			if (zeroBased >= elementCount)
				return false;
			resolved = static_cast<int32_t>(zeroBased);
			return true;
		}
		const int64_t zeroBased = static_cast<int64_t>(elementCount) + static_cast<int64_t>(rawIndex);
		if (zeroBased < 0 || zeroBased >= static_cast<int64_t>(elementCount))
			return false;
		resolved = static_cast<int32_t>(zeroBased);
		return true;
	}
	static void parseSmoothingGroup(const char* linePtr, const char* const lineEnd, uint32_t& outGroup)
	{
		Common::skipInlineWhitespace(linePtr, lineEnd);
		if (linePtr >= lineEnd)
			return void(outGroup = 0u);
		const char* const tokenStart = linePtr;
		while (linePtr < lineEnd && !Common::isInlineWhitespace(*linePtr))
			++linePtr;
		const std::string_view token(tokenStart, static_cast<size_t>(linePtr - tokenStart));
		if (token.size() == 2u && std::tolower(token[0]) == 'o' && std::tolower(token[1]) == 'n')
			return void(outGroup = 1u);
		if (token.size() == 3u && std::tolower(token[0]) == 'o' && std::tolower(token[1]) == 'f' && std::tolower(token[2]) == 'f')
			return void(outGroup = 0u);
		uint32_t value = 0u;
		outGroup = Common::parseExactNumber(token, value) ? value : 0u;
	}
	static std::string parseIdentifier(const char* linePtr, const char* const lineEnd, const std::string_view fallback)
	{
		const char* endPtr = lineEnd;
		Common::skipInlineWhitespace(linePtr, lineEnd);
		while (endPtr > linePtr && Common::isInlineWhitespace(endPtr[-1]))
			--endPtr;
		if (linePtr >= endPtr)
			return std::string(fallback);
		return std::string(linePtr, static_cast<size_t>(endPtr - linePtr));
	}
	static bool parseTrianglePositiveTripletLine(const char* const lineStart, const char* const lineEnd, std::array<hlsl::int32_t3, 3>& out, const size_t posCount, const size_t uvCount, const size_t normalCount)
	{
		const char* ptr = lineStart;
		auto parsePositive = [&](const size_t count, int32_t& outIx) -> bool {
			uint32_t value = 0u;
			if (!Common::parseNonZeroNumber(ptr, lineEnd, value))
				return false;
			if (value > count)
				return false;
			outIx = value - 1u;
			return true;
		};
		for (uint32_t corner = 0u; corner < 3u; ++corner)
		{
			Common::skipInlineWhitespace(ptr, lineEnd);
			if (ptr >= lineEnd || !Common::isDigit(*ptr))
				return false;
			int32_t posIx = -1;
			if (!parsePositive(posCount, posIx))
				return false;
			if (ptr >= lineEnd || *ptr != '/')
				return false;
			++ptr;
			int32_t uvIx = -1;
			if (!parsePositive(uvCount, uvIx))
				return false;
			if (ptr >= lineEnd || *ptr != '/')
				return false;
			++ptr;
			int32_t normalIx = -1;
			if (!parsePositive(normalCount, normalIx))
				return false;
			out[corner] = hlsl::int32_t3(posIx, uvIx, normalIx);
		}
		Common::skipInlineWhitespace(ptr, lineEnd);
		return ptr == lineEnd;
	}
	static bool parseTrianglePositivePositionNormalLine(const char* const lineStart, const char* const lineEnd, std::array<hlsl::int32_t3, 3>& out, const size_t posCount, const size_t normalCount)
	{
		const char* ptr = lineStart;
		auto parsePositive = [&](const size_t count, int32_t& outIx) -> bool {
			uint32_t value = 0u;
			if (!Common::parseNonZeroNumber(ptr, lineEnd, value))
				return false;
			if (value > count)
				return false;
			outIx = value - 1u;
			return true;
		};
		for (uint32_t corner = 0u; corner < 3u; ++corner)
		{
			Common::skipInlineWhitespace(ptr, lineEnd);
			if (ptr >= lineEnd || !Common::isDigit(*ptr))
				return false;
			int32_t posIx = -1;
			if (!parsePositive(posCount, posIx))
				return false;
			if ((ptr + 1) >= lineEnd || ptr[0] != '/' || ptr[1] != '/')
				return false;
			ptr += 2;
			int32_t normalIx = -1;
			if (!parsePositive(normalCount, normalIx))
				return false;
			out[corner] = hlsl::int32_t3(posIx, -1, normalIx);
		}
		Common::skipInlineWhitespace(ptr, lineEnd);
		return ptr == lineEnd;
	}
	static bool parseFaceVertexToken(const char*& linePtr, const char* const lineEnd, hlsl::int32_t3& idx, const size_t posCount, const size_t uvCount, const size_t normalCount)
	{
		Common::skipInlineWhitespace(linePtr, lineEnd);
		if (linePtr >= lineEnd)
			return false;
		idx = hlsl::int32_t3(-1, -1, -1);
		const char* ptr = linePtr;
		auto parsePositive = [&](const size_t count, int32_t& outIx) -> bool {
			uint32_t raw = 0u;
			if (!Common::parseNonZeroNumber(ptr, lineEnd, raw))
				return false;
			if (raw > count)
				return false;
			outIx = raw - 1u;
			return true;
		};
		auto parseResolved = [&](const size_t count, int32_t& outIx) -> bool {
			int32_t raw = 0;
			return Common::parseNonZeroNumber(ptr, lineEnd, raw) && resolveIndex(raw, count, outIx);
		};
		if (*ptr != '-' && *ptr != '+')
		{
			if (!parsePositive(posCount, idx.x))
				return false;
			if (ptr < lineEnd && *ptr == '/')
			{
				++ptr;
				if (ptr < lineEnd && *ptr != '/')
				{
					if (!parsePositive(uvCount, idx.y))
						return false;
				}
				if (ptr < lineEnd && *ptr == '/')
				{
					++ptr;
					if (ptr < lineEnd && !Common::isInlineWhitespace(*ptr))
					{
						if (!parsePositive(normalCount, idx.z))
							return false;
					}
				}
				else if (ptr < lineEnd && !Common::isInlineWhitespace(*ptr))
					return false;
			}
			else if (ptr < lineEnd && !Common::isInlineWhitespace(*ptr))
				return false;
		}
		else
		{
			if (!parseResolved(posCount, idx.x))
				return false;
			if (ptr < lineEnd && *ptr == '/')
			{
				++ptr;
				if (ptr < lineEnd && *ptr != '/')
				{
					if (!parseResolved(uvCount, idx.y))
						return false;
				}
				if (ptr < lineEnd && *ptr == '/')
				{
					++ptr;
					if (ptr < lineEnd && !Common::isInlineWhitespace(*ptr))
					{
						if (!parseResolved(normalCount, idx.z))
							return false;
					}
				}
				else if (ptr < lineEnd && !Common::isInlineWhitespace(*ptr))
					return false;
			}
			else if (ptr < lineEnd && !Common::isInlineWhitespace(*ptr))
				return false;
		}
		if (ptr < lineEnd && !Common::isInlineWhitespace(*ptr))
			return false;
		linePtr = ptr;
		return true;
	}
};
}
COBJMeshFileLoader::COBJMeshFileLoader(IAssetManager*)
{
}
COBJMeshFileLoader::~COBJMeshFileLoader() = default;
bool COBJMeshFileLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr) const
{
	if (!_file)
		return false;
	const auto fileSize = _file->getSize();
	if (fileSize <= 0)
		return false;
	constexpr size_t ProbeBytes = 4096ull;
	const size_t bytesToRead = std::min<size_t>(ProbeBytes, static_cast<size_t>(fileSize));
	std::array<char, ProbeBytes> probe = {};
	system::IFile::success_t succ;
	_file->read(succ, probe.data(), 0ull, bytesToRead);
	if (!succ || bytesToRead == 0ull)
		return false;
	const char* ptr = probe.data();
	const char* const end = probe.data() + bytesToRead;
	if ((end - ptr) >= 3 && static_cast<uint8_t>(ptr[0]) == 0xEFu && static_cast<uint8_t>(ptr[1]) == 0xBBu && static_cast<uint8_t>(ptr[2]) == 0xBFu)
		ptr += 3;
	while (ptr < end)
	{
		while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\r' || *ptr == '\n'))
			++ptr;
		if (ptr >= end)
			break;
		if (*ptr == '#')
		{
			while (ptr < end && *ptr != '\n')
				++ptr;
			continue;
		}
		switch (std::tolower(*ptr))
		{
			case 'v':
			case 'f':
			case 'o':
			case 'g':
			case 's':
			case 'u':
			case 'm':
			case 'l':
			case 'p':
				return true;
			default:
				return false;
		}
	}
	return false;
}
const char** COBJMeshFileLoader::getAssociatedFileExtensions() const
{
	static const char* ext[] = { "obj", nullptr };
	return ext;
}
asset::SAssetBundle COBJMeshFileLoader::loadAsset(
    system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params,
    asset::IAssetLoader::IAssetLoaderOverride* _override [[maybe_unused]],
    uint32_t _hierarchyLevel [[maybe_unused]]) {
    if (!_file)
        return {};
    uint64_t faceCount = 0u;
    uint64_t faceFastTokenCount = 0u;
    uint64_t faceFallbackTokenCount = 0u;
    SFileReadTelemetry ioTelemetry = {};
    const long filesize = _file->getSize();
    if (filesize <= 0)
        return {};
    impl::SLoadSession loadSession = {};
    if (!impl::SLoadSession::begin(_params.logger, "OBJ loader", _file, _params.ioPolicy, static_cast<uint64_t>(filesize), true, loadSession))
        return {};
    core::vector<uint8_t> fileContents;
    const auto* fileData = loadSession.mapOrReadWholeFile(fileContents, &ioTelemetry);
    if (!fileData)
        return {};
    const char* const buf = reinterpret_cast<const char*>(fileData);
    const char* const bufEnd = buf + static_cast<size_t>(filesize);
    const char* bufPtr = buf;
    core::vector<hlsl::float32_t3> positions;
    core::vector<hlsl::float32_t3> normals;
    core::vector<hlsl::float32_t2> uvs;
    const size_t estimatedAttributeCount =
        std::max<size_t>(16ull, static_cast<size_t>(filesize) / 32ull);
    positions.reserve(estimatedAttributeCount);
    normals.reserve(estimatedAttributeCount);
    uvs.reserve(estimatedAttributeCount);
    core::vector<hlsl::float32_t3> outPositions;
    core::vector<hlsl::float32_t3> outNormals;
    core::vector<uint8_t> outNormalNeedsGeneration;
    core::vector<hlsl::float32_t2> outUVs;
    std::optional<CPolygonGeometryManipulator::CSmoothNormalAccumulator> smoothNormalAccumulator;
    core::vector<uint32_t> indices;
    core::vector<int32_t> dedupHeadByPos;
    core::vector<Parse::VertexDedupNode> dedupNodes;
    const size_t estimatedOutVertexCount = std::max<size_t>(
        estimatedAttributeCount, static_cast<size_t>(filesize) / 20ull);
    const size_t estimatedOutIndexCount =
        (estimatedOutVertexCount <= (std::numeric_limits<size_t>::max() / 3ull))
            ? (estimatedOutVertexCount * 3ull)
            : std::numeric_limits<size_t>::max();
    const size_t initialOutVertexCapacity =
        std::max<size_t>(1ull, estimatedOutVertexCount);
    const size_t initialOutIndexCapacity =
        (estimatedOutIndexCount == std::numeric_limits<size_t>::max())
            ? 3ull
            : std::max<size_t>(3ull, estimatedOutIndexCount);
    size_t outVertexWriteCount = 0ull;
    size_t outIndexWriteCount = 0ull;
    size_t dedupNodeCount = 0ull;
    struct SDedupHotEntry {
        int32_t pos = -1;
        int32_t uv = -1;
        int32_t normal = -1;
        uint32_t outIndex = 0u;
    };
    const size_t hw = SLoaderRuntimeTuner::resolveHardwareThreads();
    const size_t hardMaxWorkers = SLoaderRuntimeTuner::resolveHardMaxWorkers(
        hw, _params.ioPolicy.runtimeTuning.workerHeadroom);
    SLoaderRuntimeTuningRequest dedupTuningRequest = {};
    dedupTuningRequest.inputBytes = static_cast<uint64_t>(filesize);
    dedupTuningRequest.totalWorkUnits = estimatedOutVertexCount;
    dedupTuningRequest.hardwareThreads = static_cast<uint32_t>(hw);
    dedupTuningRequest.hardMaxWorkers = static_cast<uint32_t>(hardMaxWorkers);
    dedupTuningRequest.targetChunksPerWorker =
        _params.ioPolicy.runtimeTuning.targetChunksPerWorker;
    dedupTuningRequest.sampleData = reinterpret_cast<const uint8_t*>(buf);
    dedupTuningRequest.sampleBytes = SLoaderRuntimeTuner::resolveSampleBytes(
        _params.ioPolicy, static_cast<uint64_t>(filesize));
    const auto dedupTuning =
        SLoaderRuntimeTuner::tune(_params.ioPolicy, dedupTuningRequest);
    const size_t dedupHotSeed = std::max<size_t>(
        16ull, estimatedOutVertexCount /
                   std::max<size_t>(1ull, dedupTuning.workerCount * 8ull));
    const size_t dedupHotEntryCount = std::bit_ceil(dedupHotSeed);
    core::vector<SDedupHotEntry> dedupHotCache(dedupHotEntryCount);
    const size_t dedupHotMask = dedupHotEntryCount - 1ull;
    struct SLoadedGeometry {
        core::smart_refctd_ptr<ICPUPolygonGeometry> geometry = {};
        std::string objectName = {};
        std::string groupName = {};
        uint64_t faceCount = 0ull;
        uint64_t faceFastTokenCount = 0ull;
        uint64_t faceFallbackTokenCount = 0ull;
    };
    core::vector<SLoadedGeometry> loadedGeometries;
    std::string currentObjectName = "default_object";
    std::string currentGroupName = "default_group";
    bool sawObjectDirective = false;
    bool sawGroupDirective = false;
    bool hasProvidedNormals = false;
    bool needsNormalGeneration = false;
    bool hasUVs = false;
    hlsl::shapes::util::AABBAccumulator3<float> parsedAABB =
        hlsl::shapes::util::createAABBAccumulator<float>();
    uint64_t currentFaceCount = 0ull;
    uint64_t currentFaceFastTokenCount = 0ull;
    uint64_t currentFaceFallbackTokenCount = 0ull;
    const auto resetBuilderState = [&]() -> void {
        outPositions.clear();
        outNormals.clear();
        outNormalNeedsGeneration.clear();
        outUVs.clear();
        smoothNormalAccumulator.reset();
        indices.clear();
        dedupNodes.clear();
        outPositions.resize(initialOutVertexCapacity);
        outNormals.resize(initialOutVertexCapacity);
        outNormalNeedsGeneration.resize(initialOutVertexCapacity, 0u);
        outUVs.resize(initialOutVertexCapacity);
        indices.resize(initialOutIndexCapacity);
        dedupHeadByPos.assign(positions.size(), -1);
        dedupNodes.resize(initialOutVertexCapacity);
        outVertexWriteCount = 0ull;
        outIndexWriteCount = 0ull;
        dedupNodeCount = 0ull;
        hasProvidedNormals = false;
        needsNormalGeneration = false;
        hasUVs = false;
        parsedAABB = hlsl::shapes::util::createAABBAccumulator<float>();
        currentFaceCount = 0ull;
        currentFaceFastTokenCount = 0ull;
        currentFaceFallbackTokenCount = 0ull;
        const SDedupHotEntry emptyHotEntry = {};
        std::fill(dedupHotCache.begin(), dedupHotCache.end(), emptyHotEntry);
    };
    const auto finalizeCurrentGeometry = [&]() -> bool {
        if (outVertexWriteCount == 0ull)
            return true;
        outPositions.resize(outVertexWriteCount);
        outNormals.resize(outVertexWriteCount);
        outNormalNeedsGeneration.resize(outVertexWriteCount);
        outUVs.resize(outVertexWriteCount);
        indices.resize(outIndexWriteCount);
        if (needsNormalGeneration) {
            // OBJ smoothing groups are already encoded in the parser-side vertex
            // split corners that must stay sharp become different output vertices
            // even if they share position. We therefore feed the parser-final
            // indexed triangles into a smoothing accumulator and finalize only
            // the normals that were missing in the source.
            if (!smoothNormalAccumulator)
                return false;
            smoothNormalAccumulator->reserveVertices(outVertexWriteCount);
            if (!smoothNormalAccumulator->finalize(
                    std::span<hlsl::float32_t3>(outNormals.data(), outNormals.size()),
                    std::span<const uint8_t>(outNormalNeedsGeneration.data(), outNormalNeedsGeneration.size())))
                return false;
        }
        const size_t outVertexCount = outPositions.size();
        auto geometry = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
        {
            auto view = SGeometryLoaderCommon::createAdoptedView<EF_R32G32B32_SFLOAT>(
                std::move(outPositions));
            if (!view)
                return false;
            geometry->setPositionView(std::move(view));
        }
        const bool hasNormals = hasProvidedNormals || needsNormalGeneration;
        if (hasNormals) {
            auto view = SGeometryLoaderCommon::createAdoptedView<EF_R32G32B32_SFLOAT>(
                std::move(outNormals));
            if (!view)
                return false;
            geometry->setNormalView(std::move(view));
        }
        if (hasUVs) {
            auto view = SGeometryLoaderCommon::createAdoptedView<EF_R32G32_SFLOAT>(
                std::move(outUVs));
            if (!view)
                return false;
            auto* const auxViews = geometry->getAuxAttributeViews();
            auxViews->resize(SOBJPolygonGeometryAuxLayout::UV0 + 1u);
            (*auxViews)[SOBJPolygonGeometryAuxLayout::UV0] = std::move(view);
        }
        if (!indices.empty()) {
            geometry->setIndexing(IPolygonGeometryBase::TriangleList());
            if (outVertexCount <=
                static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1ull) {
                core::vector<uint16_t> indices16(indices.size());
                for (size_t i = 0u; i < indices.size(); ++i)
                    indices16[i] = static_cast<uint16_t>(indices[i]);
                auto view = SGeometryLoaderCommon::createAdoptedView<EF_R16_UINT>(
                    std::move(indices16));
                if (!view)
                    return false;
                geometry->setIndexView(std::move(view));
            } else {
                auto view = SGeometryLoaderCommon::createAdoptedView<EF_R32_UINT>(
                    std::move(indices));
                if (!view)
                    return false;
                geometry->setIndexView(std::move(view));
            }
        } else {
            geometry->setIndexing(IPolygonGeometryBase::PointList());
        }
        if (!_params.loaderFlags.hasAnyFlag(
                IAssetLoader::ELPF_DONT_COMPUTE_CONTENT_HASHES))
            SPolygonGeometryContentHash::computeMissing(geometry.get(),
                                                        _params.ioPolicy);
        if (!parsedAABB.empty())
            geometry->applyAABB(parsedAABB.value);
        else
            CPolygonGeometryManipulator::recomputeAABB(geometry.get());
        loadedGeometries.push_back(SLoadedGeometry{
            .geometry = std::move(geometry),
            .objectName = currentObjectName,
            .groupName = currentGroupName,
            .faceCount = currentFaceCount,
            .faceFastTokenCount = currentFaceFastTokenCount,
            .faceFallbackTokenCount = currentFaceFallbackTokenCount});
        return true;
    };
    resetBuilderState();
    auto allocateOutVertex = [&](uint32_t& outIx) -> bool {
        if (outVertexWriteCount >= outPositions.size()) {
            const size_t newCapacity = std::max<size_t>(outVertexWriteCount + 1ull,
                                                        outPositions.size() * 2ull);
            outPositions.resize(newCapacity);
            outNormals.resize(newCapacity);
            outNormalNeedsGeneration.resize(newCapacity, 0u);
            outUVs.resize(newCapacity);
            if (smoothNormalAccumulator) {
                smoothNormalAccumulator->reserveVertices(newCapacity);
                smoothNormalAccumulator->prepareIdentityGroups(newCapacity);
            }
        }
        if (outVertexWriteCount >
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
            return false;
        outIx = static_cast<uint32_t>(outVertexWriteCount++);
        return true;
    };
    auto appendIndex = [&](const uint32_t value) -> bool {
        if (outIndexWriteCount >= indices.size()) {
            const size_t newCapacity =
                std::max<size_t>(outIndexWriteCount + 1ull, indices.size() * 2ull);
            indices.resize(newCapacity);
        }
        indices[outIndexWriteCount++] = value;
        return true;
    };
    auto allocateDedupNode = [&]() -> int32_t {
        if (dedupNodeCount >= dedupNodes.size()) {
            const size_t newCapacity =
                std::max<size_t>(dedupNodeCount + 1ull, dedupNodes.size() * 2ull);
            dedupNodes.resize(newCapacity);
        }
        if (dedupNodeCount >
            static_cast<size_t>(std::numeric_limits<int32_t>::max()))
            return -1;
        const int32_t ix = static_cast<int32_t>(dedupNodeCount++);
        return ix;
    };
    auto findCornerIndex =
        [&](const int32_t posIx, const int32_t uvIx, const int32_t normalIx,
            const uint32_t dedupSmoothingGroup, uint32_t& outIx) -> bool {
        if (posIx < 0 || static_cast<size_t>(posIx) >= positions.size())
            return false;
        if (static_cast<size_t>(posIx) >= dedupHeadByPos.size())
            dedupHeadByPos.resize(positions.size(), -1);
        int32_t nodeIx = dedupHeadByPos[static_cast<size_t>(posIx)];
        while (nodeIx >= 0) {
            const auto& node = dedupNodes[static_cast<size_t>(nodeIx)];
            if (node.uv == uvIx && node.normal == normalIx &&
                node.smoothingGroup == dedupSmoothingGroup) {
                outIx = node.outIndex;
                return true;
            }
            nodeIx = node.next;
        }
        return false;
    };
    auto materializeCornerIndex =
        [&](const int32_t posIx, const int32_t uvIx, const int32_t normalIx,
            const uint32_t dedupSmoothingGroup, uint32_t& outIx) -> bool {
        if (!allocateOutVertex(outIx))
            return false;
        const int32_t newNodeIx = allocateDedupNode();
        if (newNodeIx < 0)
            return false;
        auto& node = dedupNodes[static_cast<size_t>(newNodeIx)];
        node.uv = uvIx;
        node.normal = normalIx;
        node.smoothingGroup = dedupSmoothingGroup;
        node.outIndex = outIx;
        node.next = dedupHeadByPos[static_cast<size_t>(posIx)];
        dedupHeadByPos[static_cast<size_t>(posIx)] = newNodeIx;
        const auto& srcPos = positions[static_cast<size_t>(posIx)];
        outPositions[static_cast<size_t>(outIx)] = srcPos;
        hlsl::shapes::util::extendAABBAccumulator(parsedAABB, srcPos);
        hlsl::float32_t2 uv(0.f, 0.f);
        if (uvIx >= 0 && static_cast<size_t>(uvIx) < uvs.size()) {
            uv = uvs[static_cast<size_t>(uvIx)];
            hasUVs = true;
        }
        outUVs[static_cast<size_t>(outIx)] = uv;
        hlsl::float32_t3 normal(0.f, 0.f, 0.f);
        if (normalIx >= 0 && static_cast<size_t>(normalIx) < normals.size()) {
            normal = normals[static_cast<size_t>(normalIx)];
            hasProvidedNormals = true;
            outNormalNeedsGeneration[static_cast<size_t>(outIx)] = 0u;
        } else {
            needsNormalGeneration = true;
            outNormalNeedsGeneration[static_cast<size_t>(outIx)] = 1u;
        }
        outNormals[static_cast<size_t>(outIx)] = normal;
        return true;
    };
    auto acquireCornerIndex = [&](const hlsl::int32_t3& idx,
                                  const uint32_t smoothingGroup,
                                  uint32_t& outIx) -> bool {
        const int32_t posIx = idx.x;
        if (posIx < 0 || static_cast<size_t>(posIx) >= positions.size())
            return false;
        const uint32_t dedupSmoothingGroup = idx.z >= 0 ? 0u : smoothingGroup;
        if (findCornerIndex(posIx, idx.y, idx.z, dedupSmoothingGroup, outIx))
            return true;
        return materializeCornerIndex(posIx, idx.y, idx.z, dedupSmoothingGroup,
                                      outIx);
    };
    auto acquireCornerIndexPositiveTriplet = [&](const hlsl::int32_t3& idx,
                                                 uint32_t& outIx) -> bool {
        const uint32_t hotHash = static_cast<uint32_t>(idx.x) * 73856093u ^
                                 static_cast<uint32_t>(idx.y) * 19349663u ^
                                 static_cast<uint32_t>(idx.z) * 83492791u;
        auto& hotEntry = dedupHotCache[static_cast<size_t>(hotHash) & dedupHotMask];
        if (hotEntry.pos == idx.x && hotEntry.uv == idx.y &&
            hotEntry.normal == idx.z) {
            outIx = hotEntry.outIndex;
            return true;
        }
        if (findCornerIndex(idx.x, idx.y, idx.z, 0u, outIx) ||
            materializeCornerIndex(idx.x, idx.y, idx.z, 0u, outIx)) {
            hotEntry.pos = idx.x;
            hotEntry.uv = idx.y;
            hotEntry.normal = idx.z;
            hotEntry.outIndex = outIx;
            return true;
        }
        return false;
    };
    auto acquireCornerIndexPositiveNormal = [&](const hlsl::int32_t3& idx,
                                                uint32_t& outIx) -> bool {
        const uint32_t hotHash = static_cast<uint32_t>(idx.x) * 73856093u ^
                                 static_cast<uint32_t>(idx.z) * 83492791u ^
                                 0x9e3779b9u;
        auto& hotEntry = dedupHotCache[static_cast<size_t>(hotHash) & dedupHotMask];
        if (hotEntry.pos == idx.x && hotEntry.uv == -1 &&
            hotEntry.normal == idx.z) {
            outIx = hotEntry.outIndex;
            return true;
        }
        if (findCornerIndex(idx.x, -1, idx.z, 0u, outIx) ||
            materializeCornerIndex(idx.x, -1, idx.z, 0u, outIx)) {
            hotEntry.pos = idx.x;
            hotEntry.uv = -1;
            hotEntry.normal = idx.z;
            hotEntry.outIndex = outIx;
            return true;
        }
        return false;
    };
    auto acquireTriangleCorners = [&](auto&& acquire, const std::array<hlsl::int32_t3, 3>& triIdx, hlsl::uint32_t3& cornerIx) -> bool {
        return acquire(triIdx[0], cornerIx.x) && acquire(triIdx[1], cornerIx.y) && acquire(triIdx[2], cornerIx.z);
    };
    auto appendTriangle = [&](const hlsl::uint32_t3& cornerIx) -> bool {
        if (!(appendIndex(cornerIx.z) && appendIndex(cornerIx.y) && appendIndex(cornerIx.x)))
            return false;
        if (!needsNormalGeneration)
            return true;
        if (!smoothNormalAccumulator) {
            smoothNormalAccumulator.emplace(CPolygonGeometryManipulator::ESmoothNormalAccumulationMode::AreaWeighted);
            smoothNormalAccumulator->reserveVertices(outVertexWriteCount);
            smoothNormalAccumulator->prepareIdentityGroups(outPositions.size());
        }
        if (outNormalNeedsGeneration[static_cast<size_t>(cornerIx.x)] == 0u &&
            outNormalNeedsGeneration[static_cast<size_t>(cornerIx.y)] == 0u &&
            outNormalNeedsGeneration[static_cast<size_t>(cornerIx.z)] == 0u)
            return true;
        return smoothNormalAccumulator->addPreparedIdentityTriangle(
            cornerIx.z, outPositions[static_cast<size_t>(cornerIx.z)],
            cornerIx.y, outPositions[static_cast<size_t>(cornerIx.y)],
            cornerIx.x, outPositions[static_cast<size_t>(cornerIx.x)]);
    };
    uint32_t currentSmoothingGroup = 0u;
    while (bufPtr < bufEnd) {
        const char* const lineStart = bufPtr;
        const size_t remaining = static_cast<size_t>(bufEnd - lineStart);
        const char* lineTerminator =
            static_cast<const char*>(std::memchr(lineStart, '\n', remaining));
        if (!lineTerminator)
            lineTerminator =
                static_cast<const char*>(std::memchr(lineStart, '\r', remaining));
        if (!lineTerminator)
            lineTerminator = bufEnd;
        const char* lineEnd = lineTerminator;
        if (lineEnd > lineStart && lineEnd[-1] == '\r')
            --lineEnd;
        if (lineStart < lineEnd) {
            const char lineType = std::tolower(*lineStart);
            if (lineType == 'v') {
				auto parseVector = [&](const char* ptr, float* values,
									   const uint32_t count) -> bool {
					for (uint32_t i = 0u; i < count; ++i) {
						while (ptr < lineEnd && Parse::Common::isInlineWhitespace(*ptr))
							++ptr;
						if (ptr >= lineEnd || !Parse::Common::parseNumber(ptr, lineEnd, values[i]))
							return false;
					}
					return true;
                };
                const char subType =
                    ((lineStart + 1) < lineEnd) ? std::tolower(lineStart[1]) : '\0';
                if ((lineStart + 1) < lineEnd && subType == ' ') {
                    hlsl::float32_t3 vec{};
                    if (!parseVector(lineStart + 2, &vec.x, 3u))
                        return {};
                    positions.push_back(vec);
                    dedupHeadByPos.push_back(-1);
                } else if ((lineStart + 2) < lineEnd && subType == 'n' &&
                           Parse::Common::isInlineWhitespace(lineStart[2])) {
                    hlsl::float32_t3 vec{};
                    if (!parseVector(lineStart + 3, &vec.x, 3u))
                        return {};
                    normals.push_back(vec);
                } else if ((lineStart + 2) < lineEnd && subType == 't' &&
                           Parse::Common::isInlineWhitespace(lineStart[2])) {
                    hlsl::float32_t2 vec{};
                    if (!parseVector(lineStart + 3, &vec.x, 2u))
                        return {};
                    vec.y = 1.f - vec.y;
                    uvs.push_back(vec);
                }
            } else if (lineType == 'o' && (lineStart + 1) < lineEnd &&
                       Parse::Common::isInlineWhitespace(lineStart[1])) {
                if (!finalizeCurrentGeometry())
                    return {};
                resetBuilderState();
                currentObjectName =
                    Parse::parseIdentifier(lineStart + 2, lineEnd, "default_object");
                sawObjectDirective = true;
            } else if (lineType == 'g' && (lineStart + 1) < lineEnd &&
                       Parse::Common::isInlineWhitespace(lineStart[1])) {
                if (!finalizeCurrentGeometry())
                    return {};
                resetBuilderState();
                currentGroupName =
                    Parse::parseIdentifier(lineStart + 2, lineEnd, "default_group");
                sawGroupDirective = true;
            } else if (lineType == 's' && (lineStart + 1) < lineEnd &&
                       Parse::Common::isInlineWhitespace(lineStart[1])) {
                Parse::parseSmoothingGroup(lineStart + 2, lineEnd,
                                           currentSmoothingGroup);
            } else if (lineType == 'f' && (lineStart + 1) < lineEnd &&
                       Parse::Common::isInlineWhitespace(lineStart[1])) {
                if (positions.empty())
                    return {};
                ++faceCount;
                ++currentFaceCount;
                const size_t posCount = positions.size();
                const size_t uvCount = uvs.size();
                const size_t normalCount = normals.size();
                const char* triLinePtr = lineStart + 1;
                std::array triIdx = {hlsl::int32_t3(-1, -1, -1),
                                     hlsl::int32_t3(-1, -1, -1),
                                     hlsl::int32_t3(-1, -1, -1)};
                bool triangleFastPath = Parse::parseTrianglePositiveTripletLine(
                    lineStart + 1, lineEnd, triIdx, posCount, uvCount, normalCount);
                bool positiveNormalOnlyFastPath = false;
                if (!triangleFastPath && uvCount == 0u && normalCount != 0u) {
                    triangleFastPath = Parse::parseTrianglePositivePositionNormalLine(
                        lineStart + 1, lineEnd, triIdx, posCount, normalCount);
                    positiveNormalOnlyFastPath = triangleFastPath;
                }
                bool parsedFirstThree = triangleFastPath;
                if (!triangleFastPath) {
                    triLinePtr = lineStart + 1;
                    parsedFirstThree =
                        Parse::parseFaceVertexToken(triLinePtr, lineEnd, triIdx[0],
                                                    posCount, uvCount, normalCount) &&
                        Parse::parseFaceVertexToken(triLinePtr, lineEnd, triIdx[1],
                                                    posCount, uvCount, normalCount) &&
                        Parse::parseFaceVertexToken(triLinePtr, lineEnd, triIdx[2],
                                                    posCount, uvCount, normalCount);
                    triangleFastPath = parsedFirstThree;
                    if (parsedFirstThree) {
                        while (triLinePtr < lineEnd &&
                               Parse::Common::isInlineWhitespace(*triLinePtr))
                            ++triLinePtr;
                        triangleFastPath = (triLinePtr == lineEnd);
                    }
                }
                if (triangleFastPath && !positiveNormalOnlyFastPath) {
                    const bool fullTriplet = std::all_of(
                        triIdx.begin(), triIdx.end(), [](const hlsl::int32_t3& idx) {
                            return hlsl::all(glm::greaterThanEqual(idx, hlsl::int32_t3(0)));
                        });
                    if (!fullTriplet)
                        triangleFastPath = false;
                }
                if (triangleFastPath) {
                    hlsl::uint32_t3 cornerIx = {};
                    if (positiveNormalOnlyFastPath) {
                        if (!acquireTriangleCorners(acquireCornerIndexPositiveNormal, triIdx, cornerIx))
                            return {};
                    } else if (!acquireTriangleCorners(acquireCornerIndexPositiveTriplet, triIdx, cornerIx))
                        return {};
                    faceFastTokenCount += 3u;
                    currentFaceFastTokenCount += 3u;
                    if (!appendTriangle(cornerIx))
                        return {};
                } else {
                    const char* linePtr = lineStart + 1;
                    uint32_t firstCorner = 0u;
                    uint32_t previousCorner = 0u;
                    uint32_t cornerCount = 0u;
                    if (parsedFirstThree) {
                        hlsl::uint32_t3 cornerIx = {};
                        if (!acquireTriangleCorners([&](const hlsl::int32_t3& idx, uint32_t& outIx) { return acquireCornerIndex(idx, currentSmoothingGroup, outIx); }, triIdx, cornerIx))
                            return {};
                        faceFallbackTokenCount += 3u;
                        currentFaceFallbackTokenCount += 3u;
                        if (!appendTriangle(cornerIx))
                            return {};
                        firstCorner = cornerIx.x;
                        previousCorner = cornerIx.z;
                        cornerCount = 3u;
                        linePtr = triLinePtr;
                    }
                    while (linePtr < lineEnd) {
                        while (linePtr < lineEnd &&
                               Parse::Common::isInlineWhitespace(*linePtr))
                            ++linePtr;
                        if (linePtr >= lineEnd)
                            break;
                        hlsl::int32_t3 idx(-1, -1, -1);
                        if (!Parse::parseFaceVertexToken(linePtr, lineEnd, idx, posCount,
                                                         uvCount, normalCount))
                            return {};
                        ++faceFallbackTokenCount;
                        ++currentFaceFallbackTokenCount;
                        uint32_t cornerIx = 0u;
                        if (!acquireCornerIndex(idx, currentSmoothingGroup, cornerIx))
                            return {};
                        if (cornerCount == 0u) {
                            firstCorner = cornerIx;
                            ++cornerCount;
                            continue;
                        }
                        if (cornerCount == 1u) {
                            previousCorner = cornerIx;
                            ++cornerCount;
                            continue;
                        }
                        if (!appendTriangle(hlsl::uint32_t3(firstCorner, previousCorner, cornerIx)))
                            return {};
                        previousCorner = cornerIx;
                        ++cornerCount;
                    }
                }
            }
        }
        if (lineTerminator >= bufEnd)
            bufPtr = bufEnd;
        else if (*lineTerminator == '\r' && (lineTerminator + 1) < bufEnd &&
                 lineTerminator[1] == '\n')
            bufPtr = lineTerminator + 2;
        else
            bufPtr = lineTerminator + 1;
    }
    if (!finalizeCurrentGeometry())
        return {};
    if (loadedGeometries.empty())
        return {};
    uint64_t outVertexCount = 0ull;
    uint64_t outIndexCount = 0ull;
    uint64_t faceFastTokenCountSum = 0ull;
    uint64_t faceFallbackTokenCountSum = 0ull;
    for (const auto& loaded : loadedGeometries) {
        const auto& posView = loaded.geometry->getPositionView();
        outVertexCount +=
            static_cast<uint64_t>(posView ? posView.getElementCount() : 0ull);
        const auto& indexView = loaded.geometry->getIndexView();
        outIndexCount +=
            static_cast<uint64_t>(indexView ? indexView.getElementCount() : 0ull);
        faceFastTokenCountSum += loaded.faceFastTokenCount;
        faceFallbackTokenCountSum += loaded.faceFallbackTokenCount;
    }
    loadSession.logTinyIO(_params.logger, ioTelemetry);
    const bool buildCollections =
        sawObjectDirective || sawGroupDirective || loadedGeometries.size() > 1ull;
    core::vector<core::smart_refctd_ptr<IAsset>> outputAssets;
    uint64_t objectCount = 1ull;
    if (!buildCollections) {
        // Plain OBJ is still just one polygon geometry here.
        outputAssets.push_back(core::smart_refctd_ptr_static_cast<IAsset>(
            std::move(loadedGeometries.front().geometry)));
    } else {
        // Plain OBJ can group many polygon geometries with `o` and `g`, but it
        // still does not define a real scene graph, instancing, or node transforms.
        // Keep that as geometry collections instead of fabricating an ICPUScene on
        // load.
        core::vector<std::string> objectNames;
        core::vector<core::smart_refctd_ptr<ICPUGeometryCollection>>
            objectCollections;
        for (auto& loaded : loadedGeometries) {
            size_t objectIx = objectNames.size();
            for (size_t i = 0ull; i < objectNames.size(); ++i) {
                if (objectNames[i] == loaded.objectName) {
                    objectIx = i;
                    break;
                }
            }
            if (objectIx == objectNames.size()) {
                objectNames.push_back(loaded.objectName);
                auto collection = core::make_smart_refctd_ptr<ICPUGeometryCollection>();
                if (!collection)
                    return {};
                objectCollections.push_back(std::move(collection));
            }
            auto* refs = objectCollections[objectIx]->getGeometries();
            if (!refs)
                return {};
            IGeometryCollection<ICPUBuffer>::SGeometryReference ref = {};
            ref.geometry = core::smart_refctd_ptr_static_cast<IGeometry<ICPUBuffer>>(
                loaded.geometry);
            refs->push_back(std::move(ref));
        }
        outputAssets.reserve(objectCollections.size());
        for (auto& collection : objectCollections)
            outputAssets.push_back(
                core::smart_refctd_ptr_static_cast<IAsset>(std::move(collection)));
        objectCount = outputAssets.size();
    }
    _params.logger.log(
        "OBJ loader stats: file=%s in(v=%llu n=%llu uv=%llu) out(v=%llu idx=%llu "
        "faces=%llu face_fast_tokens=%llu face_fallback_tokens=%llu "
        "geometries=%llu objects=%llu io_reads=%llu io_min_read=%llu "
        "io_avg_read=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
        system::ILogger::ELL_PERFORMANCE, _file->getFileName().string().c_str(),
        static_cast<unsigned long long>(positions.size()),
        static_cast<unsigned long long>(normals.size()),
        static_cast<unsigned long long>(uvs.size()),
        static_cast<unsigned long long>(outVertexCount),
        static_cast<unsigned long long>(outIndexCount),
        static_cast<unsigned long long>(faceCount),
        static_cast<unsigned long long>(faceFastTokenCountSum),
        static_cast<unsigned long long>(faceFallbackTokenCountSum),
        static_cast<unsigned long long>(loadedGeometries.size()),
        static_cast<unsigned long long>(objectCount),
        static_cast<unsigned long long>(ioTelemetry.callCount),
        static_cast<unsigned long long>(ioTelemetry.getMinOrZero()),
        static_cast<unsigned long long>(ioTelemetry.getAvgOrZero()),
        system::to_string(_params.ioPolicy.strategy).c_str(),
        system::to_string(loadSession.ioPlan.strategy).c_str(),
        static_cast<unsigned long long>(loadSession.ioPlan.chunkSizeBytes()), loadSession.ioPlan.reason);
    return SAssetBundle(core::smart_refctd_ptr<IAssetMetadata>(),
                        std::move(outputAssets));
}
}
#endif // _NBL_COMPILE_WITH_OBJ_LOADER_
