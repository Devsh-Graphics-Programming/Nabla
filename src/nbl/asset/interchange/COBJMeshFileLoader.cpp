// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/core/declarations.h"

#include "nbl/asset/IAssetManager.h"
#include "nbl/asset/ICPUGeometryCollection.h"
#include "nbl/asset/interchange/SGeometryContentHash.h"
#include "nbl/asset/interchange/SGeometryLoaderCommon.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "nbl/asset/interchange/SLoaderRuntimeTuning.h"
#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
#include "nbl/builtin/hlsl/shapes/AABBAccumulator.hlsl"
#include "SOBJPolygonGeometryAuxLayout.h"

#ifdef _NBL_COMPILE_WITH_OBJ_LOADER_

#include "nbl/system/IFile.h"

#include "COBJMeshFileLoader.h"

#include <array>
#include <bit>
#include <charconv>
#include <cctype>
#include <cmath>
#include <fast_float/fast_float.h>
#include <string_view>
#include <type_traits>

namespace nbl::asset
{

namespace
{

struct ObjVertexDedupNode
{
    int32_t uv = -1;
    int32_t normal = -1;
    uint32_t smoothingGroup = 0u;
    uint32_t outIndex = 0u;
    int32_t next = -1;
};

inline bool isObjInlineWhitespace(const char c)
{
    return c == ' ' || c == '\t' || c == '\v' || c == '\f';
}

inline bool isObjDigit(const char c)
{
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

inline bool parseObjFloat(const char*& ptr, const char* const end, float& out)
{
    const auto parseResult = fast_float::from_chars(ptr, end, out);
    if (parseResult.ec != std::errc() || parseResult.ptr == ptr)
        return false;
    ptr = parseResult.ptr;
    return true;
}

bool readTextFileWithPolicy(system::IFile* file, char* dst, size_t byteCount, const SResolvedFileIOPolicy& ioPlan, SFileReadTelemetry& ioTelemetry)
{
    return SInterchangeIO::readFileWithPolicy(file, reinterpret_cast<uint8_t*>(dst), 0ull, byteCount, ioPlan, &ioTelemetry);
}

inline bool parseUnsignedObjIndex(const char*& ptr, const char* const end, uint32_t& out)
{
    uint32_t value = 0u;
    const auto parseResult = std::from_chars(ptr, end, value);
    if (parseResult.ec != std::errc() || parseResult.ptr == ptr)
        return false;
    if (value == 0u || value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
        return false;
    ptr = parseResult.ptr;
    out = value;
    return true;
}

inline void parseObjSmoothingGroup(const char* linePtr, const char* const lineEnd, uint32_t& outGroup)
{
    while (linePtr < lineEnd && isObjInlineWhitespace(*linePtr))
        ++linePtr;

    if (linePtr >= lineEnd)
    {
        outGroup = 0u;
        return;
    }

    const char* const tokenStart = linePtr;
    while (linePtr < lineEnd && !isObjInlineWhitespace(*linePtr))
        ++linePtr;
    const std::string_view token(tokenStart, static_cast<size_t>(linePtr - tokenStart));

    if (token.size() == 2u &&
        static_cast<char>(std::tolower(static_cast<unsigned char>(token[0]))) == 'o' &&
        static_cast<char>(std::tolower(static_cast<unsigned char>(token[1]))) == 'n')
    {
        outGroup = 1u;
        return;
    }
    if (token.size() == 3u &&
        static_cast<char>(std::tolower(static_cast<unsigned char>(token[0]))) == 'o' &&
        static_cast<char>(std::tolower(static_cast<unsigned char>(token[1]))) == 'f' &&
        static_cast<char>(std::tolower(static_cast<unsigned char>(token[2]))) == 'f')
    {
        outGroup = 0u;
        return;
    }

    uint32_t value = 0u;
    const auto parseResult = std::from_chars(token.data(), token.data() + token.size(), value);
    outGroup = (parseResult.ec == std::errc() && parseResult.ptr == token.data() + token.size()) ? value : 0u;
}

inline std::string parseObjIdentifier(const char* linePtr, const char* const lineEnd, const std::string_view fallback)
{
    const char* endPtr = lineEnd;
    while (linePtr < lineEnd && isObjInlineWhitespace(*linePtr))
        ++linePtr;
    while (endPtr > linePtr && isObjInlineWhitespace(endPtr[-1]))
        --endPtr;

    if (linePtr >= endPtr)
        return std::string(fallback);
    return std::string(linePtr, static_cast<size_t>(endPtr - linePtr));
}

inline bool parseObjTrianglePositiveTripletLine(const char* const lineStart, const char* const lineEnd, int32_t* idx0, int32_t* idx1, int32_t* idx2, const size_t posCount, const size_t uvCount, const size_t normalCount)
{
    const char* ptr = lineStart;
    int32_t* const out[3] = { idx0, idx1, idx2 };
    for (uint32_t corner = 0u; corner < 3u; ++corner)
    {
        while (ptr < lineEnd && isObjInlineWhitespace(*ptr))
            ++ptr;
        if (ptr >= lineEnd || !isObjDigit(*ptr))
            return false;

        int32_t posIx = -1;
        {
            uint32_t value = 0u;
            if (!parseUnsignedObjIndex(ptr, lineEnd, value))
                return false;
            if (value > posCount)
                return false;
            posIx = static_cast<int32_t>(value - 1u);
        }
        if (ptr >= lineEnd || *ptr != '/')
            return false;
        ++ptr;

        int32_t uvIx = -1;
        {
            uint32_t value = 0u;
            if (!parseUnsignedObjIndex(ptr, lineEnd, value))
                return false;
            if (value > uvCount)
                return false;
            uvIx = static_cast<int32_t>(value - 1u);
        }
        if (ptr >= lineEnd || *ptr != '/')
            return false;
        ++ptr;

        int32_t normalIx = -1;
        {
            uint32_t value = 0u;
            if (!parseUnsignedObjIndex(ptr, lineEnd, value))
                return false;
            if (value > normalCount)
                return false;
            normalIx = static_cast<int32_t>(value - 1u);
        }

        int32_t* const dst = out[corner];
        dst[0] = posIx;
        dst[1] = uvIx;
        dst[2] = normalIx;
    }

    while (ptr < lineEnd && isObjInlineWhitespace(*ptr))
        ++ptr;
    return ptr == lineEnd;
}

inline bool parseSignedObjIndex(const char*& ptr, const char* const end, int32_t& out)
{
    int32_t value = 0;
    const auto parseResult = std::from_chars(ptr, end, value);
    if (parseResult.ec != std::errc() || parseResult.ptr == ptr)
        return false;
    if (value == 0)
        return false;
    ptr = parseResult.ptr;
    out = value;
    return true;
}

inline bool resolveObjIndex(const int32_t rawIndex, const size_t elementCount, int32_t& resolved)
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

inline bool parseObjFaceVertexTokenFast(const char*& linePtr, const char* const lineEnd, int32_t* idx, const size_t posCount, const size_t uvCount, const size_t normalCount)
{
    if (!idx)
        return false;

    while (linePtr < lineEnd && isObjInlineWhitespace(*linePtr))
        ++linePtr;
    if (linePtr >= lineEnd)
        return false;

    idx[0] = -1;
    idx[1] = -1;
    idx[2] = -1;

    const char* ptr = linePtr;
    if (*ptr != '-' && *ptr != '+')
    {
        uint32_t posRaw = 0u;
        if (!parseUnsignedObjIndex(ptr, lineEnd, posRaw))
            return false;
        if (posRaw > posCount)
            return false;
        idx[0] = static_cast<int32_t>(posRaw - 1u);

        if (ptr < lineEnd && *ptr == '/')
        {
            ++ptr;
            if (ptr < lineEnd && *ptr != '/')
            {
                uint32_t uvRaw = 0u;
                if (!parseUnsignedObjIndex(ptr, lineEnd, uvRaw))
                    return false;
                if (uvRaw > uvCount)
                    return false;
                idx[1] = static_cast<int32_t>(uvRaw - 1u);
            }

            if (ptr < lineEnd && *ptr == '/')
            {
                ++ptr;
                if (ptr < lineEnd && !isObjInlineWhitespace(*ptr))
                {
                    uint32_t normalRaw = 0u;
                    if (!parseUnsignedObjIndex(ptr, lineEnd, normalRaw))
                        return false;
                    if (normalRaw > normalCount)
                        return false;
                    idx[2] = static_cast<int32_t>(normalRaw - 1u);
                }
            }
            else if (ptr < lineEnd && !isObjInlineWhitespace(*ptr))
            {
                return false;
            }
        }
        else if (ptr < lineEnd && !isObjInlineWhitespace(*ptr))
        {
            return false;
        }
    }
    else
    {
        int32_t raw = 0;
        if (!parseSignedObjIndex(ptr, lineEnd, raw))
            return false;
        if (!resolveObjIndex(raw, posCount, idx[0]))
            return false;

        if (ptr < lineEnd && *ptr == '/')
        {
            ++ptr;

            if (ptr < lineEnd && *ptr != '/')
            {
                if (!parseSignedObjIndex(ptr, lineEnd, raw))
                    return false;
                if (!resolveObjIndex(raw, uvCount, idx[1]))
                    return false;
            }

            if (ptr < lineEnd && *ptr == '/')
            {
                ++ptr;
                if (ptr < lineEnd && !isObjInlineWhitespace(*ptr))
                {
                    if (!parseSignedObjIndex(ptr, lineEnd, raw))
                        return false;
                    if (!resolveObjIndex(raw, normalCount, idx[2]))
                        return false;
                }
            }
            else if (ptr < lineEnd && !isObjInlineWhitespace(*ptr))
            {
                return false;
            }
        }
        else if (ptr < lineEnd && !isObjInlineWhitespace(*ptr))
        {
            return false;
        }
    }

    if (ptr < lineEnd && !isObjInlineWhitespace(*ptr))
        return false;
    linePtr = ptr;
    return true;
}

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

        switch (static_cast<char>(std::tolower(static_cast<unsigned char>(*ptr))))
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

asset::SAssetBundle COBJMeshFileLoader::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride*, uint32_t)
{
    if (!_file)
        return {};

    uint64_t faceCount = 0u;
    uint64_t faceFastTokenCount = 0u;
    uint64_t faceFallbackTokenCount = 0u;
    SFileReadTelemetry ioTelemetry = {};

    const long filesize = _file->getSize();
    if (filesize <= 0)
        return {};
    const bool fileMappable = core::bitflag<system::IFile::E_CREATE_FLAGS>(_file->getFlags()).hasAnyFlag(system::IFile::ECF_MAPPABLE);
    const auto ioPlan = SResolvedFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(filesize), true, fileMappable);
    if (!ioPlan.isValid())
    {
        _params.logger.log("OBJ loader: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str(), ioPlan.reason);
        return {};
    }

    std::string fileContents = {};
    const char* buf = nullptr;
    if (ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile)
    {
        const auto* constFile = static_cast<const system::IFile*>(_file);
        const auto* mapped = reinterpret_cast<const char*>(constFile->getMappedPointer());
        if (mapped)
        {
            buf = mapped;
            ioTelemetry.account(static_cast<uint64_t>(filesize));
        }
    }
    if (!buf)
    {
        fileContents.resize(static_cast<size_t>(filesize));
        if (!readTextFileWithPolicy(_file, fileContents.data(), fileContents.size(), ioPlan, ioTelemetry))
            return {};
        buf = fileContents.data();
    }

    const char* const bufEnd = buf + static_cast<size_t>(filesize);
    const char* bufPtr = buf;

    core::vector<hlsl::float32_t3> positions;
    core::vector<hlsl::float32_t3> normals;
    core::vector<hlsl::float32_t2> uvs;
    const size_t estimatedAttributeCount = std::max<size_t>(16ull, static_cast<size_t>(filesize) / 32ull);
    positions.reserve(estimatedAttributeCount);
    normals.reserve(estimatedAttributeCount);
    uvs.reserve(estimatedAttributeCount);

    core::vector<hlsl::float32_t3> outPositions;
    core::vector<hlsl::float32_t3> outNormals;
    core::vector<uint8_t> outNormalNeedsGeneration;
    core::vector<hlsl::float32_t2> outUVs;
    core::vector<uint32_t> indices;
    core::vector<int32_t> dedupHeadByPos;
    core::vector<ObjVertexDedupNode> dedupNodes;
    const size_t estimatedOutVertexCount = std::max<size_t>(estimatedAttributeCount, static_cast<size_t>(filesize) / 20ull);
    const size_t estimatedOutIndexCount = (estimatedOutVertexCount <= (std::numeric_limits<size_t>::max() / 3ull)) ? (estimatedOutVertexCount * 3ull) : std::numeric_limits<size_t>::max();
    const size_t initialOutVertexCapacity = std::max<size_t>(1ull, estimatedOutVertexCount);
    const size_t initialOutIndexCapacity = (estimatedOutIndexCount == std::numeric_limits<size_t>::max()) ? 3ull : std::max<size_t>(3ull, estimatedOutIndexCount);
    size_t outVertexWriteCount = 0ull;
    size_t outIndexWriteCount = 0ull;
    size_t dedupNodeCount = 0ull;
    struct SDedupHotEntry
    {
        int32_t pos = -1;
        int32_t uv = -1;
        int32_t normal = -1;
        uint32_t outIndex = 0u;
    };
    const size_t hw = SLoaderRuntimeTuner::resolveHardwareThreads();
    const size_t hardMaxWorkers = SLoaderRuntimeTuner::resolveHardMaxWorkers(hw, _params.ioPolicy.runtimeTuning.workerHeadroom);
    SLoaderRuntimeTuningRequest dedupTuningRequest = {};
    dedupTuningRequest.inputBytes = static_cast<uint64_t>(filesize);
    dedupTuningRequest.totalWorkUnits = estimatedOutVertexCount;
    dedupTuningRequest.hardwareThreads = static_cast<uint32_t>(hw);
    dedupTuningRequest.hardMaxWorkers = static_cast<uint32_t>(hardMaxWorkers);
    dedupTuningRequest.targetChunksPerWorker = _params.ioPolicy.runtimeTuning.targetChunksPerWorker;
    dedupTuningRequest.sampleData = reinterpret_cast<const uint8_t*>(buf);
    dedupTuningRequest.sampleBytes = SLoaderRuntimeTuner::resolveSampleBytes(_params.ioPolicy, static_cast<uint64_t>(filesize));
    const auto dedupTuning = SLoaderRuntimeTuner::tune(_params.ioPolicy, dedupTuningRequest);
    const size_t dedupHotSeed = std::max<size_t>(
        16ull,
        estimatedOutVertexCount / std::max<size_t>(1ull, dedupTuning.workerCount * 8ull));
    const size_t dedupHotEntryCount = std::bit_ceil(dedupHotSeed);
    core::vector<SDedupHotEntry> dedupHotCache(dedupHotEntryCount);
    const size_t dedupHotMask = dedupHotEntryCount - 1ull;

    struct SLoadedGeometry
    {
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
    hlsl::shapes::util::AABBAccumulator3<float> parsedAABB = hlsl::shapes::util::createAABBAccumulator<float>();
    uint64_t currentFaceCount = 0ull;
    uint64_t currentFaceFastTokenCount = 0ull;
    uint64_t currentFaceFallbackTokenCount = 0ull;

    const auto resetBuilderState = [&]() -> void
    {
        outPositions.clear();
        outNormals.clear();
        outNormalNeedsGeneration.clear();
        outUVs.clear();
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

    const auto finalizeCurrentGeometry = [&]() -> bool
    {
        if (outVertexWriteCount == 0ull)
            return true;

        outPositions.resize(outVertexWriteCount);
        outNormals.resize(outVertexWriteCount);
        outNormalNeedsGeneration.resize(outVertexWriteCount);
        outUVs.resize(outVertexWriteCount);
        indices.resize(outIndexWriteCount);

        if (needsNormalGeneration)
        {
            // OBJ smoothing groups are already encoded in the parser-side vertex split
            // corners that must stay sharp become different output vertices even if they share position.
            // This helper works on that final indexed output and fills only normals missing in the source.
            // `createSmoothVertexNormal` is still not enough here even with indexed-view support,
            // because it would also need a "missing only" mode and proper OBJ smoothing-group handling.
            if (!CPolygonGeometryManipulator::generateMissingSmoothNormals(outNormals, outPositions, indices, outNormalNeedsGeneration))
                return false;
        }

        const size_t outVertexCount = outPositions.size();
        auto geometry = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
        {
            auto view = SGeometryLoaderCommon::createAdoptedView<EF_R32G32B32_SFLOAT>(std::move(outPositions));
            if (!view)
                return false;
            geometry->setPositionView(std::move(view));
        }

        const bool hasNormals = hasProvidedNormals || needsNormalGeneration;
        if (hasNormals)
        {
            auto view = SGeometryLoaderCommon::createAdoptedView<EF_R32G32B32_SFLOAT>(std::move(outNormals));
            if (!view)
                return false;
            geometry->setNormalView(std::move(view));
        }

        if (hasUVs)
        {
            auto view = SGeometryLoaderCommon::createAdoptedView<EF_R32G32_SFLOAT>(std::move(outUVs));
            if (!view)
                return false;
            auto* const auxViews = geometry->getAuxAttributeViews();
            auxViews->resize(SOBJPolygonGeometryAuxLayout::UV0 + 1u);
            auxViews->operator[](SOBJPolygonGeometryAuxLayout::UV0) = std::move(view);
        }

        if (!indices.empty())
        {
            geometry->setIndexing(IPolygonGeometryBase::TriangleList());
            if (outVertexCount <= static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1ull)
            {
                core::vector<uint16_t> indices16(indices.size());
                for (size_t i = 0u; i < indices.size(); ++i)
                    indices16[i] = static_cast<uint16_t>(indices[i]);
                auto view = SGeometryLoaderCommon::createAdoptedView<EF_R16_UINT>(std::move(indices16));
                if (!view)
                    return false;
                geometry->setIndexView(std::move(view));
            }
            else
            {
                auto view = SGeometryLoaderCommon::createAdoptedView<EF_R32_UINT>(std::move(indices));
                if (!view)
                    return false;
                geometry->setIndexView(std::move(view));
            }
        }
        else
        {
            geometry->setIndexing(IPolygonGeometryBase::PointList());
        }

        if (!_params.loaderFlags.hasAnyFlag(IAssetLoader::ELPF_DONT_COMPUTE_CONTENT_HASHES))
            SPolygonGeometryContentHash::computeMissing(geometry.get(), _params.ioPolicy);

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
            .faceFallbackTokenCount = currentFaceFallbackTokenCount
        });
        return true;
    };

    resetBuilderState();
    auto allocateOutVertex = [&](uint32_t& outIx) -> bool
    {
        if (outVertexWriteCount >= outPositions.size())
        {
            const size_t newCapacity = std::max<size_t>(outVertexWriteCount + 1ull, outPositions.size() * 2ull);
            outPositions.resize(newCapacity);
            outNormals.resize(newCapacity);
            outNormalNeedsGeneration.resize(newCapacity, 0u);
            outUVs.resize(newCapacity);
        }
        if (outVertexWriteCount > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
            return false;
        outIx = static_cast<uint32_t>(outVertexWriteCount++);
        return true;
    };

    auto appendIndex = [&](const uint32_t value) -> bool
    {
        if (outIndexWriteCount >= indices.size())
        {
            const size_t newCapacity = std::max<size_t>(outIndexWriteCount + 1ull, indices.size() * 2ull);
            indices.resize(newCapacity);
        }
        indices[outIndexWriteCount++] = value;
        return true;
    };

    auto allocateDedupNode = [&]() -> int32_t
    {
        if (dedupNodeCount >= dedupNodes.size())
        {
            const size_t newCapacity = std::max<size_t>(dedupNodeCount + 1ull, dedupNodes.size() * 2ull);
            dedupNodes.resize(newCapacity);
        }
        if (dedupNodeCount > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
            return -1;
        const int32_t ix = static_cast<int32_t>(dedupNodeCount++);
        return ix;
    };

    auto findCornerIndex = [&](const int32_t posIx, const int32_t uvIx, const int32_t normalIx, const uint32_t dedupSmoothingGroup, uint32_t& outIx)->bool
    {
        if (posIx < 0 || static_cast<size_t>(posIx) >= positions.size())
            return false;
        if (static_cast<size_t>(posIx) >= dedupHeadByPos.size())
            dedupHeadByPos.resize(positions.size(), -1);

        int32_t nodeIx = dedupHeadByPos[static_cast<size_t>(posIx)];
        while (nodeIx >= 0)
        {
            const auto& node = dedupNodes[static_cast<size_t>(nodeIx)];
            if (node.uv == uvIx && node.normal == normalIx && node.smoothingGroup == dedupSmoothingGroup)
            {
                outIx = node.outIndex;
                return true;
            }
            nodeIx = node.next;
        }
        return false;
    };

    auto materializeCornerIndex = [&](const int32_t posIx, const int32_t uvIx, const int32_t normalIx, const uint32_t dedupSmoothingGroup, uint32_t& outIx)->bool
    {
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
        if (uvIx >= 0 && static_cast<size_t>(uvIx) < uvs.size())
        {
            uv = uvs[static_cast<size_t>(uvIx)];
            hasUVs = true;
        }
        outUVs[static_cast<size_t>(outIx)] = uv;

        hlsl::float32_t3 normal(0.f, 0.f, 0.f);
        if (normalIx >= 0 && static_cast<size_t>(normalIx) < normals.size())
        {
            normal = normals[static_cast<size_t>(normalIx)];
            hasProvidedNormals = true;
            outNormalNeedsGeneration[static_cast<size_t>(outIx)] = 0u;
        }
        else
        {
            needsNormalGeneration = true;
            outNormalNeedsGeneration[static_cast<size_t>(outIx)] = 1u;
        }
        outNormals[static_cast<size_t>(outIx)] = normal;
        return true;
    };

    auto acquireCornerIndex = [&](const int32_t* idx, const uint32_t smoothingGroup, uint32_t& outIx)->bool
    {
        if (!idx)
            return false;

        const int32_t posIx = idx[0];
        if (posIx < 0 || static_cast<size_t>(posIx) >= positions.size())
            return false;
        const uint32_t dedupSmoothingGroup = (idx[2] >= 0) ? 0u : smoothingGroup;
        if (findCornerIndex(posIx, idx[1], idx[2], dedupSmoothingGroup, outIx))
            return true;
        return materializeCornerIndex(posIx, idx[1], idx[2], dedupSmoothingGroup, outIx);
    };

    auto acquireCornerIndexPositiveTriplet = [&](const int32_t posIx, const int32_t uvIx, const int32_t normalIx, uint32_t& outIx)->bool
    {
        const uint32_t hotHash =
            static_cast<uint32_t>(posIx) * 73856093u ^
            static_cast<uint32_t>(uvIx) * 19349663u ^
            static_cast<uint32_t>(normalIx) * 83492791u;
        auto& hotEntry = dedupHotCache[static_cast<size_t>(hotHash) & dedupHotMask];
        if (hotEntry.pos == posIx && hotEntry.uv == uvIx && hotEntry.normal == normalIx)
        {
            outIx = hotEntry.outIndex;
            return true;
        }

        if (findCornerIndex(posIx, uvIx, normalIx, 0u, outIx) || materializeCornerIndex(posIx, uvIx, normalIx, 0u, outIx))
        {
            hotEntry.pos = posIx;
            hotEntry.uv = uvIx;
            hotEntry.normal = normalIx;
            hotEntry.outIndex = outIx;
            return true;
        }
        return false;
    };

    uint32_t currentSmoothingGroup = 0u;
    while (bufPtr < bufEnd)
    {
            const char* const lineStart = bufPtr;
            const size_t remaining = static_cast<size_t>(bufEnd - lineStart);
            const char* lineTerminator = static_cast<const char*>(std::memchr(lineStart, '\n', remaining));
            if (!lineTerminator)
                lineTerminator = static_cast<const char*>(std::memchr(lineStart, '\r', remaining));
            if (!lineTerminator)
                lineTerminator = bufEnd;

            const char* lineEnd = lineTerminator;
            if (lineEnd > lineStart && lineEnd[-1] == '\r')
                --lineEnd;

            if (lineStart < lineEnd)
            {
                const char lineType = static_cast<char>(std::tolower(static_cast<unsigned char>(*lineStart)));
                if (lineType == 'v')
                {
                    auto parseVector = [&](const char* ptr, float* values, const uint32_t count)->bool
                    {
                        for (uint32_t i = 0u; i < count; ++i)
                        {
                            while (ptr < lineEnd && isObjInlineWhitespace(*ptr))
                                ++ptr;
                            if (ptr >= lineEnd || !parseObjFloat(ptr, lineEnd, values[i]))
                                return false;
                        }
                        return true;
                    };
                    const char subType = ((lineStart + 1) < lineEnd) ? static_cast<char>(std::tolower(static_cast<unsigned char>(lineStart[1]))) : '\0';
                    if ((lineStart + 1) < lineEnd && subType == ' ')
                    {
                        hlsl::float32_t3 vec{};
                        if (!parseVector(lineStart + 2, &vec.x, 3u))
                            return {};
                        positions.push_back(vec);
                        dedupHeadByPos.push_back(-1);
                    }
                    else if ((lineStart + 2) < lineEnd && subType == 'n' && isObjInlineWhitespace(lineStart[2]))
                    {
                        hlsl::float32_t3 vec{};
                        if (!parseVector(lineStart + 3, &vec.x, 3u))
                            return {};
                        normals.push_back(vec);
                    }
                    else if ((lineStart + 2) < lineEnd && subType == 't' && isObjInlineWhitespace(lineStart[2]))
                    {
                        hlsl::float32_t2 vec{};
                        if (!parseVector(lineStart + 3, &vec.x, 2u))
                            return {};
                        vec.y = 1.f - vec.y;
                        uvs.push_back(vec);
                    }
                }
                else if (lineType == 'o' && (lineStart + 1) < lineEnd && isObjInlineWhitespace(lineStart[1]))
                {
                    if (!finalizeCurrentGeometry())
                        return {};
                    resetBuilderState();
                    currentObjectName = parseObjIdentifier(lineStart + 2, lineEnd, "default_object");
                    sawObjectDirective = true;
                }
                else if (lineType == 'g' && (lineStart + 1) < lineEnd && isObjInlineWhitespace(lineStart[1]))
                {
                    if (!finalizeCurrentGeometry())
                        return {};
                    resetBuilderState();
                    currentGroupName = parseObjIdentifier(lineStart + 2, lineEnd, "default_group");
                    sawGroupDirective = true;
                }
                else if (lineType == 's' && (lineStart + 1) < lineEnd && isObjInlineWhitespace(lineStart[1]))
                {
                    parseObjSmoothingGroup(lineStart + 2, lineEnd, currentSmoothingGroup);
                }
                else if (lineType == 'f' && (lineStart + 1) < lineEnd && isObjInlineWhitespace(lineStart[1]))
                {
                    if (positions.empty())
                        return {};
                    ++faceCount;
                    ++currentFaceCount;
                    const size_t posCount = positions.size();
                    const size_t uvCount = uvs.size();
                    const size_t normalCount = normals.size();
                    const char* triLinePtr = lineStart + 1;
                    int32_t triIdx0[3] = { -1, -1, -1 };
                    int32_t triIdx1[3] = { -1, -1, -1 };
                    int32_t triIdx2[3] = { -1, -1, -1 };
                    bool triangleFastPath = parseObjTrianglePositiveTripletLine(lineStart + 1, lineEnd, triIdx0, triIdx1, triIdx2, posCount, uvCount, normalCount);
                    bool parsedFirstThree = triangleFastPath;
                    if (!triangleFastPath)
                    {
                        triLinePtr = lineStart + 1;
                        parsedFirstThree =
                            parseObjFaceVertexTokenFast(triLinePtr, lineEnd, triIdx0, posCount, uvCount, normalCount) &&
                            parseObjFaceVertexTokenFast(triLinePtr, lineEnd, triIdx1, posCount, uvCount, normalCount) &&
                            parseObjFaceVertexTokenFast(triLinePtr, lineEnd, triIdx2, posCount, uvCount, normalCount);
                        triangleFastPath = parsedFirstThree;
                        if (parsedFirstThree)
                        {
                            while (triLinePtr < lineEnd && isObjInlineWhitespace(*triLinePtr))
                                ++triLinePtr;
                            triangleFastPath = (triLinePtr == lineEnd);
                        }
                    }
                    if (triangleFastPath)
                    {
                        const bool fullTriplet =
                            triIdx0[0] >= 0 && triIdx0[1] >= 0 && triIdx0[2] >= 0 &&
                            triIdx1[0] >= 0 && triIdx1[1] >= 0 && triIdx1[2] >= 0 &&
                            triIdx2[0] >= 0 && triIdx2[1] >= 0 && triIdx2[2] >= 0;
                        if (!fullTriplet)
                            triangleFastPath = false;
                    }
                    if (triangleFastPath)
                    {
                        uint32_t c0 = 0u;
                        uint32_t c1 = 0u;
                        uint32_t c2 = 0u;
                        if (!acquireCornerIndexPositiveTriplet(triIdx0[0], triIdx0[1], triIdx0[2], c0))
                            return {};
                        if (!acquireCornerIndexPositiveTriplet(triIdx1[0], triIdx1[1], triIdx1[2], c1))
                            return {};
                        if (!acquireCornerIndexPositiveTriplet(triIdx2[0], triIdx2[1], triIdx2[2], c2))
                            return {};
                        faceFastTokenCount += 3u;
                        currentFaceFastTokenCount += 3u;
                        if (!appendIndex(c2) || !appendIndex(c1) || !appendIndex(c0))
                            return {};
                    }
                    else
                    {
                        const char* linePtr = lineStart + 1;
                        uint32_t firstCorner = 0u;
                        uint32_t previousCorner = 0u;
                        uint32_t cornerCount = 0u;

                        if (parsedFirstThree)
                        {
                            uint32_t c0 = 0u;
                            uint32_t c1 = 0u;
                            uint32_t c2 = 0u;
                            if (!acquireCornerIndex(triIdx0, currentSmoothingGroup, c0))
                                return {};
                            if (!acquireCornerIndex(triIdx1, currentSmoothingGroup, c1))
                                return {};
                            if (!acquireCornerIndex(triIdx2, currentSmoothingGroup, c2))
                                return {};
                            faceFallbackTokenCount += 3u;
                            currentFaceFallbackTokenCount += 3u;
                            if (!appendIndex(c2) || !appendIndex(c1) || !appendIndex(c0))
                                return {};
                            firstCorner = c0;
                            previousCorner = c2;
                            cornerCount = 3u;
                            linePtr = triLinePtr;
                        }

                        while (linePtr < lineEnd)
                        {
                            while (linePtr < lineEnd && isObjInlineWhitespace(*linePtr))
                                ++linePtr;
                            if (linePtr >= lineEnd)
                                break;

                            int32_t idx[3] = { -1, -1, -1 };
                            if (!parseObjFaceVertexTokenFast(linePtr, lineEnd, idx, posCount, uvCount, normalCount))
                                return {};
                            ++faceFallbackTokenCount;
                            ++currentFaceFallbackTokenCount;

                            uint32_t cornerIx = 0u;
                            if (!acquireCornerIndex(idx, currentSmoothingGroup, cornerIx))
                                return {};

                            if (cornerCount == 0u)
                            {
                                firstCorner = cornerIx;
                                ++cornerCount;
                                continue;
                            }

                            if (cornerCount == 1u)
                            {
                                previousCorner = cornerIx;
                                ++cornerCount;
                                continue;
                            }

                            if (!appendIndex(cornerIx) || !appendIndex(previousCorner) || !appendIndex(firstCorner))
                                return {};
                            previousCorner = cornerIx;
                            ++cornerCount;
                        }
                    }
                }
            }

            if (lineTerminator >= bufEnd)
                bufPtr = bufEnd;
            else if (*lineTerminator == '\r' && (lineTerminator + 1) < bufEnd && lineTerminator[1] == '\n')
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
    for (const auto& loaded : loadedGeometries)
    {
        const auto& posView = loaded.geometry->getPositionView();
        outVertexCount += static_cast<uint64_t>(posView ? posView.getElementCount() : 0ull);
        const auto& indexView = loaded.geometry->getIndexView();
        outIndexCount += static_cast<uint64_t>(indexView ? indexView.getElementCount() : 0ull);
        faceFastTokenCountSum += loaded.faceFastTokenCount;
        faceFallbackTokenCountSum += loaded.faceFallbackTokenCount;
    }

    if (SInterchangeIO::isTinyIOTelemetryLikely(ioTelemetry, static_cast<uint64_t>(filesize), _params.ioPolicy))
    {
        _params.logger.log(
            "OBJ loader tiny-io guard: file=%s reads=%llu min=%llu avg=%llu",
            system::ILogger::ELL_WARNING,
            _file->getFileName().string().c_str(),
            static_cast<unsigned long long>(ioTelemetry.callCount),
            static_cast<unsigned long long>(ioTelemetry.getMinOrZero()),
            static_cast<unsigned long long>(ioTelemetry.getAvgOrZero()));
    }

    const bool buildCollections = sawObjectDirective || sawGroupDirective || loadedGeometries.size() > 1ull;
    if (!buildCollections)
    {
        _params.logger.log(
            "OBJ loader stats: file=%s in(v=%llu n=%llu uv=%llu) out(v=%llu idx=%llu faces=%llu face_fast_tokens=%llu face_fallback_tokens=%llu geometries=%llu objects=%llu io_reads=%llu io_min_read=%llu io_avg_read=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
            system::ILogger::ELL_PERFORMANCE,
            _file->getFileName().string().c_str(),
            static_cast<unsigned long long>(positions.size()),
            static_cast<unsigned long long>(normals.size()),
            static_cast<unsigned long long>(uvs.size()),
            static_cast<unsigned long long>(outVertexCount),
            static_cast<unsigned long long>(outIndexCount),
            static_cast<unsigned long long>(faceCount),
            static_cast<unsigned long long>(faceFastTokenCountSum),
            static_cast<unsigned long long>(faceFallbackTokenCountSum),
            static_cast<unsigned long long>(loadedGeometries.size()),
            1ull,
            static_cast<unsigned long long>(ioTelemetry.callCount),
            static_cast<unsigned long long>(ioTelemetry.getMinOrZero()),
            static_cast<unsigned long long>(ioTelemetry.getAvgOrZero()),
            system::to_string(_params.ioPolicy.strategy).c_str(),
            system::to_string(ioPlan.strategy).c_str(),
            static_cast<unsigned long long>(ioPlan.chunkSizeBytes()),
            ioPlan.reason);

        // Plain OBJ is still just one polygon geometry here.
        return SAssetBundle(core::smart_refctd_ptr<IAssetMetadata>(), { core::smart_refctd_ptr_static_cast<IAsset>(std::move(loadedGeometries.front().geometry)) });
    }

    // Plain OBJ can group many polygon geometries with `o` and `g`, but it still does not define
    // a real scene graph, instancing, or node transforms. Keep that as geometry collections instead
    // of fabricating an ICPUScene on load.
    core::vector<std::string> objectNames;
    core::vector<core::smart_refctd_ptr<ICPUGeometryCollection>> objectCollections;
    for (auto& loaded : loadedGeometries)
    {
        size_t objectIx = objectNames.size();
        for (size_t i = 0ull; i < objectNames.size(); ++i)
        {
            if (objectNames[i] == loaded.objectName)
            {
                objectIx = i;
                break;
            }
        }
        if (objectIx == objectNames.size())
        {
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
        ref.geometry = core::smart_refctd_ptr_static_cast<IGeometry<ICPUBuffer>>(loaded.geometry);
        refs->push_back(std::move(ref));
    }

    core::vector<core::smart_refctd_ptr<IAsset>> collectionAssets;
    collectionAssets.reserve(objectCollections.size());
    for (auto& collection : objectCollections)
        collectionAssets.push_back(core::smart_refctd_ptr_static_cast<IAsset>(std::move(collection)));

    _params.logger.log(
        "OBJ loader stats: file=%s in(v=%llu n=%llu uv=%llu) out(v=%llu idx=%llu faces=%llu face_fast_tokens=%llu face_fallback_tokens=%llu geometries=%llu objects=%llu io_reads=%llu io_min_read=%llu io_avg_read=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
        system::ILogger::ELL_PERFORMANCE,
        _file->getFileName().string().c_str(),
        static_cast<unsigned long long>(positions.size()),
        static_cast<unsigned long long>(normals.size()),
        static_cast<unsigned long long>(uvs.size()),
        static_cast<unsigned long long>(outVertexCount),
        static_cast<unsigned long long>(outIndexCount),
        static_cast<unsigned long long>(faceCount),
        static_cast<unsigned long long>(faceFastTokenCountSum),
        static_cast<unsigned long long>(faceFallbackTokenCountSum),
        static_cast<unsigned long long>(loadedGeometries.size()),
        static_cast<unsigned long long>(collectionAssets.size()),
        static_cast<unsigned long long>(ioTelemetry.callCount),
        static_cast<unsigned long long>(ioTelemetry.getMinOrZero()),
        static_cast<unsigned long long>(ioTelemetry.getAvgOrZero()),
        system::to_string(_params.ioPolicy.strategy).c_str(),
        system::to_string(ioPlan.strategy).c_str(),
        static_cast<unsigned long long>(ioPlan.chunkSizeBytes()),
        ioPlan.reason);

    return SAssetBundle(core::smart_refctd_ptr<IAssetMetadata>(), std::move(collectionAssets));
}

}

#endif // _NBL_COMPILE_WITH_OBJ_LOADER_
