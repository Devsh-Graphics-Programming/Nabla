// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/core/declarations.h"

#include "nbl/asset/IAssetManager.h"
#include "nbl/asset/utils/CPolygonGeometryManipulator.h"

#ifdef _NBL_COMPILE_WITH_OBJ_LOADER_

#include "nbl/system/IFile.h"

#include "COBJMeshFileLoader.h"

#include <algorithm>
#include <boost/unordered/unordered_flat_map.hpp>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fast_float/fast_float.h>
#include <limits>
#include <type_traits>

namespace nbl::asset
{

namespace
{

struct ObjVertexKey
{
    int32_t pos;
    int32_t uv;
    int32_t normal;

    bool operator==(const ObjVertexKey& other) const
    {
        return pos == other.pos && uv == other.uv && normal == other.normal;
    }
};

struct ObjVertexKeyHash
{
    size_t operator()(const ObjVertexKey& key) const noexcept
    {
        const uint32_t p = static_cast<uint32_t>(key.pos);
        const uint32_t t = static_cast<uint32_t>(key.uv);
        const uint32_t n = static_cast<uint32_t>(key.normal);
        return static_cast<size_t>((p * 73856093u) ^ (t * 19349663u) ^ (n * 83492791u));
    }
};

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

using Float3 = hlsl::float32_t3;
using Float2 = hlsl::float32_t2;

static_assert(sizeof(Float3) == sizeof(float) * 3ull);
static_assert(sizeof(Float2) == sizeof(float) * 2ull);

void extendAABB(hlsl::shapes::AABB<3, hlsl::float32_t>& aabb, bool& hasAABB, const Float3& p)
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

template<typename T>
IGeometry<ICPUBuffer>::SDataView createAdoptedView(core::vector<T>&& data, const E_FORMAT format)
{
    if (data.empty())
        return {};

    auto backer = core::make_smart_refctd_ptr<core::adoption_memory_resource<core::vector<T>>>(std::move(data));
    auto& storage = backer->getBacker();
    auto* const ptr = storage.data();
    const size_t byteCount = storage.size() * sizeof(T);
    auto buffer = ICPUBuffer::create({ { byteCount }, ptr, core::smart_refctd_ptr<core::refctd_memory_resource>(std::move(backer)), alignof(T) }, core::adopt_memory);
    if (!buffer)
        return {};

    IGeometry<ICPUBuffer>::SDataView view = {
        .composed = {
            .stride = sizeof(T),
            .format = format,
            .rangeFormat = IGeometryBase::getMatchingAABBFormat(format)
        },
        .src = {
            .offset = 0u,
            .size = byteCount,
            .buffer = std::move(buffer)
        }
    };
    return view;
}

bool readTextFileWithPolicy(system::IFile* file, char* dst, size_t byteCount, const SResolvedFileIOPolicy& ioPlan, double& ioMs, SFileReadTelemetry& ioTelemetry)
{
    if (!file || !dst)
        return false;

    using clock_t = std::chrono::high_resolution_clock;
    const auto ioStart = clock_t::now();
    size_t bytesRead = 0ull;
    switch (ioPlan.strategy)
    {
        case SResolvedFileIOPolicy::Strategy::WholeFile:
        {
            system::IFile::success_t success;
            file->read(success, dst, 0ull, byteCount);
            if (!success || success.getBytesProcessed() != byteCount)
                return false;
            bytesRead = byteCount;
            ioTelemetry.account(success.getBytesProcessed());
            break;
        }
        case SResolvedFileIOPolicy::Strategy::Chunked:
        default:
        {
            while (bytesRead < byteCount)
            {
                const size_t toRead = static_cast<size_t>(std::min<uint64_t>(ioPlan.chunkSizeBytes, byteCount - bytesRead));
                system::IFile::success_t success;
                file->read(success, dst + bytesRead, bytesRead, toRead);
                if (!success)
                    return false;
                const size_t processed = success.getBytesProcessed();
                if (processed == 0ull)
                    return false;
                ioTelemetry.account(processed);
                bytesRead += processed;
            }
            break;
        }
    }
    ioMs = std::chrono::duration<double, std::milli>(clock_t::now() - ioStart).count();
    return bytesRead == byteCount;
}

const char* goFirstWord(const char* buf, const char* const bufEnd, bool acrossNewlines = true)
{
    if (acrossNewlines)
        while ((buf != bufEnd) && core::isspace(*buf))
            ++buf;
    else
        while ((buf != bufEnd) && core::isspace(*buf) && (*buf != '\n'))
            ++buf;

    return buf;
}

const char* goNextWord(const char* buf, const char* const bufEnd, bool acrossNewlines = true)
{
    while ((buf != bufEnd) && !core::isspace(*buf))
        ++buf;

    return goFirstWord(buf, bufEnd, acrossNewlines);
}

const char* goNextLine(const char* buf, const char* const bufEnd)
{
    while (buf != bufEnd)
    {
        if (*buf == '\n' || *buf == '\r')
            break;
        ++buf;
    }
    return goFirstWord(buf, bufEnd);
}

bool parseFloatToken(const char*& ptr, const char* const end, float& out)
{
    const auto parseResult = fast_float::from_chars(ptr, end, out);
    if (parseResult.ec == std::errc() && parseResult.ptr != ptr)
    {
        ptr = parseResult.ptr;
        return true;
    }

    char* fallbackEnd = nullptr;
    out = std::strtof(ptr, &fallbackEnd);
    if (!fallbackEnd || fallbackEnd == ptr)
        return false;
    ptr = fallbackEnd;
    return true;
}

const char* readVec3(const char* bufPtr, float vec[3], const char* const bufEnd)
{
    bufPtr = goNextWord(bufPtr, bufEnd, false);
    for (uint32_t i = 0u; i < 3u; ++i)
    {
        if (bufPtr >= bufEnd)
            return bufPtr;

        if (!parseFloatToken(bufPtr, bufEnd, vec[i]))
            return bufPtr;

        while (bufPtr < bufEnd && core::isspace(*bufPtr) && *bufPtr != '\n' && *bufPtr != '\r')
            ++bufPtr;
    }

    return bufPtr;
}

const char* readUV(const char* bufPtr, float vec[2], const char* const bufEnd)
{
    bufPtr = goNextWord(bufPtr, bufEnd, false);
    for (uint32_t i = 0u; i < 2u; ++i)
    {
        if (bufPtr >= bufEnd)
            return bufPtr;

        if (!parseFloatToken(bufPtr, bufEnd, vec[i]))
            return bufPtr;

        while (bufPtr < bufEnd && core::isspace(*bufPtr) && *bufPtr != '\n' && *bufPtr != '\r')
            ++bufPtr;
    }

    vec[1] = 1.f - vec[1];
    return bufPtr;
}

bool parseSignedObjIndex(const char*& ptr, const char* const end, int32_t& out)
{
    const char* parseStart = ptr;
    if (ptr >= end)
        return false;

    int64_t value = 0;
    const auto parseResult = std::from_chars(ptr, end, value, 10);
    if (!(parseResult.ec == std::errc() && parseResult.ptr != ptr))
    {
        char* fallbackEnd = nullptr;
        value = std::strtoll(parseStart, &fallbackEnd, 10);
        if (!fallbackEnd || fallbackEnd == parseStart || fallbackEnd > end)
            return false;
        ptr = fallbackEnd;
    }
    else
    {
        ptr = parseResult.ptr;
    }
    if (value == 0)
        return false;
    if (value < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        return false;

    out = static_cast<int32_t>(value);
    return true;
}

bool resolveObjIndex(const int32_t rawIndex, const size_t elementCount, int32_t& resolved)
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

bool parseObjFaceVertexToken(const char* tokenBegin, const char* tokenEnd, int32_t* idx, const size_t posCount, const size_t uvCount, const size_t normalCount)
{
    if (!tokenBegin || !idx || tokenBegin >= tokenEnd)
        return false;

    idx[0] = -1;
    idx[1] = -1;
    idx[2] = -1;

    const char* ptr = tokenBegin;
    int32_t raw = 0;
    if (!parseSignedObjIndex(ptr, tokenEnd, raw))
        return false;
    if (!resolveObjIndex(raw, posCount, idx[0]))
        return false;

    if (ptr >= tokenEnd)
        return true;
    if (*ptr != '/')
        return false;
    ++ptr;

    if (ptr < tokenEnd && *ptr != '/')
    {
        if (!parseSignedObjIndex(ptr, tokenEnd, raw))
            return false;
        if (!resolveObjIndex(raw, uvCount, idx[1]))
            return false;
    }

    if (ptr >= tokenEnd)
        return true;
    if (*ptr != '/')
        return false;
    ++ptr;

    if (ptr >= tokenEnd)
        return true;
    if (!parseSignedObjIndex(ptr, tokenEnd, raw))
        return false;
    if (!resolveObjIndex(raw, normalCount, idx[2]))
        return false;

    return ptr == tokenEnd;
}

}

COBJMeshFileLoader::COBJMeshFileLoader(IAssetManager* _manager)
{
    (void)_manager;
}

COBJMeshFileLoader::~COBJMeshFileLoader() = default;

bool COBJMeshFileLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
{
    (void)logger;
    if (!_file)
        return false;
    system::IFile::success_t succ;
    char firstChar = 0;
    _file->read(succ, &firstChar, 0ull, sizeof(firstChar));
    return succ && (firstChar == '#' || firstChar == 'v');
}

const char** COBJMeshFileLoader::getAssociatedFileExtensions() const
{
    static const char* ext[] = { "obj", nullptr };
    return ext;
}

asset::SAssetBundle COBJMeshFileLoader::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    (void)_override;
    (void)_hierarchyLevel;

    if (!_file)
        return {};

    using clock_t = std::chrono::high_resolution_clock;
    const auto totalStart = clock_t::now();
    double ioMs = 0.0;
    double parseMs = 0.0;
    double buildMs = 0.0;
    double hashMs = 0.0;
    double aabbMs = 0.0;
    double parseVms = 0.0;
    double parseVNms = 0.0;
    double parseVTms = 0.0;
    double parseFaceMs = 0.0;
    double dedupMs = 0.0;
    double emitMs = 0.0;
    uint64_t faceCount = 0u;
    uint64_t faceFastTokenCount = 0u;
    uint64_t faceFallbackTokenCount = 0u;
    SFileReadTelemetry ioTelemetry = {};

    const long filesize = _file->getSize();
    if (filesize <= 0)
        return {};
    const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(filesize), true);
    if (!ioPlan.valid)
    {
        _params.logger.log("OBJ loader: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str(), ioPlan.reason);
        return {};
    }

    std::string fileContents;
    fileContents.resize(static_cast<size_t>(filesize));
    if (!readTextFileWithPolicy(_file, fileContents.data(), fileContents.size(), ioPlan, ioMs, ioTelemetry))
        return {};

    const char* const buf = fileContents.data();
    const char* const bufEnd = buf + static_cast<size_t>(filesize);
    const char* bufPtr = buf;

    core::vector<Float3> positions;
    core::vector<Float3> normals;
    core::vector<Float2> uvs;

    core::vector<Float3> outPositions;
    core::vector<Float3> outNormals;
    core::vector<Float2> outUVs;
    core::vector<uint32_t> indices;
    boost::unordered_flat_map<ObjVertexKey, uint32_t, ObjVertexKeyHash> vtxMap;

    bool hasNormals = false;
    bool hasUVs = false;
    hlsl::shapes::AABB<3, hlsl::float32_t> parsedAABB = hlsl::shapes::AABB<3, hlsl::float32_t>::create();
    bool hasParsedAABB = false;
    core::vector<uint32_t> faceCorners;
    faceCorners.reserve(16ull);

    const auto parseStart = clock_t::now();
    while (bufPtr != bufEnd)
    {
        switch (bufPtr[0])
        {
            case 'v':
                switch (bufPtr[1])
                {
                    case ' ':
                    {
                        const auto t = clock_t::now();
                        Float3 vec{};
                        bufPtr = readVec3(bufPtr, &vec.x, bufEnd);
                        positions.push_back(vec);
                        parseVms += std::chrono::duration<double, std::milli>(clock_t::now() - t).count();
                    }
                    break;
                    case 'n':
                    {
                        const auto t = clock_t::now();
                        Float3 vec{};
                        bufPtr = readVec3(bufPtr, &vec.x, bufEnd);
                        normals.push_back(vec);
                        parseVNms += std::chrono::duration<double, std::milli>(clock_t::now() - t).count();
                    }
                    break;
                    case 't':
                    {
                        const auto t = clock_t::now();
                        Float2 vec{};
                        bufPtr = readUV(bufPtr, &vec.x, bufEnd);
                        uvs.push_back(vec);
                        parseVTms += std::chrono::duration<double, std::milli>(clock_t::now() - t).count();
                    }
                    break;
                    default:
                        break;
                }
                break;
            case 'f':
            {
                if (positions.empty())
                    return {};
                ++faceCount;
                if (faceCount == 1u)
                {
                    vtxMap.reserve(positions.size() * 4ull);
                    indices.reserve(positions.size() * 6ull);
                }

                const char* endPtr = bufPtr;
                while (endPtr != bufEnd && *endPtr != '\n' && *endPtr != '\r')
                    ++endPtr;

                faceCorners.clear();

                const char* linePtr = goNextWord(bufPtr, endPtr);
                while (linePtr < endPtr)
                {
                    const auto tokenParseStart = clock_t::now();
                    int32_t idx[3] = { -1, -1, -1 };
                    const char* tokenEnd = linePtr;
                    while (tokenEnd < endPtr && !core::isspace(*tokenEnd))
                        ++tokenEnd;
                    if (!parseObjFaceVertexToken(linePtr, tokenEnd, idx, positions.size(), uvs.size(), normals.size()))
                        return {};
                    ++faceFastTokenCount;

                    if (idx[0] < 0 || static_cast<size_t>(idx[0]) >= positions.size())
                        return {};
                    parseFaceMs += std::chrono::duration<double, std::milli>(clock_t::now() - tokenParseStart).count();

                    const auto dedupStart = clock_t::now();
                    ObjVertexKey key = { idx[0], idx[1], idx[2] };
                    const uint32_t candidateIndex = static_cast<uint32_t>(outPositions.size());
                    auto [it, inserted] = vtxMap.try_emplace(key, candidateIndex);
                    uint32_t outIx = it->second;
                    if (inserted)
                    {
                        if (outPositions.empty())
                        {
                            const size_t estimatedVertexCount = positions.size() <= (std::numeric_limits<size_t>::max() / 4ull) ? positions.size() * 4ull : positions.size();
                            outPositions.reserve(estimatedVertexCount);
                            outNormals.reserve(estimatedVertexCount);
                            outUVs.reserve(estimatedVertexCount);
                        }
                        const auto& srcPos = positions[idx[0]];
                        outPositions.push_back(srcPos);
                        extendAABB(parsedAABB, hasParsedAABB, srcPos);

                        Float2 uv(0.f, 0.f);
                        if (idx[1] >= 0 && static_cast<size_t>(idx[1]) < uvs.size())
                        {
                            uv = uvs[idx[1]];
                            hasUVs = true;
                        }
                        outUVs.push_back(uv);

                        Float3 normal(0.f, 0.f, 1.f);
                        if (idx[2] >= 0 && static_cast<size_t>(idx[2]) < normals.size())
                        {
                            normal = normals[idx[2]];
                            hasNormals = true;
                        }
                        outNormals.push_back(normal);
                    }
                    dedupMs += std::chrono::duration<double, std::milli>(clock_t::now() - dedupStart).count();

                    faceCorners.push_back(outIx);

                    linePtr = goFirstWord(tokenEnd, endPtr, false);
                }

                const auto emitStart = clock_t::now();
                for (uint32_t i = 1u; i + 1u < faceCorners.size(); ++i)
                {
                    indices.push_back(faceCorners[i + 1]);
                    indices.push_back(faceCorners[i]);
                    indices.push_back(faceCorners[0]);
                }
                emitMs += std::chrono::duration<double, std::milli>(clock_t::now() - emitStart).count();
            }
            break;
            default:
                break;
        }

        bufPtr = goNextLine(bufPtr, bufEnd);
    }
    parseMs = std::chrono::duration<double, std::milli>(clock_t::now() - parseStart).count();
    const double parseScanMs = std::max(0.0, parseMs - (parseVms + parseVNms + parseVTms + parseFaceMs + dedupMs + emitMs));

    if (outPositions.empty())
        return {};

    const size_t outVertexCount = outPositions.size();
    const size_t outIndexCount = indices.size();
    const auto buildStart = clock_t::now();
    auto geometry = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
    {
        auto view = createAdoptedView(std::move(outPositions), EF_R32G32B32_SFLOAT);
        if (!view)
            return {};
        geometry->setPositionView(std::move(view));
    }

    if (hasNormals)
    {
        auto view = createAdoptedView(std::move(outNormals), EF_R32G32B32_SFLOAT);
        if (!view)
            return {};
        geometry->setNormalView(std::move(view));
    }

    if (hasUVs)
    {
        auto view = createAdoptedView(std::move(outUVs), EF_R32G32_SFLOAT);
        if (!view)
            return {};
        geometry->getAuxAttributeViews()->push_back(std::move(view));
    }

    if (!indices.empty())
    {
        geometry->setIndexing(IPolygonGeometryBase::TriangleList());
        if (outVertexCount <= static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1ull)
        {
            core::vector<uint16_t> indices16(indices.size());
            for (size_t i = 0u; i < indices.size(); ++i)
                indices16[i] = static_cast<uint16_t>(indices[i]);
            auto view = createAdoptedView(std::move(indices16), EF_R16_UINT);
            if (!view)
                return {};
            geometry->setIndexView(std::move(view));
        }
        else
        {
            auto view = createAdoptedView(std::move(indices), EF_R32_UINT);
            if (!view)
                return {};
            geometry->setIndexView(std::move(view));
        }
    }
    else
    {
        geometry->setIndexing(IPolygonGeometryBase::PointList());
    }
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
    if (
        static_cast<uint64_t>(filesize) > (1ull << 20) &&
        (
            ioTelemetry.getAvgOrZero() < 1024ull ||
            (ioTelemetry.getMinOrZero() < 64ull && ioTelemetry.callCount > 1024ull)
        )
    )
    {
        _params.logger.log(
            "OBJ loader tiny-io guard: file=%s reads=%llu min=%llu avg=%llu",
            system::ILogger::ELL_WARNING,
            _file->getFileName().string().c_str(),
            static_cast<unsigned long long>(ioTelemetry.callCount),
            static_cast<unsigned long long>(ioTelemetry.getMinOrZero()),
            static_cast<unsigned long long>(ioTelemetry.getAvgOrZero()));
    }
    _params.logger.log(
        "OBJ loader perf: file=%s total=%.3f ms io=%.3f parse=%.3f parse_scan=%.3f parse_v=%.3f parse_vn=%.3f parse_vt=%.3f parse_f=%.3f dedup=%.3f emit=%.3f build=%.3f hash=%.3f aabb=%.3f in(v=%llu n=%llu uv=%llu) out(v=%llu idx=%llu faces=%llu face_fast_tokens=%llu face_fallback_tokens=%llu io_reads=%llu io_min_read=%llu io_avg_read=%llu) io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
        system::ILogger::ELL_PERFORMANCE,
        _file->getFileName().string().c_str(),
        totalMs,
        ioMs,
        parseMs,
        parseScanMs,
        parseVms,
        parseVNms,
        parseVTms,
        parseFaceMs,
        dedupMs,
        emitMs,
        buildMs,
        hashMs,
        aabbMs,
        static_cast<unsigned long long>(positions.size()),
        static_cast<unsigned long long>(normals.size()),
        static_cast<unsigned long long>(uvs.size()),
        static_cast<unsigned long long>(outVertexCount),
        static_cast<unsigned long long>(outIndexCount),
        static_cast<unsigned long long>(faceCount),
        static_cast<unsigned long long>(faceFastTokenCount),
        static_cast<unsigned long long>(faceFallbackTokenCount),
        static_cast<unsigned long long>(ioTelemetry.callCount),
        static_cast<unsigned long long>(ioTelemetry.getMinOrZero()),
        static_cast<unsigned long long>(ioTelemetry.getAvgOrZero()),
        toString(_params.ioPolicy.strategy),
        toString(ioPlan.strategy),
        static_cast<unsigned long long>(ioPlan.chunkSizeBytes),
        ioPlan.reason);

    return SAssetBundle(core::smart_refctd_ptr<IAssetMetadata>(), { std::move(geometry) });
}

}

#endif // _NBL_COMPILE_WITH_OBJ_LOADER_
