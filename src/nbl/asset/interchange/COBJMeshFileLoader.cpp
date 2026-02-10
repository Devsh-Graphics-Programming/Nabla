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

struct ObjVertexDedupNode
{
    int32_t uv = -1;
    int32_t normal = -1;
    uint32_t outIndex = 0u;
    int32_t next = -1;
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

NBL_FORCE_INLINE bool isObjInlineWhitespace(const char c)
{
    return c == ' ' || c == '\t' || c == '\v' || c == '\f';
}

NBL_FORCE_INLINE bool isObjDigit(const char c)
{
    return c >= '0' && c <= '9';
}

NBL_FORCE_INLINE bool parseObjFloat(const char*& ptr, const char* const end, float& out)
{
    const char* const start = ptr;
    if (start >= end)
        return false;

    const char* p = start;
    bool negative = false;
    if (*p == '-' || *p == '+')
    {
        negative = (*p == '-');
        ++p;
        if (p >= end)
            return false;
    }

    if (*p == '.' || !isObjDigit(*p))
    {
        const auto parseResult = fast_float::from_chars(start, end, out);
        if (parseResult.ec == std::errc() && parseResult.ptr != start)
        {
            ptr = parseResult.ptr;
            return true;
        }
        return false;
    }

    uint64_t integerPart = 0ull;
    while (p < end && isObjDigit(*p))
    {
        integerPart = integerPart * 10ull + static_cast<uint64_t>(*p - '0');
        ++p;
    }

    double value = static_cast<double>(integerPart);
    if (p < end && *p == '.')
    {
        const char* const dot = p;
        if ((dot + 7) <= end)
        {
            const char d0 = dot[1];
            const char d1 = dot[2];
            const char d2 = dot[3];
            const char d3 = dot[4];
            const char d4 = dot[5];
            const char d5 = dot[6];
            if (
                isObjDigit(d0) && isObjDigit(d1) && isObjDigit(d2) &&
                isObjDigit(d3) && isObjDigit(d4) && isObjDigit(d5)
            )
            {
                const bool hasNext = (dot + 7) < end;
                const char next = hasNext ? dot[7] : '\0';
                if ((!hasNext || !isObjDigit(next)) && (!hasNext || (next != 'e' && next != 'E')))
                {
                    const uint32_t frac =
                        static_cast<uint32_t>(d0 - '0') * 100000u +
                        static_cast<uint32_t>(d1 - '0') * 10000u +
                        static_cast<uint32_t>(d2 - '0') * 1000u +
                        static_cast<uint32_t>(d3 - '0') * 100u +
                        static_cast<uint32_t>(d4 - '0') * 10u +
                        static_cast<uint32_t>(d5 - '0');
                    value += static_cast<double>(frac) * 1e-6;
                    p = dot + 7;
                    out = static_cast<float>(negative ? -value : value);
                    ptr = p;
                    return true;
                }
            }
        }

        static constexpr double InvPow10[] = {
            1.0,
            1e-1, 1e-2, 1e-3, 1e-4, 1e-5,
            1e-6, 1e-7, 1e-8, 1e-9, 1e-10,
            1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
            1e-16, 1e-17, 1e-18
        };
        ++p;
        uint64_t fractionPart = 0ull;
        uint32_t fractionDigits = 0u;
        while (p < end && isObjDigit(*p))
        {
            if (fractionDigits >= (std::size(InvPow10) - 1u))
            {
                const auto parseResult = fast_float::from_chars(start, end, out);
                if (parseResult.ec == std::errc() && parseResult.ptr != start)
                {
                    ptr = parseResult.ptr;
                    return true;
                }
                return false;
            }
            fractionPart = fractionPart * 10ull + static_cast<uint64_t>(*p - '0');
            ++fractionDigits;
            ++p;
        }
        value += static_cast<double>(fractionPart) * InvPow10[fractionDigits];
    }

    if (p < end && (*p == 'e' || *p == 'E'))
    {
        const auto parseResult = fast_float::from_chars(start, end, out);
        if (parseResult.ec == std::errc() && parseResult.ptr != start)
        {
            ptr = parseResult.ptr;
            return true;
        }
        return false;
    }

    out = static_cast<float>(negative ? -value : value);
    ptr = p;
    return true;
}

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

void objRecomputeContentHashes(ICPUPolygonGeometry* geometry)
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

    for (auto& buffer : buffers)
        buffer->setContentHash(buffer->computeContentHash());
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

NBL_FORCE_INLINE bool parseUnsignedObjIndex(const char*& ptr, const char* const end, uint32_t& out)
{
    if (ptr >= end || !isObjDigit(*ptr))
        return false;

    uint64_t value = 0ull;
    while (ptr < end && isObjDigit(*ptr))
    {
        value = value * 10ull + static_cast<uint64_t>(*ptr - '0');
        ++ptr;
    }
    if (value == 0ull || value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
        return false;

    out = static_cast<uint32_t>(value);
    return true;
}

NBL_FORCE_INLINE bool parseObjFaceTokenPositiveTriplet(const char*& ptr, const char* const end, int32_t* idx, const size_t posCount, const size_t uvCount, const size_t normalCount)
{
    while (ptr < end && isObjInlineWhitespace(*ptr))
        ++ptr;
    if (ptr >= end || !isObjDigit(*ptr))
        return false;

    uint32_t posRaw = 0u;
    if (!parseUnsignedObjIndex(ptr, end, posRaw))
        return false;
    if (posRaw > posCount)
        return false;

    if (ptr >= end || *ptr != '/')
        return false;
    ++ptr;

    uint32_t uvRaw = 0u;
    if (!parseUnsignedObjIndex(ptr, end, uvRaw))
        return false;
    if (uvRaw > uvCount)
        return false;

    if (ptr >= end || *ptr != '/')
        return false;
    ++ptr;

    uint32_t normalRaw = 0u;
    if (!parseUnsignedObjIndex(ptr, end, normalRaw))
        return false;
    if (normalRaw > normalCount)
        return false;

    idx[0] = static_cast<int32_t>(posRaw - 1u);
    idx[1] = static_cast<int32_t>(uvRaw - 1u);
    idx[2] = static_cast<int32_t>(normalRaw - 1u);
    return true;
}

NBL_FORCE_INLINE bool parseObjTrianglePositiveTripletLine(const char* const lineStart, const char* const lineEnd, int32_t* idx0, int32_t* idx1, int32_t* idx2, const size_t posCount, const size_t uvCount, const size_t normalCount)
{
    const char* ptr = lineStart;
    if (!parseObjFaceTokenPositiveTriplet(ptr, lineEnd, idx0, posCount, uvCount, normalCount))
        return false;
    if (!parseObjFaceTokenPositiveTriplet(ptr, lineEnd, idx1, posCount, uvCount, normalCount))
        return false;
    if (!parseObjFaceTokenPositiveTriplet(ptr, lineEnd, idx2, posCount, uvCount, normalCount))
        return false;

    while (ptr < lineEnd && isObjInlineWhitespace(*ptr))
        ++ptr;
    return ptr == lineEnd;
}

NBL_FORCE_INLINE bool parseSignedObjIndex(const char*& ptr, const char* const end, int32_t& out)
{
    if (ptr >= end)
        return false;

    bool negative = false;
    if (*ptr == '-')
    {
        negative = true;
        ++ptr;
    }
    else if (*ptr == '+')
    {
        ++ptr;
    }

    if (ptr >= end || !isObjDigit(*ptr))
        return false;

    int64_t value = 0;
    while (ptr < end && isObjDigit(*ptr))
    {
        value = value * 10ll + static_cast<int64_t>(*ptr - '0');
        ++ptr;
    }
    if (negative)
        value = -value;

    if (value == 0)
        return false;
    if (value < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        return false;

    out = static_cast<int32_t>(value);
    return true;
}

NBL_FORCE_INLINE bool resolveObjIndex(const int32_t rawIndex, const size_t elementCount, int32_t& resolved)
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

NBL_FORCE_INLINE bool parseObjFaceVertexTokenFast(const char*& linePtr, const char* const lineEnd, int32_t* idx, const size_t posCount, const size_t uvCount, const size_t normalCount)
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

    const auto ioStart = clock_t::now();
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
        if (!readTextFileWithPolicy(_file, fileContents.data(), fileContents.size(), ioPlan, ioMs, ioTelemetry))
            return {};
        buf = fileContents.data();
    }
    ioMs = std::chrono::duration<double, std::milli>(clock_t::now() - ioStart).count();

    const char* const bufEnd = buf + static_cast<size_t>(filesize);
    const char* bufPtr = buf;

    core::vector<Float3> positions;
    core::vector<Float3> normals;
    core::vector<Float2> uvs;

    core::vector<Float3> outPositions;
    core::vector<Float3> outNormals;
    core::vector<Float2> outUVs;
    core::vector<uint32_t> indices;
    core::vector<int32_t> dedupHeadByPos;
    core::vector<ObjVertexDedupNode> dedupNodes;
    const size_t estimatedAttributeCount = std::max<size_t>(16ull, static_cast<size_t>(filesize) / 32ull);
    const size_t estimatedOutVertexCount = std::max<size_t>(estimatedAttributeCount, static_cast<size_t>(filesize) / 20ull);
    const size_t estimatedOutIndexCount = (estimatedOutVertexCount <= (std::numeric_limits<size_t>::max() / 3ull)) ? (estimatedOutVertexCount * 3ull) : std::numeric_limits<size_t>::max();
    positions.reserve(estimatedAttributeCount);
    normals.reserve(estimatedAttributeCount);
    uvs.reserve(estimatedAttributeCount);
    const size_t initialOutVertexCapacity = std::max<size_t>(1ull, estimatedOutVertexCount);
    const size_t initialOutIndexCapacity = (estimatedOutIndexCount == std::numeric_limits<size_t>::max()) ? 3ull : std::max<size_t>(3ull, estimatedOutIndexCount);
    outPositions.resize(initialOutVertexCapacity);
    outNormals.resize(initialOutVertexCapacity);
    outUVs.resize(initialOutVertexCapacity);
    indices.resize(initialOutIndexCapacity);
    dedupHeadByPos.reserve(estimatedAttributeCount);
    dedupNodes.resize(initialOutVertexCapacity);
    size_t outVertexWriteCount = 0ull;
    size_t outIndexWriteCount = 0ull;
    size_t dedupNodeCount = 0ull;

    auto allocateOutVertex = [&](uint32_t& outIx) -> bool
    {
        if (outVertexWriteCount >= outPositions.size())
        {
            const size_t newCapacity = std::max<size_t>(outVertexWriteCount + 1ull, outPositions.size() * 2ull);
            outPositions.resize(newCapacity);
            outNormals.resize(newCapacity);
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

    bool hasNormals = false;
    bool hasUVs = false;
    hlsl::shapes::AABB<3, hlsl::float32_t> parsedAABB = hlsl::shapes::AABB<3, hlsl::float32_t>::create();
    bool hasParsedAABB = false;
    auto acquireCornerIndex = [&](const int32_t* idx, uint32_t& outIx)->bool
    {
        if (!idx)
            return false;

        const int32_t posIx = idx[0];
        if (posIx < 0 || static_cast<size_t>(posIx) >= positions.size())
            return false;
        if (static_cast<size_t>(posIx) >= dedupHeadByPos.size())
            dedupHeadByPos.resize(positions.size(), -1);

        int32_t nodeIx = dedupHeadByPos[posIx];
        while (nodeIx >= 0)
        {
            const auto& node = dedupNodes[static_cast<size_t>(nodeIx)];
            if (node.uv == idx[1] && node.normal == idx[2])
            {
                outIx = node.outIndex;
                return true;
            }
            nodeIx = node.next;
        }

        if (!allocateOutVertex(outIx))
            return false;
        const int32_t newNodeIx = allocateDedupNode();
        if (newNodeIx < 0)
            return false;
        auto& node = dedupNodes[static_cast<size_t>(newNodeIx)];
        node.uv = idx[1];
        node.normal = idx[2];
        node.outIndex = outIx;
        node.next = dedupHeadByPos[posIx];
        dedupHeadByPos[posIx] = newNodeIx;

        const auto& srcPos = positions[idx[0]];
        outPositions[static_cast<size_t>(outIx)] = srcPos;
        extendAABB(parsedAABB, hasParsedAABB, srcPos);

        Float2 uv(0.f, 0.f);
        if (idx[1] >= 0 && static_cast<size_t>(idx[1]) < uvs.size())
        {
            uv = uvs[idx[1]];
            hasUVs = true;
        }
        outUVs[static_cast<size_t>(outIx)] = uv;

        Float3 normal(0.f, 0.f, 1.f);
        if (idx[2] >= 0 && static_cast<size_t>(idx[2]) < normals.size())
        {
            normal = normals[idx[2]];
            hasNormals = true;
        }
        outNormals[static_cast<size_t>(outIx)] = normal;
        return true;
    };
    auto acquireCornerIndexPositiveTriplet = [&](const int32_t posIx, const int32_t uvIx, const int32_t normalIx, uint32_t& outIx)->bool
    {
        int32_t nodeIx = dedupHeadByPos[static_cast<size_t>(posIx)];
        while (nodeIx >= 0)
        {
            const auto& node = dedupNodes[static_cast<size_t>(nodeIx)];
            if (node.uv == uvIx && node.normal == normalIx)
            {
                outIx = node.outIndex;
                return true;
            }
            nodeIx = node.next;
        }

        if (!allocateOutVertex(outIx))
            return false;
        const int32_t newNodeIx = allocateDedupNode();
        if (newNodeIx < 0)
            return false;
        auto& node = dedupNodes[static_cast<size_t>(newNodeIx)];
        node.uv = uvIx;
        node.normal = normalIx;
        node.outIndex = outIx;
        node.next = dedupHeadByPos[static_cast<size_t>(posIx)];
        dedupHeadByPos[static_cast<size_t>(posIx)] = newNodeIx;

        const auto& srcPos = positions[static_cast<size_t>(posIx)];
        outPositions[static_cast<size_t>(outIx)] = srcPos;
        extendAABB(parsedAABB, hasParsedAABB, srcPos);
        outUVs[static_cast<size_t>(outIx)] = uvs[static_cast<size_t>(uvIx)];
        outNormals[static_cast<size_t>(outIx)] = normals[static_cast<size_t>(normalIx)];
        hasUVs = true;
        hasNormals = true;
        return true;
    };

    const bool trackStages =
        _params.logger.get() != nullptr &&
        ((_params.logger.get()->getLogLevelMask() & system::ILogger::ELL_PERFORMANCE).value != 0u);
    const auto parseStart = clock_t::now();
    while (bufPtr < bufEnd)
    {
            const char* const lineStart = bufPtr;
            const char* lineTerminator = lineStart;
            while (lineTerminator < bufEnd && *lineTerminator != '\n' && *lineTerminator != '\r')
                ++lineTerminator;

            const char* lineEnd = lineTerminator;

            if (lineStart < lineEnd)
            {
                if (*lineStart == 'v')
                {
                    if ((lineStart + 1) < lineEnd && lineStart[1] == ' ')
                    {
                        const auto stageStart = trackStages ? clock_t::now() : clock_t::time_point{};
                        Float3 vec{};
                        const char* ptr = lineStart + 2;
                        for (uint32_t i = 0u; i < 3u; ++i)
                        {
                            while (ptr < lineEnd && isObjInlineWhitespace(*ptr))
                                ++ptr;
                            if (ptr >= lineEnd)
                                return {};
                            if (!parseObjFloat(ptr, lineEnd, (&vec.x)[i]))
                                return {};
                        }
                        positions.push_back(vec);
                        dedupHeadByPos.push_back(-1);
                        if (trackStages)
                            parseVms += std::chrono::duration<double, std::milli>(clock_t::now() - stageStart).count();
                    }
                    else if ((lineStart + 2) < lineEnd && lineStart[1] == 'n' && isObjInlineWhitespace(lineStart[2]))
                    {
                        const auto stageStart = trackStages ? clock_t::now() : clock_t::time_point{};
                        Float3 vec{};
                        const char* ptr = lineStart + 3;
                        for (uint32_t i = 0u; i < 3u; ++i)
                        {
                            while (ptr < lineEnd && isObjInlineWhitespace(*ptr))
                                ++ptr;
                            if (ptr >= lineEnd)
                                return {};
                            if (!parseObjFloat(ptr, lineEnd, (&vec.x)[i]))
                                return {};
                        }
                        normals.push_back(vec);
                        if (trackStages)
                            parseVNms += std::chrono::duration<double, std::milli>(clock_t::now() - stageStart).count();
                    }
                    else if ((lineStart + 2) < lineEnd && lineStart[1] == 't' && isObjInlineWhitespace(lineStart[2]))
                    {
                        const auto stageStart = trackStages ? clock_t::now() : clock_t::time_point{};
                        Float2 vec{};
                        const char* ptr = lineStart + 3;
                        for (uint32_t i = 0u; i < 2u; ++i)
                        {
                            while (ptr < lineEnd && isObjInlineWhitespace(*ptr))
                                ++ptr;
                            if (ptr >= lineEnd)
                                return {};
                            if (!parseObjFloat(ptr, lineEnd, (&vec.x)[i]))
                                return {};
                        }
                        vec.y = 1.f - vec.y;
                        uvs.push_back(vec);
                        if (trackStages)
                            parseVTms += std::chrono::duration<double, std::milli>(clock_t::now() - stageStart).count();
                    }
                }
                else if (*lineStart == 'f' && (lineStart + 1) < lineEnd && isObjInlineWhitespace(lineStart[1]))
                {
                    const auto faceStart = trackStages ? clock_t::now() : clock_t::time_point{};
                    if (positions.empty())
                        return {};
                    ++faceCount;
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
                        const auto dedupStart = trackStages ? clock_t::now() : clock_t::time_point{};
                        uint32_t c0 = 0u;
                        uint32_t c1 = 0u;
                        uint32_t c2 = 0u;
                        if (!acquireCornerIndexPositiveTriplet(triIdx0[0], triIdx0[1], triIdx0[2], c0))
                            return {};
                        if (!acquireCornerIndexPositiveTriplet(triIdx1[0], triIdx1[1], triIdx1[2], c1))
                            return {};
                        if (!acquireCornerIndexPositiveTriplet(triIdx2[0], triIdx2[1], triIdx2[2], c2))
                            return {};
                        if (trackStages)
                            dedupMs += std::chrono::duration<double, std::milli>(clock_t::now() - dedupStart).count();
                        faceFastTokenCount += 3u;
                        const auto emitStart = trackStages ? clock_t::now() : clock_t::time_point{};
                        if (!appendIndex(c2) || !appendIndex(c1) || !appendIndex(c0))
                            return {};
                        if (trackStages)
                            emitMs += std::chrono::duration<double, std::milli>(clock_t::now() - emitStart).count();
                    }
                    else
                    {
                        const char* linePtr = lineStart + 1;
                        uint32_t firstCorner = 0u;
                        uint32_t previousCorner = 0u;
                        uint32_t cornerCount = 0u;

                        if (parsedFirstThree)
                        {
                            const auto dedupStart = trackStages ? clock_t::now() : clock_t::time_point{};
                            uint32_t c0 = 0u;
                            uint32_t c1 = 0u;
                            uint32_t c2 = 0u;
                            if (!acquireCornerIndex(triIdx0, c0))
                                return {};
                            if (!acquireCornerIndex(triIdx1, c1))
                                return {};
                            if (!acquireCornerIndex(triIdx2, c2))
                                return {};
                            if (trackStages)
                                dedupMs += std::chrono::duration<double, std::milli>(clock_t::now() - dedupStart).count();
                            faceFastTokenCount += 3u;
                            const auto emitStart = trackStages ? clock_t::now() : clock_t::time_point{};
                            if (!appendIndex(c2) || !appendIndex(c1) || !appendIndex(c0))
                                return {};
                            if (trackStages)
                                emitMs += std::chrono::duration<double, std::milli>(clock_t::now() - emitStart).count();
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
                            ++faceFastTokenCount;

                            const auto dedupStart = trackStages ? clock_t::now() : clock_t::time_point{};
                            uint32_t cornerIx = 0u;
                            if (!acquireCornerIndex(idx, cornerIx))
                                return {};
                            if (trackStages)
                                dedupMs += std::chrono::duration<double, std::milli>(clock_t::now() - dedupStart).count();

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

                            const auto emitStart = trackStages ? clock_t::now() : clock_t::time_point{};
                            if (!appendIndex(cornerIx) || !appendIndex(previousCorner) || !appendIndex(firstCorner))
                                return {};
                            if (trackStages)
                                emitMs += std::chrono::duration<double, std::milli>(clock_t::now() - emitStart).count();
                            previousCorner = cornerIx;
                            ++cornerCount;
                        }
                    }
                    if (trackStages)
                        parseFaceMs += std::chrono::duration<double, std::milli>(clock_t::now() - faceStart).count();
                }
            }

            if (lineTerminator >= bufEnd)
                bufPtr = bufEnd;
            else if (*lineTerminator == '\r' && (lineTerminator + 1) < bufEnd && lineTerminator[1] == '\n')
                bufPtr = lineTerminator + 2;
            else
                bufPtr = lineTerminator + 1;
    }
    parseMs = std::chrono::duration<double, std::milli>(clock_t::now() - parseStart).count();
    const double parseScanMs = std::max(0.0, parseMs - (parseVms + parseVNms + parseVTms + parseFaceMs + dedupMs + emitMs));

    if (outVertexWriteCount == 0ull)
        return {};

    outPositions.resize(outVertexWriteCount);
    outNormals.resize(outVertexWriteCount);
    outUVs.resize(outVertexWriteCount);
    indices.resize(outIndexWriteCount);

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

    if (_params.loaderFlags & IAssetLoader::ELPF_COMPUTE_CONTENT_HASHES)
    {
        const auto hashStart = clock_t::now();
        objRecomputeContentHashes(geometry.get());
        hashMs = std::chrono::duration<double, std::milli>(clock_t::now() - hashStart).count();
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
