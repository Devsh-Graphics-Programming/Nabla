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
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_map>

namespace nbl::asset
{

namespace
{

struct ObjVertexKey
{
    int32_t pos;
    int32_t uv;
    int32_t normal;

    inline bool operator==(const ObjVertexKey& other) const
    {
        return pos == other.pos && uv == other.uv && normal == other.normal;
    }
};

struct ObjVertexKeyHash
{
    inline size_t operator()(const ObjVertexKey& key) const noexcept
    {
        size_t h = static_cast<size_t>(static_cast<uint32_t>(key.pos));
        h ^= static_cast<size_t>(static_cast<uint32_t>(key.uv)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(static_cast<uint32_t>(key.normal)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Float3
{
    float x;
    float y;
    float z;
};

struct Float2
{
    float x;
    float y;
};

static_assert(sizeof(Float3) == sizeof(float) * 3ull);
static_assert(sizeof(Float2) == sizeof(float) * 2ull);

bool readTextFileWithPolicy(system::IFile* file, char* dst, size_t byteCount, const SResolvedFileIOPolicy& ioPlan, double& ioMs)
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

const char* readVec3(const char* bufPtr, float vec[3], const char* const bufEnd)
{
    bufPtr = goNextWord(bufPtr, bufEnd, false);
    for (uint32_t i = 0u; i < 3u; ++i)
    {
        if (bufPtr >= bufEnd)
            return bufPtr;

        char* endPtr = nullptr;
        vec[i] = std::strtof(bufPtr, &endPtr);
        if (endPtr == bufPtr)
            return bufPtr;
        bufPtr = endPtr;

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

        char* endPtr = nullptr;
        vec[i] = std::strtof(bufPtr, &endPtr);
        if (endPtr == bufPtr)
            return bufPtr;
        bufPtr = endPtr;

        while (bufPtr < bufEnd && core::isspace(*bufPtr) && *bufPtr != '\n' && *bufPtr != '\r')
            ++bufPtr;
    }

    vec[1] = 1.f - vec[1];
    return bufPtr;
}

bool retrieveVertexIndices(const char* tokenBegin, const char* tokenEnd, int32_t* idx, uint32_t vbsize, uint32_t vtsize, uint32_t vnsize)
{
    if (!tokenBegin || !idx)
        return false;

    idx[0] = -1;
    idx[1] = -1;
    idx[2] = -1;

    const char* p = tokenBegin;
    for (uint32_t idxType = 0u; idxType < 3u && p < tokenEnd; ++idxType)
    {
        if (*p == '/')
        {
            ++p;
            continue;
        }

        char* endNum = nullptr;
        const long parsed = std::strtol(p, &endNum, 10);
        if (endNum == p)
            return false;

        int32_t value = static_cast<int32_t>(parsed);
        if (value < 0)
        {
            switch (idxType)
            {
                case 0:
                    value += static_cast<int32_t>(vbsize);
                    break;
                case 1:
                    value += static_cast<int32_t>(vtsize);
                    break;
                case 2:
                    value += static_cast<int32_t>(vnsize);
                    break;
                default:
                    break;
            }
        }
        else
        {
            value -= 1;
        }
        idx[idxType] = value;

        p = endNum;
        if (p >= tokenEnd)
            break;

        if (*p != '/')
            break;
        ++p;
    }

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
    double aabbMs = 0.0;
    uint64_t faceCount = 0u;

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
    if (!readTextFileWithPolicy(_file, fileContents.data(), fileContents.size(), ioPlan, ioMs))
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

    std::unordered_map<ObjVertexKey, uint32_t, ObjVertexKeyHash> vtxMap;

    bool hasNormals = false;
    bool hasUVs = false;
    hlsl::shapes::AABB<3, hlsl::float32_t> parsedAABB = hlsl::shapes::AABB<3, hlsl::float32_t>::create();
    bool hasParsedAABB = false;

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
                        Float3 vec{};
                        bufPtr = readVec3(bufPtr, &vec.x, bufEnd);
                        positions.push_back(vec);
                    }
                    break;
                    case 'n':
                    {
                        Float3 vec{};
                        bufPtr = readVec3(bufPtr, &vec.x, bufEnd);
                        normals.push_back(vec);
                    }
                    break;
                    case 't':
                    {
                        Float2 vec{};
                        bufPtr = readUV(bufPtr, &vec.x, bufEnd);
                        uvs.push_back(vec);
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

                const char* endPtr = bufPtr;
                while (endPtr != bufEnd && *endPtr != '\n' && *endPtr != '\r')
                    ++endPtr;

                core::vector<uint32_t> faceCorners;
                faceCorners.reserve(16ull);

                const char* linePtr = goNextWord(bufPtr, endPtr);
                while (linePtr < endPtr && 0 != linePtr[0])
                {
                    int32_t idx[3] = { -1, -1, -1 };
                    const char* tokenEnd = linePtr;
                    while (tokenEnd < endPtr && !core::isspace(*tokenEnd))
                        ++tokenEnd;
                    if (!retrieveVertexIndices(linePtr, tokenEnd, idx, positions.size(), uvs.size(), normals.size()))
                        return {};

                    if (idx[0] < 0 || static_cast<size_t>(idx[0]) >= positions.size())
                        return {};

                    ObjVertexKey key = { idx[0], idx[1], idx[2] };
                    auto it = vtxMap.find(key);
                    uint32_t outIx = 0u;
                    if (it == vtxMap.end())
                    {
                        if (outPositions.empty())
                        {
                            outPositions.reserve(positions.size());
                            outNormals.reserve(positions.size());
                            outUVs.reserve(positions.size());
                        }
                        outIx = static_cast<uint32_t>(outPositions.size());
                        vtxMap.emplace(key, outIx);

                        const auto& srcPos = positions[idx[0]];
                        outPositions.push_back(srcPos);
                        const hlsl::float32_t3 p = { srcPos.x, srcPos.y, srcPos.z };
                        if (!hasParsedAABB)
                        {
                            parsedAABB.minVx = p;
                            parsedAABB.maxVx = p;
                            hasParsedAABB = true;
                        }
                        else
                        {
                            if (p.x < parsedAABB.minVx.x) parsedAABB.minVx.x = p.x;
                            if (p.y < parsedAABB.minVx.y) parsedAABB.minVx.y = p.y;
                            if (p.z < parsedAABB.minVx.z) parsedAABB.minVx.z = p.z;
                            if (p.x > parsedAABB.maxVx.x) parsedAABB.maxVx.x = p.x;
                            if (p.y > parsedAABB.maxVx.y) parsedAABB.maxVx.y = p.y;
                            if (p.z > parsedAABB.maxVx.z) parsedAABB.maxVx.z = p.z;
                        }

                        Float2 uv = { 0.f, 0.f };
                        if (idx[1] >= 0 && static_cast<size_t>(idx[1]) < uvs.size())
                        {
                            uv = uvs[idx[1]];
                            hasUVs = true;
                        }
                        outUVs.push_back(uv);

                        Float3 normal = { 0.f, 0.f, 1.f };
                        if (idx[2] >= 0 && static_cast<size_t>(idx[2]) < normals.size())
                        {
                            normal = normals[idx[2]];
                            hasNormals = true;
                        }
                        outNormals.push_back(normal);
                    }
                    else
                    {
                        outIx = it->second;
                    }

                    faceCorners.push_back(outIx);

                    while (tokenEnd < endPtr && core::isspace(*tokenEnd))
                        ++tokenEnd;
                    linePtr = tokenEnd;
                }

                for (uint32_t i = 1u; i + 1u < faceCorners.size(); ++i)
                {
                    indices.push_back(faceCorners[i + 1]);
                    indices.push_back(faceCorners[i]);
                    indices.push_back(faceCorners[0]);
                }
            }
            break;
            default:
                break;
        }

        bufPtr = goNextLine(bufPtr, bufEnd);
    }
    parseMs = std::chrono::duration<double, std::milli>(clock_t::now() - parseStart).count();

    if (outPositions.empty())
        return {};

    const auto buildStart = clock_t::now();
    auto geometry = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
    geometry->setPositionView(IGeometryLoader::createView(EF_R32G32B32_SFLOAT, outPositions.size(), outPositions.data()));

    if (hasNormals)
        geometry->setNormalView(IGeometryLoader::createView(EF_R32G32B32_SFLOAT, outNormals.size(), outNormals.data()));

    if (hasUVs)
        geometry->getAuxAttributeViews()->push_back(IGeometryLoader::createView(EF_R32G32_SFLOAT, outUVs.size(), outUVs.data()));

    if (!indices.empty())
    {
        geometry->setIndexing(IPolygonGeometryBase::TriangleList());
        const auto maxIndex = *std::max_element(indices.begin(), indices.end());
        if (maxIndex <= std::numeric_limits<uint16_t>::max())
        {
            core::vector<uint16_t> indices16(indices.size());
            for (size_t i = 0u; i < indices.size(); ++i)
                indices16[i] = static_cast<uint16_t>(indices[i]);
            geometry->setIndexView(IGeometryLoader::createView(EF_R16_UINT, indices16.size(), indices16.data()));
        }
        else
        {
            geometry->setIndexView(IGeometryLoader::createView(EF_R32_UINT, indices.size(), indices.data()));
        }
    }
    else
    {
        geometry->setIndexing(IPolygonGeometryBase::PointList());
    }
    buildMs = std::chrono::duration<double, std::milli>(clock_t::now() - buildStart).count();

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
        "OBJ loader perf: file=%s total=%.3f ms io=%.3f parse=%.3f build=%.3f aabb=%.3f in(v=%llu n=%llu uv=%llu) out(v=%llu idx=%llu faces=%llu) io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
        system::ILogger::ELL_PERFORMANCE,
        _file->getFileName().string().c_str(),
        totalMs,
        ioMs,
        parseMs,
        buildMs,
        aabbMs,
        static_cast<unsigned long long>(positions.size()),
        static_cast<unsigned long long>(normals.size()),
        static_cast<unsigned long long>(uvs.size()),
        static_cast<unsigned long long>(outPositions.size()),
        static_cast<unsigned long long>(indices.size()),
        static_cast<unsigned long long>(faceCount),
        toString(_params.ioPolicy.strategy),
        toString(ioPlan.strategy),
        static_cast<unsigned long long>(ioPlan.chunkSizeBytes),
        ioPlan.reason);

    return SAssetBundle(core::smart_refctd_ptr<IAssetMetadata>(), { std::move(geometry) });
}

}

#endif // _NBL_COMPILE_WITH_OBJ_LOADER_
