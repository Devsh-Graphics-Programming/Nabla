// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CPLYMeshWriter.h"

#ifdef _NBL_COMPILE_WITH_PLY_WRITER_

#include "nbl/system/IFile.h"

#include <algorithm>
#include <cstring>
#include <array>
#include <charconv>
#include <cstdio>
#include <system_error>

namespace nbl::asset
{

CPLYMeshWriter::CPLYMeshWriter()
{
    #ifdef _NBL_DEBUG
    setDebugName("CPLYMeshWriter");
    #endif
}

const char** CPLYMeshWriter::getAssociatedFileExtensions() const
{
    static const char* ext[] = { "ply", nullptr };
    return ext;
}

uint32_t CPLYMeshWriter::getSupportedFlags()
{
    return asset::EWF_BINARY;
}

uint32_t CPLYMeshWriter::getForcedFlags()
{
    return 0u;
}

static inline bool decodeVec4(const ICPUPolygonGeometry::SDataView& view, const size_t ix, hlsl::float64_t4& out)
{
    out = hlsl::float64_t4(0.0, 0.0, 0.0, 0.0);
    return view.decodeElement(ix, out);
}

static inline const hlsl::float32_t3* getTightFloat3View(const ICPUPolygonGeometry::SDataView& view)
{
    if (!view)
        return nullptr;
    if (view.composed.format != EF_R32G32B32_SFLOAT)
        return nullptr;
    if (view.composed.getStride() != sizeof(hlsl::float32_t3))
        return nullptr;
    return reinterpret_cast<const hlsl::float32_t3*>(view.getPointer());
}

static inline const hlsl::float32_t2* getTightFloat2View(const ICPUPolygonGeometry::SDataView& view)
{
    if (!view)
        return nullptr;
    if (view.composed.format != EF_R32G32_SFLOAT)
        return nullptr;
    if (view.composed.getStride() != sizeof(hlsl::float32_t2))
        return nullptr;
    return reinterpret_cast<const hlsl::float32_t2*>(view.getPointer());
}

static inline void appendUInt(std::string& out, const uint32_t value)
{
    std::array<char, 16> buf = {};
    const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), value);
    if (res.ec == std::errc())
        out.append(buf.data(), static_cast<size_t>(res.ptr - buf.data()));
}

static inline void appendFloatFixed6(std::string& out, double value)
{
    std::array<char, 64> buf = {};
    const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), value, std::chars_format::fixed, 6);
    if (res.ec == std::errc())
    {
        out.append(buf.data(), static_cast<size_t>(res.ptr - buf.data()));
        return;
    }

    const int written = std::snprintf(buf.data(), buf.size(), "%.6f", value);
    if (written > 0)
        out.append(buf.data(), static_cast<size_t>(written));
}

static inline void appendVec(std::string& out, const double* values, size_t count, bool flipVectors = false)
{
    constexpr size_t xID = 0u;
    for (size_t i = 0u; i < count; ++i)
    {
        const bool flip = flipVectors && i == xID;
        appendFloatFixed6(out, flip ? -values[i] : values[i]);
        out += " ";
    }
}

static bool writeBufferWithPolicy(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const uint8_t* data, size_t byteCount);
static bool writeBinary(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, uint8_t* dst, bool flipVectors);
static bool writeText(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, std::string& output, bool flipVectors);

bool CPLYMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if (!_override)
        getDefaultOverride(_override);

    if (!_file || !_params.rootAsset)
        return false;

    const auto* geom = IAsset::castDown<const ICPUPolygonGeometry>(_params.rootAsset);
    if (!geom || !geom->valid())
        return false;

    SAssetWriteContext ctx = { _params, _file };
    system::IFile* file = _override->getOutputFile(_file, ctx, { geom, 0u });
    if (!file)
        return false;

    const auto& positionView = geom->getPositionView();
    const auto& normalView = geom->getNormalView();
    const auto& auxViews = geom->getAuxAttributeViews();

    const bool writeNormals = static_cast<bool>(normalView);

    const ICPUPolygonGeometry::SDataView* uvView = nullptr;
    for (const auto& view : auxViews)
    {
        if (!view)
            continue;
        const auto channels = getFormatChannelCount(view.composed.format);
        if (channels >= 2u)
        {
            uvView = &view;
            break;
        }
    }

    const size_t vertexCount = positionView.getElementCount();
    if (vertexCount == 0)
        return false;

    const auto* indexing = geom->getIndexingCallback();
    if (!indexing)
        return false;

    if (indexing->knownTopology() != E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST)
        return false;

    const auto& indexView = geom->getIndexView();

    core::vector<uint32_t> indexData;
    const uint32_t* indices = nullptr;
    size_t faceCount = 0;

    if (indexView)
    {
        const size_t indexCount = indexView.getElementCount();
        if (indexCount % 3u != 0u)
            return false;

        const void* src = indexView.getPointer();
        if (!src)
            return false;

        if (indexView.composed.format == EF_R32_UINT && indexView.composed.getStride() == sizeof(uint32_t))
        {
            indices = reinterpret_cast<const uint32_t*>(src);
        }
        else if (indexView.composed.format == EF_R16_UINT && indexView.composed.getStride() == sizeof(uint16_t))
        {
            indexData.resize(indexCount);
            const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
            for (size_t i = 0; i < indexCount; ++i)
                indexData[i] = src16[i];
            indices = indexData.data();
        }
        else
        {
            indexData.resize(indexCount);
            hlsl::vector<uint32_t, 1> decoded = {};
            for (size_t i = 0; i < indexCount; ++i)
            {
                if (!indexView.decodeElement(i, decoded))
                    return false;
                indexData[i] = decoded.x;
            }
            indices = indexData.data();
        }
        faceCount = indexCount / 3u;
    }
    else
    {
        if (vertexCount % 3u != 0u)
            return false;

        indexData.resize(vertexCount);
        for (size_t i = 0; i < vertexCount; ++i)
            indexData[i] = static_cast<uint32_t>(i);

        indices = indexData.data();
        faceCount = vertexCount / 3u;
    }

    const auto flags = _override->getAssetWritingFlags(ctx, geom, 0u);
    const bool binary = (flags & E_WRITER_FLAGS::EWF_BINARY) != 0u;

    std::string header = "ply\n";
    header += binary ? "format binary_little_endian 1.0" : "format ascii 1.0";
    header += "\ncomment Nabla ";
    header += NABLA_SDK_VERSION;

    header += "\nelement vertex ";
    header += std::to_string(vertexCount);
    header += "\n";

    header += "property float x\n";
    header += "property float y\n";
    header += "property float z\n";

    if (writeNormals)
    {
        header += "property float nx\n";
        header += "property float ny\n";
        header += "property float nz\n";
    }

    if (uvView)
    {
        header += "property float u\n";
        header += "property float v\n";
    }

    header += "element face ";
    header += std::to_string(faceCount);
    header += "\nproperty list uchar uint vertex_indices\n";
    header += "end_header\n";

    const bool flipVectors = !(flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);

    if (binary)
    {
        const size_t vertexStride = sizeof(float) * (3u + (writeNormals ? 3u : 0u) + (uvView ? 2u : 0u));
        const size_t faceStride = sizeof(uint8_t) + sizeof(uint32_t) * 3u;
        const size_t bodySize = vertexCount * vertexStride + faceCount * faceStride;

        core::vector<uint8_t> output;
        output.resize(header.size() + bodySize);
        if (!header.empty())
            std::memcpy(output.data(), header.data(), header.size());
        if (!writeBinary(geom, uvView, writeNormals, vertexCount, indices, faceCount, output.data() + header.size(), flipVectors))
            return false;

        const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(output.size()), true);
        if (!ioPlan.valid)
        {
            _params.logger.log("PLY writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), ioPlan.reason);
            return false;
        }
        return writeBufferWithPolicy(file, ioPlan, output.data(), output.size());
    }

    std::string body;
    body.reserve(vertexCount * 96ull + faceCount * 32ull);
    if (!writeText(geom, uvView, writeNormals, vertexCount, indices, faceCount, body, flipVectors))
        return false;

    std::string output = header;
    output += body;
    const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(output.size()), true);
    if (!ioPlan.valid)
    {
        _params.logger.log("PLY writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), ioPlan.reason);
        return false;
    }
    return writeBufferWithPolicy(file, ioPlan, reinterpret_cast<const uint8_t*>(output.data()), output.size());
}

static bool writeBufferWithPolicy(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const uint8_t* data, size_t byteCount)
{
    if (!file || (!data && byteCount != 0ull))
        return false;

    size_t fileOffset = 0ull;
    switch (ioPlan.strategy)
    {
        case SResolvedFileIOPolicy::Strategy::WholeFile:
        {
            system::IFile::success_t success;
            file->write(success, data, fileOffset, byteCount);
            return success && success.getBytesProcessed() == byteCount;
        }
        case SResolvedFileIOPolicy::Strategy::Chunked:
        default:
        {
            while (fileOffset < byteCount)
            {
                const size_t toWrite = static_cast<size_t>(std::min<uint64_t>(ioPlan.chunkSizeBytes, byteCount - fileOffset));
                system::IFile::success_t success;
                file->write(success, data + fileOffset, fileOffset, toWrite);
                if (!success)
                    return false;
                const size_t written = success.getBytesProcessed();
                if (written == 0ull)
                    return false;
                fileOffset += written;
            }
            return true;
        }
    }
}

static bool writeBinary(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, uint8_t* dst, bool flipVectors)
{
    if (!dst)
        return false;

    const auto& positionView = geom->getPositionView();
    const auto& normalView = geom->getNormalView();
    const hlsl::float32_t3* const tightPos = getTightFloat3View(positionView);
    const hlsl::float32_t3* const tightNormal = writeNormals ? getTightFloat3View(normalView) : nullptr;
    const hlsl::float32_t2* const tightUV = uvView ? getTightFloat2View(*uvView) : nullptr;

    hlsl::float64_t4 tmp = {};
    for (size_t i = 0; i < vertexCount; ++i)
    {
        float pos[3] = {};
        if (tightPos)
        {
            pos[0] = tightPos[i].x;
            pos[1] = tightPos[i].y;
            pos[2] = tightPos[i].z;
        }
        else
        {
            if (!decodeVec4(positionView, i, tmp))
                return false;
            pos[0] = static_cast<float>(tmp.x);
            pos[1] = static_cast<float>(tmp.y);
            pos[2] = static_cast<float>(tmp.z);
        }
        if (flipVectors)
            pos[0] = -pos[0];

        std::memcpy(dst, pos, sizeof(pos));
        dst += sizeof(pos);

        if (writeNormals)
        {
            float normal[3] = {};
            if (tightNormal)
            {
                normal[0] = tightNormal[i].x;
                normal[1] = tightNormal[i].y;
                normal[2] = tightNormal[i].z;
            }
            else
            {
                if (!decodeVec4(normalView, i, tmp))
                    return false;
                normal[0] = static_cast<float>(tmp.x);
                normal[1] = static_cast<float>(tmp.y);
                normal[2] = static_cast<float>(tmp.z);
            }
            if (flipVectors)
                normal[0] = -normal[0];

            std::memcpy(dst, normal, sizeof(normal));
            dst += sizeof(normal);
        }

        if (uvView)
        {
            float uv[2] = {};
            if (tightUV)
            {
                uv[0] = tightUV[i].x;
                uv[1] = tightUV[i].y;
            }
            else
            {
                if (!decodeVec4(*uvView, i, tmp))
                    return false;
                uv[0] = static_cast<float>(tmp.x);
                uv[1] = static_cast<float>(tmp.y);
            }

            std::memcpy(dst, uv, sizeof(uv));
            dst += sizeof(uv);
        }
    }

    for (size_t i = 0; i < faceCount; ++i)
    {
        const uint8_t listSize = 3u;
        *dst++ = listSize;

        const uint32_t* tri = indices + (i * 3u);
        std::memcpy(dst, tri, sizeof(uint32_t) * 3u);
        dst += sizeof(uint32_t) * 3u;
    }

    return true;
}

static bool writeText(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, std::string& output, bool flipVectors)
{
    const auto& positionView = geom->getPositionView();
    const auto& normalView = geom->getNormalView();
    const hlsl::float32_t3* const tightPos = getTightFloat3View(positionView);
    const hlsl::float32_t3* const tightNormal = writeNormals ? getTightFloat3View(normalView) : nullptr;
    const hlsl::float32_t2* const tightUV = uvView ? getTightFloat2View(*uvView) : nullptr;

    hlsl::float64_t4 tmp = {};
    for (size_t i = 0; i < vertexCount; ++i)
    {
        double pos[3] = {};
        if (tightPos)
        {
            pos[0] = tightPos[i].x;
            pos[1] = tightPos[i].y;
            pos[2] = tightPos[i].z;
        }
        else
        {
            if (!decodeVec4(positionView, i, tmp))
                return false;
            pos[0] = tmp.x;
            pos[1] = tmp.y;
            pos[2] = tmp.z;
        }
        appendVec(output, pos, 3u, flipVectors);

        if (writeNormals)
        {
            double normal[3] = {};
            if (tightNormal)
            {
                normal[0] = tightNormal[i].x;
                normal[1] = tightNormal[i].y;
                normal[2] = tightNormal[i].z;
            }
            else
            {
                if (!decodeVec4(normalView, i, tmp))
                    return false;
                normal[0] = tmp.x;
                normal[1] = tmp.y;
                normal[2] = tmp.z;
            }
            appendVec(output, normal, 3u, flipVectors);
        }

        if (uvView)
        {
            double uv[2] = {};
            if (tightUV)
            {
                uv[0] = tightUV[i].x;
                uv[1] = tightUV[i].y;
            }
            else
            {
                if (!decodeVec4(*uvView, i, tmp))
                    return false;
                uv[0] = tmp.x;
                uv[1] = tmp.y;
            }
            appendVec(output, uv, 2u, false);
        }

        output += "\n";
    }

    for (size_t i = 0; i < faceCount; ++i)
    {
        const uint32_t* tri = indices + (i * 3u);
        output += "3 ";
        appendUInt(output, tri[0]);
        output += " ";
        appendUInt(output, tri[1]);
        output += " ";
        appendUInt(output, tri[2]);
        output += "\n";
    }
    return true;
}

} // namespace nbl::asset

#endif // _NBL_COMPILE_WITH_PLY_WRITER_

