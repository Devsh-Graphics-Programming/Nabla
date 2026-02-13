// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CPLYMeshWriter.h"
#include "nbl/asset/interchange/SInterchangeIOCommon.h"

#ifdef _NBL_COMPILE_WITH_PLY_WRITER_

#include "nbl/system/IFile.h"

#include <algorithm>
#include <cstring>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <limits>
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

namespace ply_writer_detail
{

constexpr size_t ApproxPlyTextBytesPerVertex = 96ull;
constexpr size_t ApproxPlyTextBytesPerFace = 32ull;

bool decodeVec4(const ICPUPolygonGeometry::SDataView& view, const size_t ix, hlsl::float64_t4& out)
{
    out = hlsl::float64_t4(0.0, 0.0, 0.0, 0.0);
    return view.decodeElement(ix, out);
}

template<typename ScalarType>
inline bool readVec3(const ICPUPolygonGeometry::SDataView& view, const hlsl::float32_t3* tightView, const size_t ix, ScalarType (&out)[3])
{
    if (tightView)
    {
        out[0] = static_cast<ScalarType>(tightView[ix].x);
        out[1] = static_cast<ScalarType>(tightView[ix].y);
        out[2] = static_cast<ScalarType>(tightView[ix].z);
        return true;
    }

    hlsl::float64_t4 tmp = {};
    if (!decodeVec4(view, ix, tmp))
        return false;
    out[0] = static_cast<ScalarType>(tmp.x);
    out[1] = static_cast<ScalarType>(tmp.y);
    out[2] = static_cast<ScalarType>(tmp.z);
    return true;
}

template<typename ScalarType>
inline bool readVec2(const ICPUPolygonGeometry::SDataView& view, const hlsl::float32_t2* tightView, const size_t ix, ScalarType (&out)[2])
{
    if (tightView)
    {
        out[0] = static_cast<ScalarType>(tightView[ix].x);
        out[1] = static_cast<ScalarType>(tightView[ix].y);
        return true;
    }

    hlsl::float64_t4 tmp = {};
    if (!decodeVec4(view, ix, tmp))
        return false;
    out[0] = static_cast<ScalarType>(tmp.x);
    out[1] = static_cast<ScalarType>(tmp.y);
    return true;
}

struct SExtraAuxView
{
    const ICPUPolygonGeometry::SDataView* view = nullptr;
    uint32_t components = 0u;
    uint32_t auxIndex = 0u;
};

template<typename ScalarType, typename EmitFn>
inline bool emitExtraAuxValues(const core::vector<SExtraAuxView>& extraAuxViews, const size_t ix, EmitFn&& emit)
{
    hlsl::float64_t4 tmp = {};
    for (const auto& extra : extraAuxViews)
    {
        if (!extra.view || !decodeVec4(*extra.view, ix, tmp))
            return false;
        const ScalarType values[4] = {
            static_cast<ScalarType>(tmp.x),
            static_cast<ScalarType>(tmp.y),
            static_cast<ScalarType>(tmp.z),
            static_cast<ScalarType>(tmp.w)
        };
        emit(values, extra.components);
    }
    return true;
}

const hlsl::float32_t3* getTightFloat3View(const ICPUPolygonGeometry::SDataView& view)
{
    if (!view)
        return nullptr;
    if (view.composed.format != EF_R32G32B32_SFLOAT)
        return nullptr;
    if (view.composed.getStride() != sizeof(hlsl::float32_t3))
        return nullptr;
    return reinterpret_cast<const hlsl::float32_t3*>(view.getPointer());
}

const hlsl::float32_t2* getTightFloat2View(const ICPUPolygonGeometry::SDataView& view)
{
    if (!view)
        return nullptr;
    if (view.composed.format != EF_R32G32_SFLOAT)
        return nullptr;
    if (view.composed.getStride() != sizeof(hlsl::float32_t2))
        return nullptr;
    return reinterpret_cast<const hlsl::float32_t2*>(view.getPointer());
}

void appendUInt(std::string& out, const uint32_t value)
{
    std::array<char, 16> buf = {};
    const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), value);
    if (res.ec == std::errc())
        out.append(buf.data(), static_cast<size_t>(res.ptr - buf.data()));
}

void appendFloatFixed6(std::string& out, double value)
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

void appendVec(std::string& out, const double* values, size_t count, bool flipVectors = false)
{
    constexpr size_t xID = 0u;
    for (size_t i = 0u; i < count; ++i)
    {
        const bool flip = flipVectors && i == xID;
        appendFloatFixed6(out, flip ? -values[i] : values[i]);
        out.push_back(' ');
    }
}

bool writeBinary(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, const core::vector<SExtraAuxView>& extraAuxViews, size_t extraAuxFloatCount, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, uint8_t* dst, bool flipVectors);
bool writeText(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, const core::vector<SExtraAuxView>& extraAuxViews, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, std::string& output, bool flipVectors);

} // namespace ply_writer_detail

bool CPLYMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    using namespace ply_writer_detail;
    using clock_t = std::chrono::high_resolution_clock;

    const auto totalStart = clock_t::now();
    double encodeMs = 0.0;
    double formatMs = 0.0;
    double writeMs = 0.0;
    SFileWriteTelemetry ioTelemetry = {};

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

    core::vector<SExtraAuxView> extraAuxViews;
    size_t extraAuxFloatCount = 0ull;
    extraAuxViews.reserve(auxViews.size());
    for (uint32_t auxIx = 0u; auxIx < static_cast<uint32_t>(auxViews.size()); ++auxIx)
    {
        const auto& view = auxViews[auxIx];
        if (!view || (&view == uvView))
            continue;
        const uint32_t channels = getFormatChannelCount(view.composed.format);
        if (channels == 0u)
            continue;
        const uint32_t components = std::min(4u, channels);
        extraAuxViews.push_back({ &view, components, auxIx });
        extraAuxFloatCount += components;
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
    const auto encodeStart = clock_t::now();

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
    encodeMs = std::chrono::duration<double, std::milli>(clock_t::now() - encodeStart).count();

    const auto flags = _override->getAssetWritingFlags(ctx, geom, 0u);
    const bool binary = (flags & E_WRITER_FLAGS::EWF_BINARY) != 0u;

    const auto formatStart = clock_t::now();
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

    for (const auto& extra : extraAuxViews)
    {
        for (uint32_t component = 0u; component < extra.components; ++component)
        {
            header += "property float aux";
            header += std::to_string(extra.auxIndex);
            if (extra.components > 1u)
            {
                header += "_";
                header += std::to_string(component);
            }
            header += "\n";
        }
    }

    header += "element face ";
    header += std::to_string(faceCount);
    header += "\nproperty list uchar uint vertex_indices\n";
    header += "end_header\n";
    formatMs += std::chrono::duration<double, std::milli>(clock_t::now() - formatStart).count();

    const bool flipVectors = !(flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);

    bool writeOk = false;
    size_t outputBytes = 0ull;
    if (binary)
    {
        const size_t vertexStride = sizeof(float) * (3u + (writeNormals ? 3u : 0u) + (uvView ? 2u : 0u) + extraAuxFloatCount);
        const size_t faceStride = sizeof(uint8_t) + sizeof(uint32_t) * 3u;
        const size_t bodySize = vertexCount * vertexStride + faceCount * faceStride;

        const auto binaryEncodeStart = clock_t::now();
        core::vector<uint8_t> body;
        body.resize(bodySize);
        if (!writeBinary(geom, uvView, extraAuxViews, extraAuxFloatCount, writeNormals, vertexCount, indices, faceCount, body.data(), flipVectors))
            return false;
        encodeMs += std::chrono::duration<double, std::milli>(clock_t::now() - binaryEncodeStart).count();

        const size_t outputSize = header.size() + body.size();
        const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(outputSize), true);
        if (!ioPlan.valid)
        {
            _params.logger.log("PLY writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), ioPlan.reason);
            return false;
        }

        outputBytes = outputSize;
        const auto writeStart = clock_t::now();
        writeOk = writeTwoBuffersWithPolicy(
            file,
            ioPlan,
            reinterpret_cast<const uint8_t*>(header.data()),
            header.size(),
            body.data(),
            body.size(),
            &ioTelemetry);
        writeMs = std::chrono::duration<double, std::milli>(clock_t::now() - writeStart).count();

        const double totalMs = std::chrono::duration<double, std::milli>(clock_t::now() - totalStart).count();
        const double miscMs = std::max(0.0, totalMs - (encodeMs + formatMs + writeMs));
        const uint64_t ioMinWrite = ioTelemetry.getMinOrZero();
        const uint64_t ioAvgWrite = ioTelemetry.getAvgOrZero();
        if (isTinyIOTelemetryLikely(ioTelemetry, static_cast<uint64_t>(outputBytes)))
        {
            _params.logger.log(
                "PLY writer tiny-io guard: file=%s writes=%llu min=%llu avg=%llu",
                system::ILogger::ELL_WARNING,
                file->getFileName().string().c_str(),
                static_cast<unsigned long long>(ioTelemetry.callCount),
                static_cast<unsigned long long>(ioMinWrite),
                static_cast<unsigned long long>(ioAvgWrite));
        }
        _params.logger.log(
            "PLY writer stats: file=%s bytes=%llu vertices=%llu faces=%llu binary=%d io_writes=%llu io_min_write=%llu io_avg_write=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
            system::ILogger::ELL_PERFORMANCE,
            file->getFileName().string().c_str(),
            static_cast<unsigned long long>(outputBytes),
            static_cast<unsigned long long>(vertexCount),
            static_cast<unsigned long long>(faceCount),
            binary ? 1 : 0,
            static_cast<unsigned long long>(ioTelemetry.callCount),
            static_cast<unsigned long long>(ioMinWrite),
            static_cast<unsigned long long>(ioAvgWrite),
            toString(_params.ioPolicy.strategy),
            toString(ioPlan.strategy),
            static_cast<unsigned long long>(ioPlan.chunkSizeBytes),
            ioPlan.reason);
        (void)totalMs;
        (void)encodeMs;
        (void)formatMs;
        (void)writeMs;
        (void)miscMs;
        return writeOk;
    }

    const auto textEncodeStart = clock_t::now();
    std::string body;
    body.reserve(vertexCount * ApproxPlyTextBytesPerVertex + faceCount * ApproxPlyTextBytesPerFace);
    if (!writeText(geom, uvView, extraAuxViews, writeNormals, vertexCount, indices, faceCount, body, flipVectors))
        return false;
    encodeMs += std::chrono::duration<double, std::milli>(clock_t::now() - textEncodeStart).count();

    const size_t outputSize = header.size() + body.size();
    const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(outputSize), true);
    if (!ioPlan.valid)
    {
        _params.logger.log("PLY writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), ioPlan.reason);
        return false;
    }

    outputBytes = outputSize;
    const auto writeStart = clock_t::now();
    writeOk = writeTwoBuffersWithPolicy(
        file,
        ioPlan,
        reinterpret_cast<const uint8_t*>(header.data()),
        header.size(),
        reinterpret_cast<const uint8_t*>(body.data()),
        body.size(),
        &ioTelemetry);
    writeMs = std::chrono::duration<double, std::milli>(clock_t::now() - writeStart).count();

    const double totalMs = std::chrono::duration<double, std::milli>(clock_t::now() - totalStart).count();
    const double miscMs = std::max(0.0, totalMs - (encodeMs + formatMs + writeMs));
    const uint64_t ioMinWrite = ioTelemetry.getMinOrZero();
    const uint64_t ioAvgWrite = ioTelemetry.getAvgOrZero();
    if (isTinyIOTelemetryLikely(ioTelemetry, static_cast<uint64_t>(outputBytes)))
    {
        _params.logger.log(
            "PLY writer tiny-io guard: file=%s writes=%llu min=%llu avg=%llu",
            system::ILogger::ELL_WARNING,
            file->getFileName().string().c_str(),
            static_cast<unsigned long long>(ioTelemetry.callCount),
            static_cast<unsigned long long>(ioMinWrite),
            static_cast<unsigned long long>(ioAvgWrite));
    }
    _params.logger.log(
        "PLY writer stats: file=%s bytes=%llu vertices=%llu faces=%llu binary=%d io_writes=%llu io_min_write=%llu io_avg_write=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
        system::ILogger::ELL_PERFORMANCE,
        file->getFileName().string().c_str(),
        static_cast<unsigned long long>(outputBytes),
        static_cast<unsigned long long>(vertexCount),
        static_cast<unsigned long long>(faceCount),
        binary ? 1 : 0,
        static_cast<unsigned long long>(ioTelemetry.callCount),
        static_cast<unsigned long long>(ioMinWrite),
        static_cast<unsigned long long>(ioAvgWrite),
        toString(_params.ioPolicy.strategy),
        toString(ioPlan.strategy),
        static_cast<unsigned long long>(ioPlan.chunkSizeBytes),
        ioPlan.reason);
    (void)totalMs;
    (void)encodeMs;
    (void)formatMs;
    (void)writeMs;
    (void)miscMs;
    return writeOk;
}

bool ply_writer_detail::writeBinary(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, const core::vector<SExtraAuxView>& extraAuxViews, size_t extraAuxFloatCount, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, uint8_t* dst, bool flipVectors)
{
    if (!dst)
        return false;

    constexpr size_t Float3Bytes = sizeof(float) * 3ull;
    constexpr size_t Float2Bytes = sizeof(float) * 2ull;
    const auto& positionView = geom->getPositionView();
    const auto& normalView = geom->getNormalView();
    const hlsl::float32_t3* const tightPos = getTightFloat3View(positionView);
    const hlsl::float32_t3* const tightNormal = writeNormals ? getTightFloat3View(normalView) : nullptr;
    const bool hasUV = uvView != nullptr;
    const hlsl::float32_t2* const tightUV = hasUV ? getTightFloat2View(*uvView) : nullptr;
    const bool hasExtraAux = extraAuxFloatCount > 0ull;

    if (tightPos && (!writeNormals || tightNormal) && (!hasUV || tightUV) && !hasExtraAux && !flipVectors)
    {
        for (size_t i = 0; i < vertexCount; ++i)
        {
            std::memcpy(dst, tightPos + i, Float3Bytes);
            dst += Float3Bytes;
            if (writeNormals)
            {
                std::memcpy(dst, tightNormal + i, Float3Bytes);
                dst += Float3Bytes;
            }
            if (hasUV)
            {
                std::memcpy(dst, tightUV + i, Float2Bytes);
                dst += Float2Bytes;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < vertexCount; ++i)
        {
            float pos[3] = {};
            if (!readVec3(positionView, tightPos, i, pos))
                return false;
            if (flipVectors)
                pos[0] = -pos[0];
            std::memcpy(dst, pos, Float3Bytes);
            dst += Float3Bytes;

            if (writeNormals)
            {
                float normal[3] = {};
                if (!readVec3(normalView, tightNormal, i, normal))
                    return false;
                if (flipVectors)
                    normal[0] = -normal[0];
                std::memcpy(dst, normal, Float3Bytes);
                dst += Float3Bytes;
            }

            if (hasUV)
            {
                float uv[2] = {};
                if (!readVec2(*uvView, tightUV, i, uv))
                    return false;
                std::memcpy(dst, uv, Float2Bytes);
                dst += Float2Bytes;
            }

            if (hasExtraAux)
            {
                if (!emitExtraAuxValues<float>(extraAuxViews, i, [&](const float* values, const uint32_t components)
                {
                    std::memcpy(dst, values, sizeof(float) * components);
                    dst += sizeof(float) * components;
                }))
                    return false;
            }
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

bool ply_writer_detail::writeText(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, const core::vector<SExtraAuxView>& extraAuxViews, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, std::string& output, bool flipVectors)
{
    const auto& positionView = geom->getPositionView();
    const auto& normalView = geom->getNormalView();
    const hlsl::float32_t3* const tightPos = getTightFloat3View(positionView);
    const hlsl::float32_t3* const tightNormal = writeNormals ? getTightFloat3View(normalView) : nullptr;
    const hlsl::float32_t2* const tightUV = uvView ? getTightFloat2View(*uvView) : nullptr;

    for (size_t i = 0; i < vertexCount; ++i)
    {
        double pos[3] = {};
        if (!readVec3(positionView, tightPos, i, pos))
            return false;
        appendVec(output, pos, 3u, flipVectors);

        if (writeNormals)
        {
            double normal[3] = {};
            if (!readVec3(normalView, tightNormal, i, normal))
                return false;
            appendVec(output, normal, 3u, flipVectors);
        }

        if (uvView)
        {
            double uv[2] = {};
            if (!readVec2(*uvView, tightUV, i, uv))
                return false;
            appendVec(output, uv, 2u, false);
        }

        if (!emitExtraAuxValues<double>(extraAuxViews, i, [&](const double* values, const uint32_t components)
        {
            appendVec(output, values, components, false);
        }))
            return false;

        output += "\n";
    }

    for (size_t i = 0; i < faceCount; ++i)
    {
        const uint32_t* tri = indices + (i * 3u);
        output.append("3 ");
        appendUInt(output, tri[0]);
        output.push_back(' ');
        appendUInt(output, tri[1]);
        output.push_back(' ');
        appendUInt(output, tri[2]);
        output.push_back('\n');
    }
    return true;
}

} // namespace nbl::asset

#endif // _NBL_COMPILE_WITH_PLY_WRITER_

