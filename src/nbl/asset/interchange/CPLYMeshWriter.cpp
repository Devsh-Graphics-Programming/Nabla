// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CPLYMeshWriter.h"
#include "nbl/asset/interchange/SGeometryWriterCommon.h"
#include "nbl/asset/interchange/SInterchangeIOCommon.h"

#ifdef _NBL_COMPILE_WITH_PLY_WRITER_

#include "nbl/system/IFile.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <array>
#include <charconv>
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

enum class EPlyScalarType : uint8_t
{
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64
};

struct SPlyScalarMeta
{
    const char* name = "float32";
    uint32_t byteSize = sizeof(float);
    bool integer = false;
    bool signedType = true;
};

SPlyScalarMeta getPlyScalarMeta(const EPlyScalarType type)
{
    switch (type)
    {
        case EPlyScalarType::Int8: return { "int8", sizeof(int8_t), true, true };
        case EPlyScalarType::UInt8: return { "uint8", sizeof(uint8_t), true, false };
        case EPlyScalarType::Int16: return { "int16", sizeof(int16_t), true, true };
        case EPlyScalarType::UInt16: return { "uint16", sizeof(uint16_t), true, false };
        case EPlyScalarType::Int32: return { "int32", sizeof(int32_t), true, true };
        case EPlyScalarType::UInt32: return { "uint32", sizeof(uint32_t), true, false };
        case EPlyScalarType::Float64: return { "float64", sizeof(double), false, true };
        default: return { "float32", sizeof(float), false, true };
    }
}

bool isPlyUnsupportedPackedFormat(const E_FORMAT format)
{
    switch (format)
    {
        case EF_A2R10G10B10_UINT_PACK32:
        case EF_A2R10G10B10_SINT_PACK32:
        case EF_A2R10G10B10_UNORM_PACK32:
        case EF_A2R10G10B10_SNORM_PACK32:
        case EF_A2R10G10B10_USCALED_PACK32:
        case EF_A2R10G10B10_SSCALED_PACK32:
        case EF_A2B10G10R10_UINT_PACK32:
        case EF_A2B10G10R10_SINT_PACK32:
        case EF_A2B10G10R10_UNORM_PACK32:
        case EF_A2B10G10R10_SNORM_PACK32:
        case EF_A2B10G10R10_USCALED_PACK32:
        case EF_A2B10G10R10_SSCALED_PACK32:
        case EF_B10G11R11_UFLOAT_PACK32:
        case EF_E5B9G9R9_UFLOAT_PACK32:
            return true;
        default:
            return false;
    }
}

EPlyScalarType selectPlyScalarType(const E_FORMAT format)
{
    if (format == EF_UNKNOWN || isPlyUnsupportedPackedFormat(format))
        return EPlyScalarType::Float32;
    if (isNormalizedFormat(format) || isScaledFormat(format))
        return EPlyScalarType::Float32;

    const uint32_t channels = getFormatChannelCount(format);
    if (channels == 0u)
        return EPlyScalarType::Float32;

    const auto bytesPerPixel = getBytesPerPixel(format);
    if (bytesPerPixel.getDenominator() != 1u)
        return EPlyScalarType::Float32;
    const uint32_t pixelBytes = bytesPerPixel.getNumerator();
    if (pixelBytes == 0u || (pixelBytes % channels) != 0u)
        return EPlyScalarType::Float32;
    const uint32_t bytesPerChannel = pixelBytes / channels;

    if (isIntegerFormat(format))
    {
        const bool signedType = isSignedFormat(format);
        switch (bytesPerChannel)
        {
            case 1u: return signedType ? EPlyScalarType::Int8 : EPlyScalarType::UInt8;
            case 2u: return signedType ? EPlyScalarType::Int16 : EPlyScalarType::UInt16;
            case 4u: return signedType ? EPlyScalarType::Int32 : EPlyScalarType::UInt32;
            default: return EPlyScalarType::Float64;
        }
    }

    if (isFloatingPointFormat(format))
        return bytesPerChannel >= 8u ? EPlyScalarType::Float64 : EPlyScalarType::Float32;

    return EPlyScalarType::Float32;
}

bool decodeVec4(const ICPUPolygonGeometry::SDataView& view, const size_t ix, hlsl::float64_t4& out)
{
    out = hlsl::float64_t4(0.0, 0.0, 0.0, 0.0);
    if (!view.composed.isFormatted())
        return false;

    const void* src = view.getPointer(ix);
    if (!src)
        return false;

    const void* srcArr[4] = { src, nullptr, nullptr, nullptr };
    double tmp[4] = {};
    if (!decodePixels<double>(view.composed.format, srcArr, tmp, 0u, 0u))
        return false;

    const uint32_t channels = std::min<uint32_t>(4u, getFormatChannelCount(view.composed.format));
    if (isNormalizedFormat(view.composed.format))
    {
        const auto range = view.composed.getRange<hlsl::shapes::AABB<4, hlsl::float64_t>>();
        for (uint32_t i = 0u; i < channels; ++i)
            (&out.x)[i] = tmp[i] * (range.maxVx[i] - range.minVx[i]) + range.minVx[i];
    }
    else
    {
        for (uint32_t i = 0u; i < channels; ++i)
            (&out.x)[i] = tmp[i];
    }
    return true;
}

bool decodeSigned4Raw(const ICPUPolygonGeometry::SDataView& view, const size_t ix, int64_t (&out)[4])
{
    const void* src = view.getPointer(ix);
    if (!src)
        return false;
    const void* srcArr[4] = { src, nullptr, nullptr, nullptr };
    return decodePixels<int64_t>(view.composed.format, srcArr, out, 0u, 0u);
}

bool decodeUnsigned4Raw(const ICPUPolygonGeometry::SDataView& view, const size_t ix, uint64_t (&out)[4])
{
    const void* src = view.getPointer(ix);
    if (!src)
        return false;
    const void* srcArr[4] = { src, nullptr, nullptr, nullptr };
    return decodePixels<uint64_t>(view.composed.format, srcArr, out, 0u, 0u);
}

void appendUInt(std::string& out, const uint64_t value)
{
    std::array<char, 32> buf = {};
    const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), value);
    if (res.ec == std::errc())
        out.append(buf.data(), static_cast<size_t>(res.ptr - buf.data()));
}

void appendInt(std::string& out, const int64_t value)
{
    std::array<char, 32> buf = {};
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

inline bool writeTypedViewBinary(const ICPUPolygonGeometry::SDataView& view, const size_t ix, const uint32_t componentCount, const EPlyScalarType scalarType, const bool flipVectors, uint8_t*& dst)
{
    if (!dst)
        return false;

    switch (scalarType)
    {
        case EPlyScalarType::Float64:
        case EPlyScalarType::Float32:
        {
            hlsl::float64_t4 tmp = {};
            if (!decodeVec4(view, ix, tmp))
                return false;
            for (uint32_t c = 0u; c < componentCount; ++c)
            {
                double value = (&tmp.x)[c];
                if (flipVectors && c == 0u)
                    value = -value;
                if (scalarType == EPlyScalarType::Float64)
                {
                    const double typed = value;
                    std::memcpy(dst, &typed, sizeof(typed));
                    dst += sizeof(typed);
                }
                else
                {
                    const float typed = static_cast<float>(value);
                    std::memcpy(dst, &typed, sizeof(typed));
                    dst += sizeof(typed);
                }
            }
            return true;
        }
        case EPlyScalarType::Int8:
        case EPlyScalarType::Int16:
        case EPlyScalarType::Int32:
        {
            int64_t tmp[4] = {};
            if (!decodeSigned4Raw(view, ix, tmp))
                return false;
            for (uint32_t c = 0u; c < componentCount; ++c)
            {
                int64_t value = tmp[c];
                if (flipVectors && c == 0u)
                    value = -value;
                switch (scalarType)
                {
                    case EPlyScalarType::Int8:
                    {
                        const int8_t typed = static_cast<int8_t>(value);
                        std::memcpy(dst, &typed, sizeof(typed));
                        dst += sizeof(typed);
                        break;
                    }
                    case EPlyScalarType::Int16:
                    {
                        const int16_t typed = static_cast<int16_t>(value);
                        std::memcpy(dst, &typed, sizeof(typed));
                        dst += sizeof(typed);
                        break;
                    }
                    default:
                    {
                        const int32_t typed = static_cast<int32_t>(value);
                        std::memcpy(dst, &typed, sizeof(typed));
                        dst += sizeof(typed);
                        break;
                    }
                }
            }
            return true;
        }
        case EPlyScalarType::UInt8:
        case EPlyScalarType::UInt16:
        case EPlyScalarType::UInt32:
        {
            uint64_t tmp[4] = {};
            if (!decodeUnsigned4Raw(view, ix, tmp))
                return false;
            for (uint32_t c = 0u; c < componentCount; ++c)
            {
                uint64_t value = tmp[c];
                switch (scalarType)
                {
                    case EPlyScalarType::UInt8:
                    {
                        const uint8_t typed = static_cast<uint8_t>(value);
                        std::memcpy(dst, &typed, sizeof(typed));
                        dst += sizeof(typed);
                        break;
                    }
                    case EPlyScalarType::UInt16:
                    {
                        const uint16_t typed = static_cast<uint16_t>(value);
                        std::memcpy(dst, &typed, sizeof(typed));
                        dst += sizeof(typed);
                        break;
                    }
                    default:
                    {
                        const uint32_t typed = static_cast<uint32_t>(value);
                        std::memcpy(dst, &typed, sizeof(typed));
                        dst += sizeof(typed);
                        break;
                    }
                }
            }
            return true;
        }
    }
    return false;
}

inline bool writeTypedViewText(std::string& output, const ICPUPolygonGeometry::SDataView& view, const size_t ix, const uint32_t componentCount, const EPlyScalarType scalarType, const bool flipVectors)
{
    switch (scalarType)
    {
        case EPlyScalarType::Float64:
        case EPlyScalarType::Float32:
        {
            hlsl::float64_t4 tmp = {};
            if (!decodeVec4(view, ix, tmp))
                return false;
            for (uint32_t c = 0u; c < componentCount; ++c)
            {
                double value = (&tmp.x)[c];
                if (flipVectors && c == 0u)
                    value = -value;
                appendFloatFixed6(output, value);
                output.push_back(' ');
            }
            return true;
        }
        case EPlyScalarType::Int8:
        case EPlyScalarType::Int16:
        case EPlyScalarType::Int32:
        {
            int64_t tmp[4] = {};
            if (!decodeSigned4Raw(view, ix, tmp))
                return false;
            for (uint32_t c = 0u; c < componentCount; ++c)
            {
                int64_t value = tmp[c];
                if (flipVectors && c == 0u)
                    value = -value;
                appendInt(output, value);
                output.push_back(' ');
            }
            return true;
        }
        case EPlyScalarType::UInt8:
        case EPlyScalarType::UInt16:
        case EPlyScalarType::UInt32:
        {
            uint64_t tmp[4] = {};
            if (!decodeUnsigned4Raw(view, ix, tmp))
                return false;
            for (uint32_t c = 0u; c < componentCount; ++c)
            {
                appendUInt(output, tmp[c]);
                output.push_back(' ');
            }
            return true;
        }
    }
    return false;
}

struct SExtraAuxView
{
    const ICPUPolygonGeometry::SDataView* view = nullptr;
    uint32_t components = 0u;
    uint32_t auxIndex = 0u;
    EPlyScalarType scalarType = EPlyScalarType::Float32;
};

bool writeBinary(
    const ICPUPolygonGeometry* geom,
    const EPlyScalarType positionScalarType,
    const ICPUPolygonGeometry::SDataView* uvView,
    const EPlyScalarType uvScalarType,
    const core::vector<SExtraAuxView>& extraAuxViews,
    const bool writeNormals,
    const EPlyScalarType normalScalarType,
    const size_t vertexCount,
    const uint32_t* indices,
    const size_t faceCount,
    const bool write16BitIndices,
    uint8_t* dst,
    const bool flipVectors);
bool writeText(
    const ICPUPolygonGeometry* geom,
    const EPlyScalarType positionScalarType,
    const ICPUPolygonGeometry::SDataView* uvView,
    const EPlyScalarType uvScalarType,
    const core::vector<SExtraAuxView>& extraAuxViews,
    const bool writeNormals,
    const EPlyScalarType normalScalarType,
    const size_t vertexCount,
    const uint32_t* indices,
    const size_t faceCount,
    std::string& output,
    const bool flipVectors);

} // namespace ply_writer_detail

bool CPLYMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    using namespace ply_writer_detail;
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
        if (channels == 2u)
        {
            uvView = &view;
            break;
        }
    }

    core::vector<SExtraAuxView> extraAuxViews;
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
        extraAuxViews.push_back({ &view, components, auxIx, selectPlyScalarType(view.composed.format) });
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
    const bool flipVectors = !(flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);
    const bool write16BitIndices = vertexCount <= static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1ull;

    EPlyScalarType positionScalarType = selectPlyScalarType(positionView.composed.format);
    if (flipVectors && getPlyScalarMeta(positionScalarType).integer && !getPlyScalarMeta(positionScalarType).signedType)
        positionScalarType = EPlyScalarType::Float32;
    EPlyScalarType normalScalarType = selectPlyScalarType(normalView.composed.format);
    if (flipVectors && getPlyScalarMeta(normalScalarType).integer && !getPlyScalarMeta(normalScalarType).signedType)
        normalScalarType = EPlyScalarType::Float32;
    const EPlyScalarType uvScalarType = uvView ? selectPlyScalarType(uvView->composed.format) : EPlyScalarType::Float32;

    const auto positionMeta = getPlyScalarMeta(positionScalarType);
    const auto normalMeta = getPlyScalarMeta(normalScalarType);
    const auto uvMeta = getPlyScalarMeta(uvScalarType);

    size_t extraAuxBytesPerVertex = 0ull;
    for (const auto& extra : extraAuxViews)
        extraAuxBytesPerVertex += static_cast<size_t>(extra.components) * getPlyScalarMeta(extra.scalarType).byteSize;

    std::string header = "ply\n";
    header += binary ? "format binary_little_endian 1.0" : "format ascii 1.0";
    header += "\ncomment Nabla ";
    header += NABLA_SDK_VERSION;

    header += "\nelement vertex ";
    header += std::to_string(vertexCount);
    header += "\n";

    header += "property ";
    header += positionMeta.name;
    header += " x\n";
    header += "property ";
    header += positionMeta.name;
    header += " y\n";
    header += "property ";
    header += positionMeta.name;
    header += " z\n";

    if (writeNormals)
    {
        header += "property ";
        header += normalMeta.name;
        header += " nx\n";
        header += "property ";
        header += normalMeta.name;
        header += " ny\n";
        header += "property ";
        header += normalMeta.name;
        header += " nz\n";
    }

    if (uvView)
    {
        header += "property ";
        header += uvMeta.name;
        header += " u\n";
        header += "property ";
        header += uvMeta.name;
        header += " v\n";
    }

    for (const auto& extra : extraAuxViews)
    {
        const auto extraMeta = getPlyScalarMeta(extra.scalarType);
        for (uint32_t component = 0u; component < extra.components; ++component)
        {
            header += "property ";
            header += extraMeta.name;
            header += " aux";
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
    header += write16BitIndices ? "\nproperty list uchar uint16 vertex_indices\n" : "\nproperty list uchar uint32 vertex_indices\n";
    header += "end_header\n";

    bool writeOk = false;
    size_t outputBytes = 0ull;
    if (binary)
    {
        const size_t vertexStride =
            static_cast<size_t>(positionMeta.byteSize) * 3ull +
            (writeNormals ? static_cast<size_t>(normalMeta.byteSize) * 3ull : 0ull) +
            (uvView ? static_cast<size_t>(uvMeta.byteSize) * 2ull : 0ull) +
            extraAuxBytesPerVertex;
        const size_t faceStride = sizeof(uint8_t) + (write16BitIndices ? sizeof(uint16_t) : sizeof(uint32_t)) * 3u;
        const size_t bodySize = vertexCount * vertexStride + faceCount * faceStride;

        core::vector<uint8_t> body;
        body.resize(bodySize);
        if (!writeBinary(geom, positionScalarType, uvView, uvScalarType, extraAuxViews, writeNormals, normalScalarType, vertexCount, indices, faceCount, write16BitIndices, body.data(), flipVectors))
            return false;

        const size_t outputSize = header.size() + body.size();
        const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(outputSize), true);
        if (!ioPlan.valid)
        {
            _params.logger.log("PLY writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), ioPlan.reason);
            return false;
        }

        outputBytes = outputSize;
        writeOk = writeTwoBuffersWithPolicy(
            file,
            ioPlan,
            reinterpret_cast<const uint8_t*>(header.data()),
            header.size(),
            body.data(),
            body.size(),
            &ioTelemetry);
        const uint64_t ioMinWrite = ioTelemetry.getMinOrZero();
        const uint64_t ioAvgWrite = ioTelemetry.getAvgOrZero();
        if (isTinyIOTelemetryLikely(ioTelemetry, static_cast<uint64_t>(outputBytes), _params.ioPolicy))
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
        return writeOk;
    }

    std::string body;
    body.reserve(vertexCount * ApproxPlyTextBytesPerVertex + faceCount * ApproxPlyTextBytesPerFace);
    if (!writeText(geom, positionScalarType, uvView, uvScalarType, extraAuxViews, writeNormals, normalScalarType, vertexCount, indices, faceCount, body, flipVectors))
        return false;

    const size_t outputSize = header.size() + body.size();
    const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(outputSize), true);
    if (!ioPlan.valid)
    {
        _params.logger.log("PLY writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), ioPlan.reason);
        return false;
    }

    outputBytes = outputSize;
    writeOk = writeTwoBuffersWithPolicy(
        file,
        ioPlan,
        reinterpret_cast<const uint8_t*>(header.data()),
        header.size(),
        reinterpret_cast<const uint8_t*>(body.data()),
        body.size(),
        &ioTelemetry);
    const uint64_t ioMinWrite = ioTelemetry.getMinOrZero();
    const uint64_t ioAvgWrite = ioTelemetry.getAvgOrZero();
    if (isTinyIOTelemetryLikely(ioTelemetry, static_cast<uint64_t>(outputBytes), _params.ioPolicy))
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
    return writeOk;
}

bool ply_writer_detail::writeBinary(
    const ICPUPolygonGeometry* geom,
    const EPlyScalarType positionScalarType,
    const ICPUPolygonGeometry::SDataView* uvView,
    const EPlyScalarType uvScalarType,
    const core::vector<SExtraAuxView>& extraAuxViews,
    const bool writeNormals,
    const EPlyScalarType normalScalarType,
    const size_t vertexCount,
    const uint32_t* indices,
    const size_t faceCount,
    const bool write16BitIndices,
    uint8_t* dst,
    const bool flipVectors)
{
    if (!dst)
        return false;

    const auto& positionView = geom->getPositionView();
    const auto& normalView = geom->getNormalView();

    for (size_t i = 0; i < vertexCount; ++i)
    {
        if (!writeTypedViewBinary(positionView, i, 3u, positionScalarType, flipVectors, dst))
            return false;

        if (writeNormals && !writeTypedViewBinary(normalView, i, 3u, normalScalarType, flipVectors, dst))
            return false;

        if (uvView && !writeTypedViewBinary(*uvView, i, 2u, uvScalarType, false, dst))
            return false;

        for (const auto& extra : extraAuxViews)
        {
            if (!extra.view || !writeTypedViewBinary(*extra.view, i, extra.components, extra.scalarType, false, dst))
                return false;
        }
    }

    for (size_t i = 0; i < faceCount; ++i)
    {
        const uint8_t listSize = 3u;
        *dst++ = listSize;

        const uint32_t* tri = indices + (i * 3u);
        if (write16BitIndices)
        {
            const uint16_t tri16[3] = {
                static_cast<uint16_t>(tri[0]),
                static_cast<uint16_t>(tri[1]),
                static_cast<uint16_t>(tri[2])
            };
            std::memcpy(dst, tri16, sizeof(tri16));
            dst += sizeof(tri16);
        }
        else
        {
            std::memcpy(dst, tri, sizeof(uint32_t) * 3u);
            dst += sizeof(uint32_t) * 3u;
        }
    }

    return true;
}

bool ply_writer_detail::writeText(
    const ICPUPolygonGeometry* geom,
    const EPlyScalarType positionScalarType,
    const ICPUPolygonGeometry::SDataView* uvView,
    const EPlyScalarType uvScalarType,
    const core::vector<SExtraAuxView>& extraAuxViews,
    const bool writeNormals,
    const EPlyScalarType normalScalarType,
    const size_t vertexCount,
    const uint32_t* indices,
    const size_t faceCount,
    std::string& output,
    const bool flipVectors)
{
    const auto& positionView = geom->getPositionView();
    const auto& normalView = geom->getNormalView();

    for (size_t i = 0; i < vertexCount; ++i)
    {
        if (!writeTypedViewText(output, positionView, i, 3u, positionScalarType, flipVectors))
            return false;

        if (writeNormals)
        {
            if (!writeTypedViewText(output, normalView, i, 3u, normalScalarType, flipVectors))
                return false;
        }

        if (uvView)
        {
            if (!writeTypedViewText(output, *uvView, i, 2u, uvScalarType, false))
                return false;
        }

        for (const auto& extra : extraAuxViews)
        {
            if (!extra.view || !writeTypedViewText(output, *extra.view, i, extra.components, extra.scalarType, false))
                return false;
        }

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

