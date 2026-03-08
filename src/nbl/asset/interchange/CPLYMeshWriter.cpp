// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CPLYMeshWriter.h"
#include "nbl/asset/interchange/SGeometryWriterCommon.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "SPLYPolygonGeometryAuxLayout.h"

#ifdef _NBL_COMPILE_WITH_PLY_WRITER_

#include "nbl/system/IFile.h"

#include <algorithm>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <array>
#include <charconv>
#include <cstdio>
#include <limits>
#include <sstream>
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

writer_flags_t CPLYMeshWriter::getSupportedFlags()
{
    return asset::EWF_BINARY;
}

writer_flags_t CPLYMeshWriter::getForcedFlags()
{
    return EWF_NONE;
}

namespace ply_writer_detail
{

constexpr size_t ApproxPlyTextBytesPerVertex = sizeof("0.000000 0.000000 0.000000 0.000000 0.000000 0.000000\n") - 1ull;
constexpr size_t ApproxPlyTextBytesPerFace = sizeof("3 4294967295 4294967295 4294967295\n") - 1ull;

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

bool isPlySupportedScalarFormat(const E_FORMAT format)
{
    if (format == EF_UNKNOWN)
        return false;

    const uint32_t channels = getFormatChannelCount(format);
    if (channels == 0u)
        return false;

    if (!(isIntegerFormat(format) || isFloatingPointFormat(format) || isNormalizedFormat(format) || isScaledFormat(format)))
        return false;

    const auto bytesPerPixel = getBytesPerPixel(format);
    if (bytesPerPixel.getDenominator() != 1u)
        return false;
    const uint32_t pixelBytes = bytesPerPixel.getNumerator();
    if (pixelBytes == 0u || (pixelBytes % channels) != 0u)
        return false;

    const uint32_t bytesPerChannel = pixelBytes / channels;
    return bytesPerChannel == 1u || bytesPerChannel == 2u || bytesPerChannel == 4u || bytesPerChannel == 8u;
}

EPlyScalarType selectPlyScalarType(const E_FORMAT format)
{
    if (!isPlySupportedScalarFormat(format))
        return EPlyScalarType::Float32;
    if (isNormalizedFormat(format) || isScaledFormat(format))
        return EPlyScalarType::Float32;

    const uint32_t channels = getFormatChannelCount(format);
    if (channels == 0u)
    {
        assert(format == EF_UNKNOWN);
        return EPlyScalarType::Float32;
    }

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

template<typename T>
void appendIntegral(std::string& out, const T value)
{
    std::array<char, 32> buf = {};
    const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), value);
    if (res.ec == std::errc())
        out.append(buf.data(), static_cast<size_t>(res.ptr - buf.data()));
}

constexpr size_t MaxFloatTextChars = std::numeric_limits<double>::max_digits10 + 16ull;

void appendFloat(std::string& out, double value)
{
    const size_t oldSize = out.size();
    out.resize(oldSize + MaxFloatTextChars);
    char* const begin = out.data() + oldSize;
    char* const end = begin + MaxFloatTextChars;
    char* const cursor = SGeometryWriterCommon::appendFloatToBuffer(begin, end, value);
    out.resize(oldSize + static_cast<size_t>(cursor - begin));
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
                appendFloat(output, value);
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
                appendIntegral(output, value);
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
                appendIntegral(output, tmp[c]);
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

struct SWriteInput
{
    const ICPUPolygonGeometry* geom = nullptr;
    EPlyScalarType positionScalarType = EPlyScalarType::Float32;
    const ICPUPolygonGeometry::SDataView* uvView = nullptr;
    EPlyScalarType uvScalarType = EPlyScalarType::Float32;
    const core::vector<SExtraAuxView>* extraAuxViews = nullptr;
    bool writeNormals = false;
    EPlyScalarType normalScalarType = EPlyScalarType::Float32;
    size_t vertexCount = 0ull;
    size_t faceCount = 0ull;
    bool write16BitIndices = false;
    bool flipVectors = false;
};

bool writeBinary(
    const SWriteInput& input,
    uint8_t* dst);
bool writeText(
    const SWriteInput& input,
    std::string& output);

} // namespace ply_writer_detail

bool CPLYMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    using namespace ply_writer_detail;
    SFileWriteTelemetry ioTelemetry = {};

    if (!_override)
        getDefaultOverride(_override);

    if (!_file || !_params.rootAsset)
    {
        _params.logger.log("PLY writer: missing output file or root asset.", system::ILogger::ELL_ERROR);
        return false;
    }

    const auto items = SGeometryWriterCommon::collectPolygonGeometryWriteItems(_params.rootAsset);
    if (items.size() != 1u)
    {
        _params.logger.log("PLY writer: expected exactly one polygon geometry to write.", system::ILogger::ELL_ERROR);
        return false;
    }
    const auto& item = items.front();
    const auto* geom = item.geometry;
    if (!geom || !geom->valid())
    {
        _params.logger.log("PLY writer: root asset is not a valid polygon geometry.", system::ILogger::ELL_ERROR);
        return false;
    }
    if (!SGeometryWriterCommon::isIdentityTransform(item.transform))
    {
        _params.logger.log("PLY writer: transformed scene or collection export is not supported.", system::ILogger::ELL_ERROR);
        return false;
    }

    SAssetWriteContext ctx = { _params, _file };
    system::IFile* file = _override->getOutputFile(_file, ctx, { geom, 0u });
    if (!file)
    {
        _params.logger.log("PLY writer: output override returned null file.", system::ILogger::ELL_ERROR);
        return false;
    }

    const auto& positionView = geom->getPositionView();
    const auto& normalView = geom->getNormalView();
    const size_t vertexCount = positionView.getElementCount();
    if (vertexCount == 0ull)
    {
        _params.logger.log("PLY writer: empty position view.", system::ILogger::ELL_ERROR);
        return false;
    }
    const bool writeNormals = static_cast<bool>(normalView);
    if (writeNormals && normalView.getElementCount() != vertexCount)
    {
        _params.logger.log("PLY writer: normal vertex count mismatch.", system::ILogger::ELL_ERROR);
        return false;
    }

    const ICPUPolygonGeometry::SDataView* uvView = SGeometryWriterCommon::getAuxViewAt(geom, SPLYPolygonGeometryAuxLayout::UV0, vertexCount);
    if (uvView && getFormatChannelCount(uvView->composed.format) != 2u)
        uvView = nullptr;

    core::vector<SExtraAuxView> extraAuxViews;
    const auto& auxViews = geom->getAuxAttributeViews();
    extraAuxViews.reserve(auxViews.size());
    for (uint32_t auxIx = 0u; auxIx < static_cast<uint32_t>(auxViews.size()); ++auxIx)
    {
        const auto& view = auxViews[auxIx];
        if (!view || (uvView && auxIx == SPLYPolygonGeometryAuxLayout::UV0))
            continue;
        if (view.getElementCount() != vertexCount)
            continue;
        const uint32_t channels = getFormatChannelCount(view.composed.format);
        if (channels == 0u)
            continue;
        const uint32_t components = std::min(4u, channels);
        extraAuxViews.push_back({ &view, components, auxIx, selectPlyScalarType(view.composed.format) });
    }

    const auto* indexing = geom->getIndexingCallback();
    if (!indexing)
    {
        _params.logger.log("PLY writer: missing indexing callback.", system::ILogger::ELL_ERROR);
        return false;
    }

    if (indexing->knownTopology() != E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST)
    {
        _params.logger.log("PLY writer: only triangle-list topology is supported.", system::ILogger::ELL_ERROR);
        return false;
    }

    size_t faceCount = 0ull;
    if (!SGeometryWriterCommon::getTriangleFaceCount(geom, faceCount))
    {
        _params.logger.log("PLY writer: failed to validate triangle indexing.", system::ILogger::ELL_ERROR);
        return false;
    }

    const auto flags = _override->getAssetWritingFlags(ctx, geom, 0u);
    const bool binary = flags.hasAnyFlag(E_WRITER_FLAGS::EWF_BINARY);
    const bool flipVectors = !flags.hasAnyFlag(E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);
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

    std::ostringstream headerBuilder;
    headerBuilder << "ply\n";
    headerBuilder << (binary ? "format binary_little_endian 1.0" : "format ascii 1.0");
    headerBuilder << "\ncomment Nabla " << NABLA_SDK_VERSION;
    headerBuilder << "\nelement vertex " << vertexCount << "\n";

    headerBuilder << "property " << positionMeta.name << " x\n";
    headerBuilder << "property " << positionMeta.name << " y\n";
    headerBuilder << "property " << positionMeta.name << " z\n";

    if (writeNormals)
    {
        headerBuilder << "property " << normalMeta.name << " nx\n";
        headerBuilder << "property " << normalMeta.name << " ny\n";
        headerBuilder << "property " << normalMeta.name << " nz\n";
    }

    if (uvView)
    {
        headerBuilder << "property " << uvMeta.name << " u\n";
        headerBuilder << "property " << uvMeta.name << " v\n";
    }

    for (const auto& extra : extraAuxViews)
    {
        const auto extraMeta = getPlyScalarMeta(extra.scalarType);
        for (uint32_t component = 0u; component < extra.components; ++component)
        {
            headerBuilder << "property " << extraMeta.name << " aux" << extra.auxIndex;
            if (extra.components > 1u)
                headerBuilder << "_" << component;
            headerBuilder << "\n";
        }
    }

    headerBuilder << "element face " << faceCount;
    headerBuilder << (write16BitIndices ? "\nproperty list uchar uint16 vertex_indices\n" : "\nproperty list uchar uint32 vertex_indices\n");
    headerBuilder << "end_header\n";
    const std::string header = headerBuilder.str();

    const SWriteInput input = {
        .geom = geom,
        .positionScalarType = positionScalarType,
        .uvView = uvView,
        .uvScalarType = uvScalarType,
        .extraAuxViews = &extraAuxViews,
        .writeNormals = writeNormals,
        .normalScalarType = normalScalarType,
        .vertexCount = vertexCount,
        .faceCount = faceCount,
        .write16BitIndices = write16BitIndices,
        .flipVectors = flipVectors
    };

    bool writeOk = false;
    size_t outputBytes = 0ull;
    auto writePayload = [&](const uint8_t* bodyData, const size_t bodySize) -> bool
    {
        const size_t outputSize = header.size() + bodySize;
        const bool fileMappable = core::bitflag<system::IFile::E_CREATE_FLAGS>(file->getFlags()).hasAnyFlag(system::IFile::ECF_MAPPABLE);
        const auto ioPlan = SResolvedFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(outputSize), true, fileMappable);
        if (!ioPlan.isValid())
        {
            _params.logger.log("PLY writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), ioPlan.reason);
            return false;
        }

        outputBytes = outputSize;
        const SInterchangeIO::SBufferRange writeBuffers[] =
        {
            { .data = reinterpret_cast<const uint8_t*>(header.data()), .byteCount = header.size() },
            { .data = bodyData, .byteCount = bodySize }
        };
        writeOk = SInterchangeIO::writeBuffersWithPolicy(file, ioPlan, writeBuffers, &ioTelemetry);
        const uint64_t ioMinWrite = ioTelemetry.getMinOrZero();
        const uint64_t ioAvgWrite = ioTelemetry.getAvgOrZero();
        if (SInterchangeIO::isTinyIOTelemetryLikely(ioTelemetry, static_cast<uint64_t>(outputBytes), _params.ioPolicy))
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
            system::to_string(_params.ioPolicy.strategy).c_str(),
            system::to_string(ioPlan.strategy).c_str(),
            static_cast<unsigned long long>(ioPlan.chunkSizeBytes()),
            ioPlan.reason);
        return writeOk;
    };
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
        if (!writeBinary(input, body.data()))
        {
            _params.logger.log("PLY writer: binary payload generation failed.", system::ILogger::ELL_ERROR);
            return false;
        }
        return writePayload(body.data(), body.size());
    }

    std::string body;
    body.reserve(vertexCount * ApproxPlyTextBytesPerVertex + faceCount * ApproxPlyTextBytesPerFace);
    if (!writeText(input, body))
    {
        _params.logger.log("PLY writer: text payload generation failed.", system::ILogger::ELL_ERROR);
        return false;
    }
    return writePayload(reinterpret_cast<const uint8_t*>(body.data()), body.size());
}

bool ply_writer_detail::writeBinary(
    const SWriteInput& input,
    uint8_t* dst)
{
    if (!input.geom || !input.extraAuxViews || !dst)
        return false;

    const auto& positionView = input.geom->getPositionView();
    const auto& normalView = input.geom->getNormalView();
    const auto& extraAuxViews = *input.extraAuxViews;

    for (size_t i = 0; i < input.vertexCount; ++i)
    {
        if (!writeTypedViewBinary(positionView, i, 3u, input.positionScalarType, input.flipVectors, dst))
            return false;

        if (input.writeNormals && !writeTypedViewBinary(normalView, i, 3u, input.normalScalarType, input.flipVectors, dst))
            return false;

        if (input.uvView && !writeTypedViewBinary(*input.uvView, i, 2u, input.uvScalarType, false, dst))
            return false;

        for (const auto& extra : extraAuxViews)
        {
            if (!extra.view || !writeTypedViewBinary(*extra.view, i, extra.components, extra.scalarType, false, dst))
                return false;
        }
    }

    return SGeometryWriterCommon::visitTriangleIndices(input.geom, [&](const uint32_t i0, const uint32_t i1, const uint32_t i2)->bool
    {
        const uint8_t listSize = 3u;
        *dst++ = listSize;

        if (input.write16BitIndices)
        {
            const uint16_t tri16[3] = {
                static_cast<uint16_t>(i0),
                static_cast<uint16_t>(i1),
                static_cast<uint16_t>(i2)
            };
            std::memcpy(dst, tri16, sizeof(tri16));
            dst += sizeof(tri16);
        }
        else
        {
            const uint32_t tri[3] = { i0, i1, i2 };
            std::memcpy(dst, tri, sizeof(tri));
            dst += sizeof(tri);
        }
        return true;
    });
}

bool ply_writer_detail::writeText(
    const SWriteInput& input,
    std::string& output)
{
    if (!input.geom || !input.extraAuxViews)
        return false;

    const auto& positionView = input.geom->getPositionView();
    const auto& normalView = input.geom->getNormalView();
    const auto& extraAuxViews = *input.extraAuxViews;

    for (size_t i = 0; i < input.vertexCount; ++i)
    {
        if (!writeTypedViewText(output, positionView, i, 3u, input.positionScalarType, input.flipVectors))
            return false;

        if (input.writeNormals)
        {
            if (!writeTypedViewText(output, normalView, i, 3u, input.normalScalarType, input.flipVectors))
                return false;
        }

        if (input.uvView)
        {
            if (!writeTypedViewText(output, *input.uvView, i, 2u, input.uvScalarType, false))
                return false;
        }

        for (const auto& extra : extraAuxViews)
        {
            if (!extra.view || !writeTypedViewText(output, *extra.view, i, extra.components, extra.scalarType, false))
                return false;
        }

        output += "\n";
    }

    return SGeometryWriterCommon::visitTriangleIndices(input.geom, [&](const uint32_t i0, const uint32_t i1, const uint32_t i2)
    {
        output.append("3 ");
        appendIntegral(output, i0);
        output.push_back(' ');
        appendIntegral(output, i1);
        output.push_back(' ');
        appendIntegral(output, i2);
        output.push_back('\n');
    });
}

} // namespace nbl::asset

#endif // _NBL_COMPILE_WITH_PLY_WRITER_

