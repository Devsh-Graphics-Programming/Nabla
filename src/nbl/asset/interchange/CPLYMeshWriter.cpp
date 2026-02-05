// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CPLYMeshWriter.h"

#ifdef _NBL_COMPILE_WITH_PLY_WRITER_

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"
#include "nbl/asset/utils/CPolygonGeometryManipulator.h"

#include <sstream>
#include <iomanip>

namespace nbl::asset
{

CPLYMeshWriter::CPLYMeshWriter()
{
    #ifdef _NBL_DEBUG
    setDebugName("CPLYMeshWriter");
    #endif
}

static inline bool decodeVec4(const ICPUPolygonGeometry::SDataView& view, const size_t ix, hlsl::float64_t4& out)
{
    out = hlsl::float64_t4(0.0, 0.0, 0.0, 0.0);
    return view.decodeElement(ix, out);
}

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

        indexData.resize(indexCount);
        const void* src = indexView.getPointer();
        if (!src)
            return false;

        if (indexView.composed.format == EF_R32_UINT)
        {
            memcpy(indexData.data(), src, indexCount * sizeof(uint32_t));
        }
        else if (indexView.composed.format == EF_R16_UINT)
        {
            const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
            for (size_t i = 0; i < indexCount; ++i)
                indexData[i] = src16[i];
        }
        else
        {
            return false;
        }

        indices = indexData.data();
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

    SContext context = { SAssetWriteContext{ ctx.params, file } };
    {
        system::IFile::success_t success;
        file->write(success, header.c_str(), context.fileOffset, header.size());
        context.fileOffset += success.getBytesProcessed();
    }

    if (binary)
        writeBinary(geom, uvView, writeNormals, vertexCount, indices, faceCount, context);
    else
        writeText(geom, uvView, writeNormals, vertexCount, indices, faceCount, context);

    return true;
}

void CPLYMeshWriter::writeBinary(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, SContext& context) const
{
    const bool flipVectors = !(context.writeContext.params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);
    const size_t vertexStride = sizeof(float) * (3u + (writeNormals ? 3u : 0u) + (uvView ? 2u : 0u));
    const size_t faceStride = sizeof(uint8_t) + sizeof(uint32_t) * 3u;
    const size_t totalSize = vertexCount * vertexStride + faceCount * faceStride;
    core::vector<uint8_t> blob;
    blob.resize(totalSize);
    uint8_t* dst = blob.data();

    hlsl::float64_t4 tmp = {};
    for (size_t i = 0; i < vertexCount; ++i)
    {
        if (!decodeVec4(geom->getPositionView(), i, tmp))
            return;

        float pos[3] = { static_cast<float>(tmp.x), static_cast<float>(tmp.y), static_cast<float>(tmp.z) };
        if (flipVectors)
            pos[0] = -pos[0];

        memcpy(dst, pos, sizeof(pos));
        dst += sizeof(pos);

        if (writeNormals)
        {
            if (!decodeVec4(geom->getNormalView(), i, tmp))
                return;
            float normal[3] = { static_cast<float>(tmp.x), static_cast<float>(tmp.y), static_cast<float>(tmp.z) };
            if (flipVectors)
                normal[0] = -normal[0];

            memcpy(dst, normal, sizeof(normal));
            dst += sizeof(normal);
        }

        if (uvView)
        {
            if (!decodeVec4(*uvView, i, tmp))
                return;
            float uv[2] = { static_cast<float>(tmp.x), static_cast<float>(tmp.y) };

            memcpy(dst, uv, sizeof(uv));
            dst += sizeof(uv);
        }
    }

    for (size_t i = 0; i < faceCount; ++i)
    {
        const uint8_t listSize = 3u;
        *dst++ = listSize;

        const uint32_t* tri = indices + (i * 3u);
        memcpy(dst, tri, sizeof(uint32_t) * 3u);
        dst += sizeof(uint32_t) * 3u;
    }

    system::IFile::success_t success;
    context.writeContext.outputFile->write(success, blob.data(), context.fileOffset, blob.size());
    context.fileOffset += success.getBytesProcessed();
}

void CPLYMeshWriter::writeText(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView* uvView, bool writeNormals, size_t vertexCount, const uint32_t* indices, size_t faceCount, SContext& context) const
{
    const bool flipVectors = !(context.writeContext.params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);

    hlsl::float64_t4 tmp = {};
    for (size_t i = 0; i < vertexCount; ++i)
    {
        if (!decodeVec4(geom->getPositionView(), i, tmp))
            return;
        const double pos[3] = { tmp.x, tmp.y, tmp.z };
        writeVectorAsText(context, pos, 3u, flipVectors);

        if (writeNormals)
        {
            if (!decodeVec4(geom->getNormalView(), i, tmp))
                return;
            const double normal[3] = { tmp.x, tmp.y, tmp.z };
            writeVectorAsText(context, normal, 3u, flipVectors);
        }

        if (uvView)
        {
            if (!decodeVec4(*uvView, i, tmp))
                return;
            const double uv[2] = { tmp.x, tmp.y };
            writeVectorAsText(context, uv, 2u, false);
        }

        system::IFile::success_t success;
        context.writeContext.outputFile->write(success, "\n", context.fileOffset, 1);
        context.fileOffset += success.getBytesProcessed();
    }

    for (size_t i = 0; i < faceCount; ++i)
    {
        const uint32_t* tri = indices + (i * 3u);
        std::stringstream ss;
        ss << "3 " << tri[0] << " " << tri[1] << " " << tri[2] << "\n";
        const auto str = ss.str();

        system::IFile::success_t success;
        context.writeContext.outputFile->write(success, str.c_str(), context.fileOffset, str.size());
        context.fileOffset += success.getBytesProcessed();
    }
}

} // namespace nbl::asset

#endif // _NBL_COMPILE_WITH_PLY_WRITER_
