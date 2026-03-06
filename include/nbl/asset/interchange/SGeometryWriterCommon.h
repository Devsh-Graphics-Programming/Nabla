// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_WRITER_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_WRITER_COMMON_H_INCLUDED_


#include "nbl/asset/ICPUScene.h"
#include "nbl/asset/ICPUPolygonGeometry.h"

#include <charconv>
#include <cstdio>
#include <limits>
#include <system_error>


namespace nbl::asset
{

class SGeometryWriterCommon
{
    public:
        static inline const ICPUPolygonGeometry* resolvePolygonGeometry(const IAsset* rootAsset)
        {
            if (!rootAsset)
                return nullptr;

            if (const auto* geometry = IAsset::castDown<const ICPUPolygonGeometry>(rootAsset))
                return geometry;

            const auto* scene = IAsset::castDown<const ICPUScene>(rootAsset);
            if (!scene)
                return nullptr;

            for (const auto& morphTargetsRef : scene->getInstances().getMorphTargets())
            {
                const auto* morphTargets = morphTargetsRef.get();
                if (!morphTargets)
                    continue;
                for (const auto& target : morphTargets->getTargets())
                {
                    const auto* const collection = target.geoCollection.get();
                    if (!collection)
                        continue;
                    for (const auto& geoRef : collection->getGeometries())
                    {
                        if (const auto* geometry = IAsset::castDown<const ICPUPolygonGeometry>(geoRef.geometry.get()))
                            return geometry;
                    }
                }
            }

            return nullptr;
        }

        static inline const ICPUPolygonGeometry::SDataView* findFirstAuxViewByChannelCount(const ICPUPolygonGeometry* geom, const uint32_t channels, const size_t requiredElementCount = 0ull)
        {
            if (!geom || channels == 0u)
                return nullptr;

            for (const auto& view : geom->getAuxAttributeViews())
            {
                if (!view)
                    continue;
                if (requiredElementCount && view.getElementCount() != requiredElementCount)
                    continue;
                if (getFormatChannelCount(view.composed.format) == channels)
                    return &view;
            }

            return nullptr;
        }

        static inline bool decodeTriangleIndices(const ICPUPolygonGeometry* geom, core::vector<uint32_t>& indexData, const uint32_t*& outIndices, size_t& outFaceCount)
        {
            outIndices = nullptr;
            outFaceCount = 0ull;
            if (!geom)
                return false;

            const auto& positionView = geom->getPositionView();
            const size_t vertexCount = positionView.getElementCount();
            if (vertexCount == 0ull)
                return false;

            const auto& indexView = geom->getIndexView();
            if (indexView)
            {
                const size_t indexCount = indexView.getElementCount();
                if ((indexCount % 3ull) != 0ull)
                    return false;

                const void* src = indexView.getPointer();
                if (!src)
                    return false;

                if (indexView.composed.format == EF_R32_UINT && indexView.composed.getStride() == sizeof(uint32_t))
                {
                    outIndices = reinterpret_cast<const uint32_t*>(src);
                }
                else if (indexView.composed.format == EF_R16_UINT && indexView.composed.getStride() == sizeof(uint16_t))
                {
                    indexData.resize(indexCount);
                    const auto* src16 = reinterpret_cast<const uint16_t*>(src);
                    for (size_t i = 0ull; i < indexCount; ++i)
                        indexData[i] = src16[i];
                    outIndices = indexData.data();
                }
                else
                {
                    indexData.resize(indexCount);
                    hlsl::vector<uint32_t, 1> decoded = {};
                    for (size_t i = 0ull; i < indexCount; ++i)
                    {
                        if (!indexView.decodeElement(i, decoded))
                            return false;
                        indexData[i] = decoded.x;
                    }
                    outIndices = indexData.data();
                }

                for (size_t i = 0ull; i < indexCount; ++i)
                    if (outIndices[i] >= vertexCount)
                        return false;

                outFaceCount = indexCount / 3ull;
                return true;
            }

            if ((vertexCount % 3ull) != 0ull)
                return false;

            indexData.resize(vertexCount);
            for (size_t i = 0ull; i < vertexCount; ++i)
                indexData[i] = static_cast<uint32_t>(i);
            outIndices = indexData.data();
            outFaceCount = vertexCount / 3ull;
            return true;
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

        static inline char* appendFloatFixed6ToBuffer(char* dst, char* const end, const float value)
        {
            if (!dst || dst >= end)
                return end;

            const auto result = std::to_chars(dst, end, value, std::chars_format::fixed, 6);
            if (result.ec == std::errc())
                return result.ptr;

            const int written = std::snprintf(dst, static_cast<size_t>(end - dst), "%.6f", static_cast<double>(value));
            if (written <= 0)
                return dst;
            const size_t writeLen = static_cast<size_t>(written);
            return (writeLen < static_cast<size_t>(end - dst)) ? (dst + writeLen) : end;
        }

        static inline char* appendUIntToBuffer(char* dst, char* const end, const uint32_t value)
        {
            if (!dst || dst >= end)
                return end;

            const auto result = std::to_chars(dst, end, value);
            if (result.ec == std::errc())
                return result.ptr;

            const int written = std::snprintf(dst, static_cast<size_t>(end - dst), "%u", value);
            if (written <= 0)
                return dst;
            const size_t writeLen = static_cast<size_t>(written);
            return (writeLen < static_cast<size_t>(end - dst)) ? (dst + writeLen) : end;
        }
};

}


#endif
