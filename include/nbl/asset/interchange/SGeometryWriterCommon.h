// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_WRITER_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_WRITER_COMMON_H_INCLUDED_


#include <concepts>
#include "nbl/asset/ICPUScene.h"
#include "nbl/asset/ICPUGeometryCollection.h"
#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"

#include <charconv>
#include <cstdio>
#include <system_error>


namespace nbl::asset
{

namespace impl
{

inline hlsl::float32_t3x4 identityAffineTransform()
{
    return hlsl::math::linalg::diagonal<hlsl::float32_t3x4>(1.f);
}

template<typename Container> concept PolygonGeometryWriteItemContainer = requires(Container& c, const ICPUPolygonGeometry* geometry, const hlsl::float32_t3x4 transform, const uint32_t instanceIx, const uint32_t targetIx, const uint32_t geometryIx) { c.emplace_back(geometry, transform, instanceIx, targetIx, geometryIx); };

template<typename Container>
static inline void appendPolygonGeometryWriteItemsFromCollection(Container& out, const ICPUGeometryCollection* collection, const hlsl::float32_t3x4& parentTransform, const uint32_t instanceIx, const uint32_t targetIx)
{
    if (!collection)
        return;

    const auto identity = identityAffineTransform();
    const auto& geometries = collection->getGeometries();
    for (uint32_t geometryIx = 0u; geometryIx < geometries.size(); ++geometryIx)
    {
        const auto& ref = geometries[geometryIx];
        if (!ref.geometry || ref.geometry->getPrimitiveType() != IGeometryBase::EPrimitiveType::Polygon)
            continue;
        const auto* geometry = static_cast<const ICPUPolygonGeometry*>(ref.geometry.get());
        const auto localTransform = ref.hasTransform() ? ref.transform : identity;
        out.emplace_back(geometry, hlsl::math::linalg::promoted_mul(parentTransform, localTransform), instanceIx, targetIx, geometryIx);
    }
}

}

class SGeometryWriterCommon
{
    public:
        struct SPolygonGeometryWriteItem
        {
            inline SPolygonGeometryWriteItem(const ICPUPolygonGeometry* _geometry, const hlsl::float32_t3x4& _transform, const uint32_t _instanceIx, const uint32_t _targetIx, const uint32_t _geometryIx) : geometry(_geometry), transform(_transform), instanceIx(_instanceIx), targetIx(_targetIx), geometryIx(_geometryIx) {}

            const ICPUPolygonGeometry* geometry = nullptr;
            hlsl::float32_t3x4 transform = impl::identityAffineTransform();
            uint32_t instanceIx = ~0u;
            uint32_t targetIx = ~0u;
            uint32_t geometryIx = 0u;
        };

        template<typename Container = core::vector<SPolygonGeometryWriteItem>> requires impl::PolygonGeometryWriteItemContainer<Container>
        static inline Container collectPolygonGeometryWriteItems(const IAsset* rootAsset)
        {
            Container out = {};
            if (!rootAsset)
                return out;

            const auto identity = impl::identityAffineTransform();
            if (rootAsset->getAssetType() == IAsset::ET_GEOMETRY)
            {
                const auto* geometry = static_cast<const IGeometry<ICPUBuffer>*>(rootAsset);
                if (geometry->getPrimitiveType() == IGeometryBase::EPrimitiveType::Polygon)
                    out.emplace_back(static_cast<const ICPUPolygonGeometry*>(rootAsset), identity, ~0u, ~0u, 0u);
                return out;
            }

            if (rootAsset->getAssetType() == IAsset::ET_GEOMETRY_COLLECTION)
            {
                impl::appendPolygonGeometryWriteItemsFromCollection(out, static_cast<const ICPUGeometryCollection*>(rootAsset), identity, ~0u, ~0u);
                return out;
            }

            if (rootAsset->getAssetType() != IAsset::ET_SCENE)
                return out;
            const auto* scene = static_cast<const ICPUScene*>(rootAsset);

            const auto& instances = scene->getInstances();
            const auto& morphTargets = instances.getMorphTargets();
            const auto& initialTransforms = instances.getInitialTransforms();
            for (uint32_t instanceIx = 0u; instanceIx < morphTargets.size(); ++instanceIx)
            {
                const auto* targets = morphTargets[instanceIx].get();
                if (!targets)
                    continue;

                const auto instanceTransform = initialTransforms.empty() ? identity : initialTransforms[instanceIx];
                const auto& targetList = targets->getTargets();
                for (uint32_t targetIx = 0u; targetIx < targetList.size(); ++targetIx)
                    impl::appendPolygonGeometryWriteItemsFromCollection(out, targetList[targetIx].geoCollection.get(), instanceTransform, instanceIx, targetIx);
            }

            return out;
        }

        static inline bool isIdentityTransform(const hlsl::float32_t3x4& transform)
        {
            return
                transform[0].x == 1.f && transform[0].y == 0.f && transform[0].z == 0.f && transform[0].w == 0.f &&
                transform[1].x == 0.f && transform[1].y == 1.f && transform[1].z == 0.f && transform[1].w == 0.f &&
                transform[2].x == 0.f && transform[2].y == 0.f && transform[2].z == 1.f && transform[2].w == 0.f;
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

        template<typename T, E_FORMAT ExpectedFormat>
        static inline const T* getTightView(const ICPUPolygonGeometry::SDataView& view)
        {
            if (!view)
                return nullptr;
            if (view.composed.format != ExpectedFormat)
                return nullptr;
            if (view.composed.getStride() != sizeof(T))
                return nullptr;
            return reinterpret_cast<const T*>(view.getPointer());
        }

        static char* appendFloatToBuffer(char* dst, char* end, float value);
        static char* appendFloatToBuffer(char* dst, char* end, double value);

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
