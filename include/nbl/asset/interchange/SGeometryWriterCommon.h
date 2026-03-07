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
#include <type_traits>


namespace nbl::asset
{

namespace impl
{
template<typename Container> concept PolygonGeometryWriteItemContainer = requires(Container& c, const ICPUPolygonGeometry* geometry, const hlsl::float32_t3x4 transform, const uint32_t instanceIx, const uint32_t targetIx, const uint32_t geometryIx) { c.emplace_back(geometry, transform, instanceIx, targetIx, geometryIx); };
}

class SGeometryWriterCommon
{
        template<typename Container>
        struct SPolygonGeometryWriteItemCollector
        {
            static inline void appendFromCollection(Container& out, const ICPUGeometryCollection* collection, const hlsl::float32_t3x4& parentTransform, const uint32_t instanceIx, const uint32_t targetIx)
            {
                if (!collection)
                    return;

                const auto identity = hlsl::math::linalg::identity<hlsl::float32_t3x4>();
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
        };

    public:
        struct SPolygonGeometryWriteItem
        {
            inline SPolygonGeometryWriteItem(const ICPUPolygonGeometry* _geometry, const hlsl::float32_t3x4& _transform, const uint32_t _instanceIx, const uint32_t _targetIx, const uint32_t _geometryIx) : geometry(_geometry), transform(_transform), instanceIx(_instanceIx), targetIx(_targetIx), geometryIx(_geometryIx) {}

            const ICPUPolygonGeometry* geometry = nullptr;
            hlsl::float32_t3x4 transform = hlsl::math::linalg::identity<hlsl::float32_t3x4>();
            uint32_t instanceIx = ~0u;
            uint32_t targetIx = ~0u;
            uint32_t geometryIx = 0u;
        };

        // Collects every polygon geometry a writer can serialize from a geometry, collection, or flattened scene.
        template<typename Container = core::vector<SPolygonGeometryWriteItem>> requires impl::PolygonGeometryWriteItemContainer<Container>
        static inline Container collectPolygonGeometryWriteItems(const IAsset* rootAsset)
        {
            Container out = {};
            if (!rootAsset)
                return out;

            const auto identity = hlsl::math::linalg::identity<hlsl::float32_t3x4>();
            if (rootAsset->getAssetType() == IAsset::ET_GEOMETRY)
            {
                const auto* geometry = static_cast<const IGeometry<ICPUBuffer>*>(rootAsset);
                if (geometry->getPrimitiveType() == IGeometryBase::EPrimitiveType::Polygon)
                    out.emplace_back(static_cast<const ICPUPolygonGeometry*>(rootAsset), identity, ~0u, ~0u, 0u);
                return out;
            }

            if (rootAsset->getAssetType() == IAsset::ET_GEOMETRY_COLLECTION)
            {
                SPolygonGeometryWriteItemCollector<Container>::appendFromCollection(out, static_cast<const ICPUGeometryCollection*>(rootAsset), identity, ~0u, ~0u);
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
                    SPolygonGeometryWriteItemCollector<Container>::appendFromCollection(out, targetList[targetIx].geoCollection.get(), instanceTransform, instanceIx, targetIx);
            }

            return out;
        }

        static inline bool isIdentityTransform(const hlsl::float32_t3x4& transform)
        {
            return transform == hlsl::math::linalg::identity<hlsl::float32_t3x4>();
        }

        // Returns the aux view stored at a specific semantic slot when it exists.
        static inline const ICPUPolygonGeometry::SDataView* getAuxViewAt(const ICPUPolygonGeometry* geom, const uint32_t auxViewIx, const size_t requiredElementCount = 0ull)
        {
            if (!geom)
                return nullptr;

            const auto& auxViews = geom->getAuxAttributeViews();
            if (auxViewIx >= auxViews.size())
                return nullptr;

            const auto& view = auxViews[auxViewIx];
            if (!view)
                return nullptr;
            if (requiredElementCount && view.getElementCount() != requiredElementCount)
                return nullptr;
            return &view;
        }

        // Validates triangle-list indexing and returns the number of faces the writer will emit.
        static inline bool getTriangleFaceCount(const ICPUPolygonGeometry* geom, size_t& outFaceCount)
        {
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
                outFaceCount = indexCount / 3ull;
                return true;
            }

            if ((vertexCount % 3ull) != 0ull)
                return false;

            outFaceCount = vertexCount / 3ull;
            return true;
        }

        // Calls `visitor(i0, i1, i2)` once per triangle after validating indices and normalizing implicit/R16/R32 indexing to uint32_t.
        template<typename Visitor>
        static inline bool visitTriangleIndices(const ICPUPolygonGeometry* geom, Visitor&& visitor)
        {
            if (!geom)
                return false;

            const auto& positionView = geom->getPositionView();
            const size_t vertexCount = positionView.getElementCount();
            if (vertexCount == 0ull)
                return false;

            auto visit = [&]<typename IndexT>(const IndexT i0, const IndexT i1, const IndexT i2)->bool
            {
                const uint32_t u0 = static_cast<uint32_t>(i0);
                const uint32_t u1 = static_cast<uint32_t>(i1);
                const uint32_t u2 = static_cast<uint32_t>(i2);
                if (u0 >= vertexCount || u1 >= vertexCount || u2 >= vertexCount)
                    return false;

                if constexpr (std::is_same_v<std::invoke_result_t<Visitor&, uint32_t, uint32_t, uint32_t>, bool>)
                    return visitor(u0, u1, u2);
                else
                {
                    visitor(u0, u1, u2);
                    return true;
                }
            };

            const auto& indexView = geom->getIndexView();
            if (!indexView)
            {
                if ((vertexCount % 3ull) != 0ull)
                    return false;

                for (uint32_t i = 0u; i < vertexCount; i += 3u)
                    if (!visit(i + 0u, i + 1u, i + 2u))
                        return false;
                return true;
            }

            const size_t indexCount = indexView.getElementCount();
            if ((indexCount % 3ull) != 0ull)
                return false;

            const void* const src = indexView.getPointer();
            if (!src)
                return false;

            auto visitIndexed = [&]<typename IndexT>()->bool
            {
                const auto* indices = reinterpret_cast<const IndexT*>(src);
                for (size_t i = 0ull; i < indexCount; i += 3ull)
                    if (!visit(indices[i + 0ull], indices[i + 1ull], indices[i + 2ull]))
                        return false;
                return true;
            };

            switch (geom->getIndexType())
            {
                case EIT_32BIT: return visitIndexed.template operator()<uint32_t>();
                case EIT_16BIT: return visitIndexed.template operator()<uint16_t>();
                default:
                    return false;
            }
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
