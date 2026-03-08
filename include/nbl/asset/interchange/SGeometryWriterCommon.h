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

#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <limits>
#include <system_error>
#include <type_traits>
namespace nbl::asset
{
class SGeometryWriterCommon
{
    public:
        struct SWriteState { hlsl::float32_t3x4 transform = hlsl::math::linalg::identity<hlsl::float32_t3x4>(); uint32_t instanceIx = ~0u; uint32_t targetIx = ~0u; uint32_t geometryIx = 0u; };
        struct SPolygonGeometryWriteItem : SWriteState { const ICPUPolygonGeometry* geometry = nullptr; };

        template<typename Container = core::vector<SPolygonGeometryWriteItem>> requires requires(Container& c, const SPolygonGeometryWriteItem& item) { c.emplace_back(item); }
        static inline Container collectPolygonGeometryWriteItems(const IAsset* rootAsset)
        {
            Container out = {};
            if (!rootAsset)
                return out;

            const auto identity = hlsl::math::linalg::identity<hlsl::float32_t3x4>();
            auto appendFromCollection = [&](const ICPUGeometryCollection* collection, const hlsl::float32_t3x4& transform, const uint32_t instanceIx, const uint32_t targetIx) -> void {
                if (!collection)
                    return;
                const auto& geometries = collection->getGeometries();
                for (uint32_t geometryIx = 0u; geometryIx < geometries.size(); ++geometryIx)
                {
                    const auto& ref = geometries[geometryIx];
                    if (!ref.geometry || ref.geometry->getPrimitiveType() != IGeometryBase::EPrimitiveType::Polygon)
                        continue;
                    SPolygonGeometryWriteItem item = {};
                    item.geometry = static_cast<const ICPUPolygonGeometry*>(ref.geometry.get());
                    item.transform = hlsl::math::linalg::promoted_mul(transform, ref.hasTransform() ? ref.transform : identity);
                    item.instanceIx = instanceIx;
                    item.targetIx = targetIx;
                    item.geometryIx = geometryIx;
                    out.emplace_back(item);
                }
            };
            if (rootAsset->getAssetType() == IAsset::ET_GEOMETRY)
            {
                const auto* geometry = static_cast<const IGeometry<ICPUBuffer>*>(rootAsset);
                if (geometry->getPrimitiveType() == IGeometryBase::EPrimitiveType::Polygon)
                {
                    SPolygonGeometryWriteItem item = {};
                    item.geometry = static_cast<const ICPUPolygonGeometry*>(rootAsset);
                    out.emplace_back(item);
                }
                return out;
            }

            if (rootAsset->getAssetType() == IAsset::ET_GEOMETRY_COLLECTION)
            {
                appendFromCollection(static_cast<const ICPUGeometryCollection*>(rootAsset), identity, ~0u, ~0u);
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
                    appendFromCollection(targetList[targetIx].geoCollection.get(), instanceTransform, instanceIx, targetIx);
            }

            return out;
        }

        static inline bool isIdentityTransform(const hlsl::float32_t3x4& transform) { return transform == hlsl::math::linalg::identity<hlsl::float32_t3x4>(); }

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
        static inline const T* getTightView(const ICPUPolygonGeometry::SDataView& view) { return view && view.composed.format == ExpectedFormat && view.composed.getStride() == sizeof(T) ? reinterpret_cast<const T*>(view.getPointer()) : nullptr; }

        static inline char* appendFloatToBuffer(char* dst, char* end, float value) { return appendFloatingPointToBuffer(dst, end, value); }
        static inline char* appendFloatToBuffer(char* dst, char* end, double value) { return appendFloatingPointToBuffer(dst, end, value); }

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

    private:
        template<typename T>
        static inline char* appendFloatingPointToBuffer(char* dst, char* const end, const T value)
        {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

            if (!dst || dst >= end)
                return end;

            const auto result = std::to_chars(dst, end, value);
            if (result.ec == std::errc())
                return result.ptr;

            constexpr size_t FloatingPointScratchSize = std::numeric_limits<T>::max_digits10 + 9ull;
            std::array<char, FloatingPointScratchSize> scratch = {};
            constexpr int Precision = std::numeric_limits<T>::max_digits10;
            const int written = std::snprintf(scratch.data(), scratch.size(), "%.*g", Precision, static_cast<double>(value));
            if (written <= 0)
                return dst;

            const size_t writeLen = static_cast<size_t>(written);
            if (writeLen > static_cast<size_t>(end - dst))
                return end;

            std::memcpy(dst, scratch.data(), writeLen);
            return dst + writeLen;
        }
};
}
#endif
