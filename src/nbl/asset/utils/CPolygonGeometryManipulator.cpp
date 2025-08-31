// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/asset/asset.h"

#include <functional>
#include <algorithm>

#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
#include "nbl/asset/utils/CSmoothNormalGenerator.h"


namespace nbl::asset
{

core::smart_refctd_ptr<ICPUPolygonGeometry> CPolygonGeometryManipulator::createUnweldedList(const ICPUPolygonGeometry* inGeo)
{
    const auto* indexing = inGeo->getIndexingCallback();
    if (!indexing) return nullptr;
    if (indexing->degree() != 3) return nullptr;

    const auto indexView = inGeo->getIndexView();
    const auto primCount = inGeo->getPrimitiveCount();

    const auto outGeometry = core::move_and_static_cast<ICPUPolygonGeometry>(inGeo->clone(0u));

    auto* outGeo = outGeometry.get();
    outGeo->setIndexing(IPolygonGeometryBase::TriangleList());

    auto createOutView = [&](const ICPUPolygonGeometry::SDataView& inView) -> ICPUPolygonGeometry::SDataView
      {
        if (!inView) return {};
        auto buffer = ICPUBuffer::create({ inGeo->getPrimitiveCount() * indexing->degree() * inView.composed.stride , inView.src.buffer->getUsageFlags() });
        return {
          .composed = inView.composed,
          .src = {.offset = 0, .size = buffer->getSize(), .buffer = std::move(buffer)}
        };
      };

    const auto inIndexView = inGeo->getIndexView();
    auto outIndexView = createOutView(inIndexView);
    auto indexBuffer = outIndexView.src.buffer;
    const auto indexSize = inIndexView.composed.stride;
    std::byte* outIndexes = reinterpret_cast<std::byte*>(outIndexView.getPointer());
    outGeo->setIndexView({});

    const auto inVertexView = inGeo->getPositionView();
    auto outVertexView = createOutView(inVertexView);
    auto vertexBuffer = outVertexView.src.buffer;
    const auto vertexSize = inVertexView.composed.stride;
    const std::byte* inVertexes = reinterpret_cast<const std::byte*>(inVertexView.getPointer());
    std::byte* const outVertexes = reinterpret_cast<std::byte*>(vertexBuffer->getPointer());
    outGeo->setPositionView(std::move(outVertexView));

    const auto inNormalView = inGeo->getNormalView();
    const std::byte* const inNormals = reinterpret_cast<const std::byte*>(inNormalView.getPointer());
    auto outNormalView = createOutView(inNormalView);
    auto outNormalBuffer = outNormalView.src.buffer;
    outGeo->setNormalView(std::move(outNormalView));

    outGeometry->getJointWeightViews()->resize(inGeo->getJointWeightViews().size());
    for (uint64_t jointView_i = 0u; jointView_i < inGeo->getJointWeightViews().size(); jointView_i++)
    {
      auto& inJointWeightView = inGeo->getJointWeightViews()[jointView_i];
      auto& outJointWeightView = outGeometry->getJointWeightViews()->operator[](jointView_i);
      outJointWeightView.indices = createOutView(inJointWeightView.indices);
      outJointWeightView.weights = createOutView(inJointWeightView.weights);
    }

    outGeometry->getAuxAttributeViews()->resize(inGeo->getAuxAttributeViews().size());
    for (uint64_t auxView_i = 0u; auxView_i < inGeo->getAuxAttributeViews().size(); auxView_i++)
    {
      outGeo->getAuxAttributeViews()->operator[](auxView_i) = createOutView(inGeo->getAuxAttributeViews()[auxView_i]);
    }

    for (uint64_t prim_i = 0u; prim_i < primCount; prim_i++)
    {
      hlsl::uint32_t3 indexes;
      IPolygonGeometryBase::IIndexingCallback::SContext<uint32_t> context{
        .indexBuffer = indexView.getPointer(),
        .indexSize = indexView.composed.stride,
        .beginPrimitive = prim_i,
        .endPrimitive = prim_i + 1,
        .out = &indexes
      };
      indexing->operator()(context);
      for (uint64_t primIndex_i = 0u; primIndex_i < indexing->degree(); primIndex_i++)
      {
        const auto outIndex = prim_i * indexing->degree() + primIndex_i;
        const auto inIndex = indexes[primIndex_i];
        memcpy(outIndexes + outIndex * indexSize, &outIndex, indexSize);
        memcpy(outVertexes + outIndex * vertexSize, inVertexes + inIndex * vertexSize, vertexSize);
        if (inNormalView)
        {
          std::byte* const outNormals = reinterpret_cast<std::byte*>(outNormalBuffer->getPointer());
          const auto normalSize = inNormalView.composed.stride;
          memcpy(outNormals + outIndex * normalSize, inNormals + inIndex * normalSize, normalSize);
        }

        for (uint64_t jointView_i = 0u; jointView_i < inGeo->getJointWeightViews().size(); jointView_i++)
        {
          auto& inView = inGeo->getJointWeightViews()[jointView_i];
          auto& outView = outGeometry->getJointWeightViews()->operator[](jointView_i);

          const std::byte* const inJointIndices = reinterpret_cast<const std::byte*>(inView.indices.getPointer());
          const auto jointIndexSize = inView.indices.composed.stride;
          std::byte* const outJointIndices = reinterpret_cast<std::byte*>(outView.indices.getPointer());
          memcpy(outJointIndices + outIndex * jointIndexSize, inJointIndices + inIndex * jointIndexSize, jointIndexSize);

          const std::byte* const inWeights = reinterpret_cast<const std::byte*>(inView.weights.getPointer());
          const auto jointWeightSize = inView.weights.composed.stride;
          std::byte* const outWeights = reinterpret_cast<std::byte*>(outView.weights.getPointer());
          memcpy(outWeights + outIndex * jointWeightSize, outWeights + inIndex * jointWeightSize, jointWeightSize);
        }

        for (uint64_t auxView_i = 0u; auxView_i < inGeo->getAuxAttributeViews().size(); auxView_i++)
        {
          auto& inView = inGeo->getAuxAttributeViews()[auxView_i];
          auto& outView = outGeometry->getAuxAttributeViews()->operator[](auxView_i);
          const auto attrSize = inView.composed.stride;
          const std::byte* const inAuxs = reinterpret_cast<const std::byte*>(inView.getPointer());
          std::byte* const outAuxs = reinterpret_cast<std::byte*>(outView.getPointer());
          memcpy(outAuxs + outIndex * attrSize, inAuxs + inIndex * attrSize, attrSize);
        }
      }
    }

    recomputeContentHashes(outGeo);
    return outGeometry;
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CPolygonGeometryManipulator::createSmoothVertexNormal(const ICPUPolygonGeometry* inPolygon, float epsilon, VxCmpFunction vxcmp)
{
    if (inPolygon == nullptr)
    {
        _NBL_DEBUG_BREAK_IF(true);
        return nullptr;
    }

    core::smart_refctd_ptr<ICPUPolygonGeometry> outPolygon;
    outPolygon = core::move_and_static_cast<ICPUPolygonGeometry>(inPolygon->clone(0u));
    static constexpr auto Format = EF_R32G32B32_SFLOAT;
    const auto normalFormatBytesize = asset::getTexelOrBlockBytesize(Format);
    auto normalBuf = ICPUBuffer::create({ normalFormatBytesize * outPolygon->getPositionView().getElementCount()});
    auto normalView = inPolygon->getNormalView();

    hlsl::shapes::AABB<4,hlsl::float32_t> aabb;
    aabb.maxVx = hlsl::float32_t4(1, 1, 1, 0.f);
    aabb.minVx = -aabb.maxVx;
    outPolygon->setNormalView({
      .composed = {
        .encodedDataRange = {.f32 = aabb},
        .stride = sizeof(hlsl::float32_t3),
        .format = EF_R32G32B32_SFLOAT,
        .rangeFormat = IGeometryBase::EAABBFormat::F32
      },
      .src = { .offset = 0, .size = normalBuf->getSize(), .buffer = std::move(normalBuf) },
     });

    CSmoothNormalGenerator::calculateNormals(outPolygon.get(), epsilon, vxcmp);

    return outPolygon;
}

} // end namespace nbl::asset

