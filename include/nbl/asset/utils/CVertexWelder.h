// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_POLYGON_VERTEX_WELDER_H_INCLUDED_
#define _NBL_ASSET_C_POLYGON_VERTEX_WELDER_H_INCLUDED_

#include "nbl/asset/utils/CPolygonGeometryManipulator.h"

namespace nbl::asset {

class CVertexWelder {
	
public:
  using WeldPredicateFn = std::function<bool(const ICPUPolygonGeometry* geom, uint32_t idx1, uint32_t idx2)>;

  class DefaultWeldPredicate
  {
    private:
      static bool isAttributeValEqual(const ICPUPolygonGeometry::SDataView& view, uint32_t index1, uint32_t index2, float epsilon)
      {
        if (!view) return true;
        const auto channelCount = getFormatChannelCount(view.composed.format);
        switch (view.composed.rangeFormat)
        {
          case IGeometryBase::EAABBFormat::U64:
          case IGeometryBase::EAABBFormat::U32:
          {
            hlsl::uint64_t4 val1, val2;
            view.decodeElement<hlsl::uint64_t4>(index1, val1);
            view.decodeElement<hlsl::uint64_t4>(index2, val2);
            for (auto channel_i = 0u; channel_i < channelCount; channel_i++)
              if (val1[channel_i] != val2[channel_i]) return false;
            break;
          }
          case IGeometryBase::EAABBFormat::S64:
          case IGeometryBase::EAABBFormat::S32:
          {
            hlsl::int64_t4 val1, val2;
            view.decodeElement<hlsl::int64_t4>(index1, val1);
            view.decodeElement<hlsl::int64_t4>(index2, val2);
            for (auto channel_i = 0u; channel_i < channelCount; channel_i++)
              if (val1[channel_i] != val2[channel_i]) return false;
            break;
          }
          default:
          {
            hlsl::float64_t4 val1, val2;
            view.decodeElement<hlsl::float64_t4>(index1, val1);
            view.decodeElement<hlsl::float64_t4>(index2, val2);
            for (auto channel_i = 0u; channel_i < channelCount; channel_i++)
            {
              const auto diff = abs(val1[channel_i] - val2[channel_i]);
              if (diff > epsilon) return false;
            }
            break;
          }
        }
        return true;
      }

      static bool isAttributeDirEqual(const ICPUPolygonGeometry::SDataView& view, uint32_t index1, uint32_t index2, float epsilon)
      {
        if (!view) return true;
        const auto channelCount = getFormatChannelCount(view.composed.format);
        switch (view.composed.rangeFormat)
        {
          case IGeometryBase::EAABBFormat::U64:
          case IGeometryBase::EAABBFormat::U32:
          {
            hlsl::uint64_t4 val1, val2;
            view.decodeElement<hlsl::uint64_t4>(index1, val1);
            view.decodeElement<hlsl::uint64_t4>(index2, val2);
            return (1.0 - hlsl::dot(val1, val2)) < epsilon;
          }
          case IGeometryBase::EAABBFormat::S64:
          case IGeometryBase::EAABBFormat::S32:
          {
            hlsl::int64_t4 val1, val2;
            view.decodeElement<hlsl::int64_t4>(index1, val1);
            view.decodeElement<hlsl::int64_t4>(index2, val2);
            return (1.0 - hlsl::dot(val1, val2)) < epsilon;
          }
          default:
          {
            hlsl::float64_t4 val1, val2;
            view.decodeElement<hlsl::float64_t4>(index1, val1);
            view.decodeElement<hlsl::float64_t4>(index2, val2);
            return (1.0 - hlsl::dot(val1, val2)) < epsilon;
          }
        }
        return true;
      }

      float m_epsilon;

    public:

      DefaultWeldPredicate(float epsilon) : m_epsilon(epsilon) {}

      bool operator()(const ICPUPolygonGeometry* polygon, uint32_t index1, uint32_t index2)
      {
        if (!isAttributeValEqual(polygon->getPositionView(), index1, index2, m_epsilon))
          return false;
        if (!isAttributeDirEqual(polygon->getNormalView(), index1, index2, m_epsilon))
          return false;
        for (const auto& jointWeightView : polygon->getJointWeightViews())
        {
          if (!isAttributeValEqual(jointWeightView.indices, index1, index2, m_epsilon)) return false;
          if (!isAttributeValEqual(jointWeightView.weights, index1, index2, m_epsilon)) return false;
        }
        for (const auto& auxAttributeView : polygon->getAuxAttributeViews())
          if (!isAttributeValEqual(auxAttributeView, index1, index2, m_epsilon)) return false;

        return true;
      }
        
  };

  template <typename AccelStructureT>
  static core::smart_refctd_ptr<ICPUPolygonGeometry> weldVertices(const ICPUPolygonGeometry* polygon, const AccelStructureT& as, WeldPredicateFn shouldWeldFn) {
    auto outPolygon = core::move_and_static_cast<ICPUPolygonGeometry>(polygon->clone(0u));
    outPolygon->setIndexing(IPolygonGeometryBase::TriangleList());

    core::vector<uint32_t> vertexIndexToAsIndex(as.getVertexCount());

    for (uint32_t vertexData_i = 0u; vertexData_i < as.getVertexCount(); vertexData_i++)
    {
      const auto& vertexData = as.vertices()[vertexData_i];
      vertexIndexToAsIndex[vertexData.index] = vertexData.index;
    }

    static constexpr auto INVALID_INDEX = std::numeric_limits<uint32_t>::max();
    core::vector<uint32_t> remappedVertexIndexes(as.getVertexCount());
    std::fill(remappedVertexIndexes.begin(), remappedVertexIndexes.end(), INVALID_INDEX);

    uint32_t maxRemappedIndex = 0;
    // iterate by index, so that we always use the smallest index when multiple vertexes can be welded together
    for (uint32_t index = 0; index < as.getVertexCount(); index++)
    {
      const auto asIndex = vertexIndexToAsIndex[index];
      const auto& vertexData = as.vertices()[asIndex];
      auto& remappedVertexIndex = remappedVertexIndexes[index];
      as.forEachBroadphaseNeighborCandidates(vertexData, [&, polygon, index](const typename AccelStructureT::vertex_data_t& neighbor) {
        const auto neighborRemappedIndex = remappedVertexIndexes[neighbor.index];
        if (shouldWeldFn(polygon, index, neighbor.index) && neighborRemappedIndex != INVALID_INDEX) {
          remappedVertexIndex = neighborRemappedIndex;
          return false;
        }
        return true;
      });
      if (remappedVertexIndex != INVALID_INDEX) {
        remappedVertexIndex = vertexData.index;
        maxRemappedIndex = vertexData.index;
      }
    }

    const auto& indexView = outPolygon->getIndexView();
    if (indexView)
    {
      auto remappedIndexView = [&]
      {
        const auto bytesize = indexView.src.size;
        auto indices = ICPUBuffer::create({bytesize,IBuffer::EUF_INDEX_BUFFER_BIT});

        auto retval = indexView;
        retval.src.buffer = std::move(indices);
        if (retval.composed.rangeFormat == IGeometryBase::EAABBFormat::U16)
          retval.composed.encodedDataRange.u16.maxVx[0] = maxRemappedIndex;
        else if (retval.composed.rangeFormat == IGeometryBase::EAABBFormat::U32)
          retval.composed.encodedDataRange.u32.maxVx[0] = maxRemappedIndex;

        return retval;
      }();


      auto remappedIndexes = [&]<typename IndexT>() {
        auto* indexPtr = reinterpret_cast<IndexT*>(remappedIndexView.getPointer());
        for (uint64_t index_i = 0; index_i < polygon->getIndexCount(); index_i++)
        {
          hlsl::vector<IndexT, 1> index;
          indexView.decodeElement<hlsl::vector<IndexT, 1>>(index_i, index);
          IndexT remappedIndex = remappedVertexIndexes[index.x];
          indexPtr[index_i] = remappedIndex;
        }
      };

      if (indexView.composed.rangeFormat == IGeometryBase::EAABBFormat::U16) {
        remappedIndexes.template operator()<uint16_t>();
      }
      else if (indexView.composed.rangeFormat == IGeometryBase::EAABBFormat::U32) {
        remappedIndexes.template operator()<uint32_t>();
      }

      outPolygon->setIndexView(std::move(remappedIndexView));
    } else
    {
      const uint32_t indexSize = (outPolygon->getPositionView().getElementCount() - 1 < std::numeric_limits<uint16_t>::max()) ? sizeof(uint16_t) : sizeof(uint32_t);
      auto remappedIndexBuffer = ICPUBuffer::create({indexSize * outPolygon->getVertexReferenceCount(), IBuffer::EUF_INDEX_BUFFER_BIT});
      auto remappedIndexView = ICPUPolygonGeometry::SDataView{
        .composed = {
          .stride = indexSize,
        },
        .src = {
          .offset = 0,
          .size = remappedIndexBuffer->getSize(),
          .buffer = std::move(remappedIndexBuffer)
        }
      };

      auto fillRemappedIndex = [&]<typename IndexT>(){
        auto remappedIndexBufferPtr = reinterpret_cast<IndexT*>(remappedIndexBuffer->getPointer());
        for (uint64_t index = 0; index < outPolygon->getPositionView().getElementCount(); index++)
        {
          remappedIndexBufferPtr[index] = remappedVertexIndexes[index];
        }
      };

      if (indexView.composed.rangeFormat == IGeometryBase::EAABBFormat::U16) {
        fillRemappedIndex.template operator()<uint16_t>();
      }
      else if (indexView.composed.rangeFormat == IGeometryBase::EAABBFormat::U32) {
        fillRemappedIndex.template operator()<uint32_t>();
      }
      
      outPolygon->setIndexView(std::move(remappedIndexView));
      
    }

    CPolygonGeometryManipulator::recomputeContentHashes(outPolygon.get());
    return outPolygon;
  }
};

}

#endif
