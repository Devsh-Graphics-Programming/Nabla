// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_POLYGON_VERTEX_WELDER_H_INCLUDED_
#define _NBL_ASSET_C_POLYGON_VERTEX_WELDER_H_INCLUDED_

#include "nbl/asset/utils/CPolygonGeometryManipulator.h"

namespace nbl::asset {

template <typename T>
concept VertexWelderAccelerationStructure = requires(T const cobj, hlsl::float32_t3 position, std::function<bool(const typename T::vertex_data_t&)> fn)
{
  typename T::vertex_data_t;
  { std::same_as<decltype(T::vertex_data_t::index), uint32_t> };
  { cobj.forEachBroadphaseNeighborCandidates(position, fn) } -> std::same_as<void>;
};

class CVertexWelder {
    
  public:

    class WeldPredicate
    {
      public:
        virtual bool init(const ICPUPolygonGeometry* geom) = 0;
        virtual bool operator()(const ICPUPolygonGeometry* geom, uint32_t idx1, uint32_t idx2) const = 0;
        virtual ~WeldPredicate() = default;
    };

    class DefaultWeldPredicate : public WeldPredicate
    {
      private:

        static inline bool isIntegralElementEqual(const ICPUPolygonGeometry::SDataView& view, uint32_t index1, uint32_t index2)
        {
          const auto byteSize = getTexelOrBlockBytesize(view.composed.format);
          const auto* basePtr = reinterpret_cast<const std::byte*>(view.getPointer());
          const auto stride = view.composed.stride;
          return (memcmp(basePtr + (index1 * stride), basePtr + (index2 * stride), byteSize) == 0);
        }

        static inline bool isRealElementEqual(const ICPUPolygonGeometry::SDataView& view, uint32_t index1, uint32_t index2, uint32_t channelCount, float epsilon)
        {
          // TODO: Handle 16,32,64 bit float vectors once the pixel encode/decode functions get reimplemented in HLSL and decodeElement can actually benefit from that.
          hlsl::float64_t4 val1, val2;
          view.decodeElement<hlsl::float64_t4>(index1, val1);
          view.decodeElement<hlsl::float64_t4>(index2, val2);
          for (auto channel_i = 0u; channel_i < channelCount; channel_i++)
          {
            const auto diff = abs(val1[channel_i] - val2[channel_i]);
            if (diff > epsilon) return false;
          }
          return true;
        }

        static inline bool isAttributeValEqual(const ICPUPolygonGeometry::SDataView& view, uint32_t index1, uint32_t index2, float epsilon)
        {
          if (!view) return true;
          const auto channelCount = getFormatChannelCount(view.composed.format);
          // TODO: use memcmp to compare for integral equality
          const auto byteSize = getTexelOrBlockBytesize(view.composed.format);
          switch (view.composed.rangeFormat)
          {
            case IGeometryBase::EAABBFormat::U64:
            case IGeometryBase::EAABBFormat::U32:
            case IGeometryBase::EAABBFormat::S64:
            case IGeometryBase::EAABBFormat::S32:
            {
              return isIntegralElementEqual(view, index1, index2);
            }
            default:
            {
              return isRealElementEqual(view, index1, index2, channelCount, epsilon);
            }
          }
          return true;
        }

        static inline bool isAttributeDirEqual(const ICPUPolygonGeometry::SDataView& view, uint32_t index1, uint32_t index2, float epsilon)
        {
          if (!view) return true;
          const auto channelCount = getFormatChannelCount(view.composed.format);
          switch (view.composed.rangeFormat)
          {
            case IGeometryBase::EAABBFormat::U64:
            case IGeometryBase::EAABBFormat::U32:
            case IGeometryBase::EAABBFormat::S64:
            case IGeometryBase::EAABBFormat::S32:
            {
              return isIntegralElementEqual(view, index1, index2);
            }
            default:
            {
              if (channelCount != 3)
                return isRealElementEqual(view, index1, index2, channelCount, epsilon);

              hlsl::float64_t4 val1, val2;
              view.decodeElement<hlsl::float64_t4>(index1, val1);
              view.decodeElement<hlsl::float64_t4>(index2, val2);
              return (1.0 - hlsl::dot(val1, val2)) < epsilon;
            }
          }
        }

        float m_epsilon;

      public:

        DefaultWeldPredicate(float epsilon) : m_epsilon(epsilon) {}

        inline bool init(const ICPUPolygonGeometry* polygon) override
        {
          return polygon->valid();
        }

        inline bool operator()(const ICPUPolygonGeometry* polygon, uint32_t index1, uint32_t index2) const override
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

        ~DefaultWeldPredicate() override = default;
          
    };

    template <VertexWelderAccelerationStructure AccelStructureT>
    static inline core::smart_refctd_ptr<ICPUPolygonGeometry> weldVertices(const ICPUPolygonGeometry* polygon, const AccelStructureT& as, const WeldPredicate& shouldWeldFn) {
      auto outPolygon = core::move_and_static_cast<ICPUPolygonGeometry>(polygon->clone(0u));

      const auto& positionView = polygon->getPositionView();
      const auto vertexCount = positionView.getElementCount();

      constexpr auto INVALID_INDEX = std::numeric_limits<uint32_t>::max();
      core::vector<uint32_t> remappedVertexIndexes(vertexCount);
      std::fill(remappedVertexIndexes.begin(), remappedVertexIndexes.end(), INVALID_INDEX);

      uint32_t maxRemappedIndex = 0;
      // iterate by index, so that we always use the smallest index when multiple vertexes can be welded together
      for (uint32_t index = 0; index < vertexCount; index++)
      {
        hlsl::float32_t3 position;
        positionView.decodeElement<hlsl::float32_t3>(index, position);
        auto remappedVertexIndex = INVALID_INDEX;
        bool foundVertex = false;
        as.forEachBroadphaseNeighborCandidates(position, [&](const typename AccelStructureT::vertex_data_t& candidate) {
          const auto neighborRemappedIndex = remappedVertexIndexes[candidate.index];
          if (index == candidate.index) {
            foundVertex = true;
          }
          else if (neighborRemappedIndex != INVALID_INDEX && shouldWeldFn(polygon, index, candidate.index)) {
            remappedVertexIndex = neighborRemappedIndex;
          }
          return !(foundVertex && remappedVertexIndex != INVALID_INDEX);
        });
        if (foundVertex)
        {
          if (remappedVertexIndex == INVALID_INDEX) {
            remappedVertexIndex = index;
            maxRemappedIndex = index;
          }
        }
        remappedVertexIndexes[index] = remappedVertexIndex;
      }

      const auto& indexView = outPolygon->getIndexView();
      const auto remappedRangeFormat = (maxRemappedIndex - 1) < std::numeric_limits<uint16_t>::max() ? IGeometryBase::EAABBFormat::U16 : IGeometryBase::EAABBFormat::U32;

      auto createRemappedIndexView = [&](size_t indexCount) {
        const uint32_t indexSize = remappedRangeFormat == IGeometryBase::EAABBFormat::U16 ? sizeof(uint16_t) : sizeof(uint32_t);
        auto remappedIndexBuffer = ICPUBuffer::create({indexSize * indexCount, IBuffer::EUF_INDEX_BUFFER_BIT});
        auto remappedIndexView = ICPUPolygonGeometry::SDataView{
          .composed = {
            .stride = indexSize,
            .rangeFormat = remappedRangeFormat
          },
          .src = {
            .offset = 0,
            .size = remappedIndexBuffer->getSize(),
            .buffer = std::move(remappedIndexBuffer)
          }
        };

        if (remappedRangeFormat == IGeometryBase::EAABBFormat::U16)
        {
          hlsl::shapes::AABB<4, uint16_t> aabb;
          aabb.minVx[0] = 0;
          aabb.maxVx[0] = maxRemappedIndex;
          remappedIndexView.composed.encodedDataRange.u16 = aabb;
          remappedIndexView.composed.format = EF_R16_UINT;
        }
        else if (remappedRangeFormat == IGeometryBase::EAABBFormat::U32) {
          hlsl::shapes::AABB<4, uint32_t> aabb;
          aabb.minVx[0] = 0;
          aabb.maxVx[0] = maxRemappedIndex;
          remappedIndexView.composed.encodedDataRange.u32 = aabb;
          remappedIndexView.composed.format = EF_R32_UINT;
        }

        return remappedIndexView;
      };


      if (indexView)
      {
        auto remappedIndexView = createRemappedIndexView(polygon->getIndexCount());
        auto remappedIndexes = [&]<typename IndexT>() -> bool {
          auto* remappedIndexPtr = reinterpret_cast<IndexT*>(remappedIndexView.getPointer());
          for (uint32_t index_i = 0; index_i < polygon->getIndexCount(); index_i++)
          {
            hlsl::vector<IndexT, 1> index;
            indexView.decodeElement<hlsl::vector<IndexT, 1>>(index_i, index);
            IndexT remappedIndex = remappedVertexIndexes[index.x];
            remappedIndexPtr[index_i] = remappedIndex;
            if (remappedIndex == INVALID_INDEX) return false;
          }
          return true;
        };

        if (remappedRangeFormat == IGeometryBase::EAABBFormat::U16) {
          if (!remappedIndexes.template operator()<uint16_t>()) return nullptr;
        }
        else if (remappedRangeFormat == IGeometryBase::EAABBFormat::U32) {
          if (!remappedIndexes.template operator()<uint32_t>()) return nullptr;
        }

        outPolygon->setIndexView(std::move(remappedIndexView));

      } else
      {
        auto remappedIndexView = createRemappedIndexView(remappedVertexIndexes.size());

        auto fillRemappedIndex = [&]<typename IndexT>(){
          auto remappedIndexBufferPtr = reinterpret_cast<IndexT*>(remappedIndexView.getPointer());
          for (uint32_t index_i = 0; index_i < remappedVertexIndexes.size(); index_i++)
          {
            if (remappedVertexIndexes[index_i] == INVALID_INDEX) return false;
            remappedIndexBufferPtr[index_i] = remappedVertexIndexes[index_i];
          }
          return true;
        };
        if (remappedRangeFormat == IGeometryBase::EAABBFormat::U16) {
          if (!fillRemappedIndex.template operator()<uint16_t>()) return nullptr;
        }
        else if (remappedRangeFormat == IGeometryBase::EAABBFormat::U32) {
          if (!fillRemappedIndex.template operator()<uint32_t>()) return nullptr;
        }

        outPolygon->setIndexView(std::move(remappedIndexView));
      }

      CPolygonGeometryManipulator::recomputeContentHashes(outPolygon.get());
      return outPolygon;
    }
};

}

#endif
