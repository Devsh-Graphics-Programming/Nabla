// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_KDOP_OBB_H_INCLUDED__
#define __NBL_KDOP_OBB_H_INCLUDED__

#include "matrix3x4SIMD.h"

namespace nbl
{
  namespace core
  {
    /**
     * @struct OBB (Oriented Bounding Box)
     */
    struct OBB final
    {
      matrix3x4SIMD asMat3x4; // equivalent of glsl's mat4x3

      float bestVal = 0;  // the best obb quality value
      matrix3x4SIMD b;    // The currently best found OBB orientation (transposed)
      vectorSIMDf bMin;
      vectorSIMDf bMax;
      vectorSIMDf dims; // (bLen)
      vectorSIMDf extents; // half-extents

      /**
       *
       * @param[in] scale
       * @param[in] rotation
       * @param[in] mid
       *
       * TODO: Store world-space coordinate of the local OBB origin
       * Construct from offset (local origin) and full extent, then OBB can have the form:
       *    column0 = localXaxis
       *    column1 = localYaxis
       *    column2 = localZaxis
       *    column3 = offset
       * no concatenation of matrices necessary
       */
      inline void init(
        const vectorSIMDf&    scale     = vectorSIMDf(),
        const matrix3x4SIMD&  rotation  = matrix3x4SIMD(),
        const vectorSIMDf&    mid       = vectorSIMDf()
      )
      {
        matrix3x4SIMD scaleMat;
        scaleMat.setScale(scale);

        asMat3x4 = concatenateBFollowedByA(rotation, scaleMat);
        asMat3x4.setTranslation(mid);
      }
    };

    /**
     * @class LBT (Large Base Triangle)
     *
     * @tparam projCount (k) Number of sample directions (projections)
     */
    template<uint8_t projCount>
    class LBT final
    {
      // if any DiTO-k algo needs LBT, should be friended here for explicit access
      template<uint8_t, uint8_t>
      friend class KDOP;

      using VectorSIMDfHalfArr    = std::array<vectorSIMDf, projCount>;
      using VectorSIMDfFullArr    = std::array<vectorSIMDf, projCount * 2>;
      using VectorSIMDfArrayPair  = std::pair<VectorSIMDfHalfArr, VectorSIMDfHalfArr>;

      struct Data final
      {
        vectorSIMDf p0, p1, p2; // Vertices of the large base triangle
        vectorSIMDf e0, e1, e2; // Edge vectors of the large base triangle
        vectorSIMDf n;          // Unit normal of the large base triangle
      };

      enum class DegenCase
      {
        NONE,
        EDGE, // max|p0 - p1| < threshold (eps)
        TRI,  // max|p0 - e0| < threshold (eps)
        TETRA //
      };

      private:
        inline Data getData() const noexcept { return m_data; }
        inline DegenCase getDegenerateCase() const noexcept { return m_degenerate; }

        /**
         * @brief calculate first 2 vertices
         *
         * @param[in] extremalPoints
         */
        inline void calcFarthestPointPair(
          const VectorSIMDfArrayPair& extremalPoints
        ) noexcept
        {
          auto& [minVtxPos, maxVtxPos] = extremalPoints;

//          auto distanceSquared = [=](const vectorSIMDf& a, const vectorSIMDf& b)
//          {
//            vectorSIMDf diff(b.x - a.x, b.y - a.y, b.z - a.z); // sub
//
//            return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z; // sqLength/dot
//          };

          auto maxDistance = distancesquared(minVtxPos[0], maxVtxPos[0]).x;
          uint32_t farthestPairIdx = 0u;

          for(uint32_t k = 1u; k < projCount; k++)
          {
            auto distance = distancesquared(minVtxPos[k], maxVtxPos[k]).x;

            if(distance > maxDistance) maxDistance = distance; farthestPairIdx = k;
          }

          m_data.p0 = minVtxPos[farthestPairIdx];
          m_data.p1 = maxVtxPos[farthestPairIdx];

          // Detrimental Case 1 Check: Degenerate Edge
          // If the found furthest points are located very close, return OBB aligned with the initial AABB
          if(distancesquared(m_data.p0, m_data.p1).x < s_threshold) m_degenerate = DegenCase::EDGE;
        }

        /**
         *
         * @param[in] selVtxArr
         */
        inline void calcThirdPoint(const VectorSIMDfFullArr& selVtxArr, uint8_t pointCount) noexcept
        {
          // TODO: Need to be consistent, if it's vector from p0 then the directions all need to be -p0 for oriented lines
          m_data.e0 = normalize(m_data.p0 - m_data.p1);

          // Find a third vertex the furthest away from line given by p0, e0
          auto maxDistance = calcPointToLineDistance(m_data.p0, m_data.e0, selVtxArr[0]);
          m_data.p2 = selVtxArr[0];

          for(uint32_t i = 1u; i < pointCount; i++)
          {
            float distance = calcPointToLineDistance(m_data.p0, m_data.e0, selVtxArr[i]);

            if(distance > maxDistance)
            {
              m_data.p2 = selVtxArr[i];
              maxDistance = distance;
            }
          }

          // TODO: construct OBB with local X axis along the p1-p0 line and any random rotation (fixed?)
          // Detrimental Case 2 Check: Degenerate Triangle
          // If the third point is located very close to the line, return an OBB aligned with the line
          if(maxDistance < s_threshold) m_degenerate = DegenCase::TRI;
        }

        /**
         * @brief calculate edges of the base triangle
         */
        inline void calcEdges() noexcept
        {
          m_data.e1 = normalize(m_data.p1 - m_data.p2);
          m_data.e2 = normalize(m_data.p2 - m_data.p0);
        }

        /**
         * @brief calculate normal of the base triangle
         */
        inline void calcNormal() noexcept
        {
          m_data.n  = normalize(cross(m_data.e1, m_data.e0));
        }

        /**
         *
         * @param[in] p
         * @param[in] dir
         * @param[in] v
         *
         * @return point to line distance
         */
        inline static float calcPointToLineDistance(
          const vectorSIMDf& p,
          const vectorSIMDf& dir,
          const vectorSIMDf& v
        ) noexcept
        {
          _NBL_DEBUG_BREAK_IF(length(dir).x < s_threshold);

          return length(cross(v - p, dir)).x;
        }

      private:
        static constexpr float s_threshold = 0.000001f; // epsilon (eps)

        Data m_data;
        DegenCase m_degenerate;
    };

    /**
     * @class KDOP (k-dimensional Discrete Oriented Polytope)
     *
     * @tparam projCount        (k)  Number of sample directions (projections)
     * @tparam maxPointCount    (np) Maximum Number of points selected along the sample directions
     *
     * @brief DiTO-14 and DiTO-26
     */
    template<uint8_t projCount = 13, uint8_t maxPointCount = projCount * 2>
    class KDOP final
    {
      struct Slab;

      using LBT                   = LBT<projCount>;
      using VectorSIMDfHalfArr    = typename LBT::VectorSIMDfHalfArr;
      using VectorSIMDfFullArr    = typename LBT::VectorSIMDfFullArr;
      using VectorSIMDfArrayPair  = typename LBT::VectorSIMDfArrayPair;
      using VtxPosGetCallback     = std::function<vectorSIMDf(size_t)>;
      using SlabArr               = std::array<Slab, projCount>;

      enum class OBBType
      {
        DEFAULT,
        AA, // Axis Aligned
        LA // Line Aligned
      };

      struct Slab final
      {
        float min = FLT_MAX;
        float max = -FLT_MAX;
      };

      struct AABB final
      {
        vectorSIMDf dims;     // axis aligned dimensions of the vertices (alLen)
        vectorSIMDf midPoint; // axis aligned mid point of the vertices (alMid)
        float area = 0;       // quality measure of the axis-aligned box (alVal)
      };

      /**
       * @brief constructs efficient normal sets for DiTO-14 and DiTO-26 with k-dop computation
       */
      struct EtaNormal final
      {
        VectorSIMDfHalfArr sets;

        EtaNormal()
        {
          sets = {
            vectorSIMDf(1.0f, 0.0f, 0.0f),
            vectorSIMDf(0.0f, 1.0f, 0.0f),
            vectorSIMDf(0.0f, 0.0f, 1.0f),
            vectorSIMDf(1.0f, 1.0f, 1.0f),
            vectorSIMDf(1.0f, 1.0f, -1.0f),
            vectorSIMDf(1.0f, -1.0f, 1.0f),
            vectorSIMDf(1.0f, -1.0f, -1.0f)
          };

          if(projCount == 13)
          {
            sets[7]   = vectorSIMDf(1.0f, 1.0f, 0.0f);
            sets[8]   = vectorSIMDf(1.0f, -1.0f, 0.0f);
            sets[9]   = vectorSIMDf(1.0f, 0.0f, 1.0f);
            sets[10]  = vectorSIMDf(1.0f, 0.0f, -1.0f);
            sets[11]  = vectorSIMDf(0.0f, 1.0f, 1.0f);
            sets[12]  = vectorSIMDf(0.0f, 1.0f, -1.0f);
          }
        }
      };

      public:
        /**
         *
         * @param[in] selVtxArray fixed array of pre-selected vertices (capped to maxPointCount)
         * @param[in] totalVtxCount total number of vertices
         */
        KDOP(const VectorSIMDfFullArr& selVtxArray, size_t totalVtxCount)
        : m_selVtxArray   (selVtxArray)
        , m_totalVtxCount (totalVtxCount)
        , m_baseVtxPos    (m_selVtxArray[0]) {}

        /**
         * @note this ctor may be slightly slower due to lambda's heap allocation but more convenient
         *
         * TODO: figure if can be further optimized
         *
         * @param[in] vtxPosGetCallback lambda expression returning vertex position by specified index
         * @param[in] totalVtxCount total number of vertices
         */
        KDOP(const VtxPosGetCallback& vtxPosGetCallback, size_t totalVtxCount)
        : m_totalVtxCount (totalVtxCount)
        , m_selVtxArray   (selectVtxArray(vtxPosGetCallback))
        , m_baseVtxPos    (m_selVtxArray[0]) {}

      public:
        /**
         *
         * @param[out] obb
         */
        inline void compute(OBB& obb) noexcept
        {
          if(m_totalVtxCount <= 0)
          {
            finalizeOBB(obb, OBBType::AA);
            return;
          }

          VectorSIMDfArrayPair extremalPoints;
          SlabArr slabs;

          findExtremalValues(extremalPoints, slabs);
          calcAABBOrientationForStdBase(slabs, obb.bestVal);

          // if vtxCount > pointCount then array is capped to maxPointCount (selected extremal points)
//          m_pointCount = m_vtxCount > m_pointCount ? m_pointCount : m_vtxCount;

          LBT lbt;
          lbt.calcFarthestPointPair(extremalPoints);

          if(lbt.getDegenerateCase() == LBT::DegenCase::EDGE) { finalizeOBB(obb, OBBType::AA); return; }

          lbt.calcThirdPoint(m_selVtxArray, maxPointCount);

          if(lbt.getDegenerateCase() == LBT::DegenCase::TRI) { finalizeOBB(obb, OBBType::LA); return; }

          lbt.calcEdges();
          lbt.calcNormal();

          calcOBBAxes(lbt.getData(), extremalPoints, obb.b, obb.bestVal);
          calcOBBDimensions(obb.b, obb.bMin, obb.bMax, obb.dims);

          finalizeOBB(obb);
        }

      private:
        inline bool isIdxInvalid(size_t idx) const noexcept { return idx >= maxPointCount; }

        /**
         *
         * @param vtxPosGetCallback
         * @return fixed array of pre-selected vertices (capped to maxPointCount)
         */
        inline VectorSIMDfFullArr selectVtxArray(
          const VtxPosGetCallback& vtxPosGetCallback
        ) noexcept
        {
          VectorSIMDfFullArr vtxArray;

          for(auto i = 0u; i < m_totalVtxCount; i++)
          {
            if(isIdxInvalid(i)) break; // cap to max

            vtxArray[i] = vtxPosGetCallback(i);
          }

          return vtxArray;
        }

        /**
         *
         * @param[in] vtxPos
         * @param[in] normalSet
         * @param[in] idx
         * @return normalized projection/slab (float)
         */
        inline float normalizeSlab(
          const vectorSIMDf& vtxPos,
          const vectorSIMDf& normalSet,
          uint8_t idx
        ) const noexcept
        {
          const auto& slab = dot(vtxPos, normalSet);
          return idx < 3
              ? (idx == 0 ? slab.x : (idx == 1 ? slab.y : slab.z)) // slabs 0, 1 and 2 define AABB
              : slab.x + slab.y + slab.z;
        }

         /**
          * @brief calculate/initialize max & min points and projections of the meshBuffer
          *
          * @param[out] extremalPoints
          * @param[out] slabs
          */
        inline void findExtremalValues(
          VectorSIMDfArrayPair& extremalPoints,
          SlabArr& slabs
        ) noexcept
        {
          EtaNormal normal;
          auto& [minVtxPos, maxVtxPos] = extremalPoints;

          for(auto k = 0u; k < projCount; k++)
          {
            slabs[k].min = slabs[k].max = normalizeSlab(m_baseVtxPos, normal.sets[k], k);
          }

          // TODO: implement a SIMD memset to zero out faster
          std::fill(std::begin(minVtxPos), std::end(minVtxPos), m_baseVtxPos);
          std::fill(std::begin(maxVtxPos), std::end(maxVtxPos), m_baseVtxPos);

          for(auto i = 1u; i < m_totalVtxCount; i++)
          {
            if(isIdxInvalid(i)) break; // cap to max

            const auto& vtxPos = m_selVtxArray[i];

            for(auto j = 0u; j < projCount; j++)
            {
              auto& minSlab = slabs[j].min;
              auto& maxSlab = slabs[j].max;
              auto vtxProj = normalizeSlab(vtxPos, normal.sets[j], j);

              if(vtxProj < minSlab) { minSlab = vtxProj; minVtxPos[j] = vtxPos; }
              if(vtxProj > maxSlab) { maxSlab = vtxProj; maxVtxPos[j] = vtxPos; }
            }
          }
        }

        /**
         * @brief calculate size of AABB (slabs 0, 1 and 2 define AABB)
         * and initialize the best found orientation so far to be the standard base
         *
         * @param[in] slabs
         * @param[out] bestVal
         */
        inline void calcAABBOrientationForStdBase(const SlabArr& slabs, float& bestVal) noexcept
        {
          // TODO: keep track of origin instead (OBB forms an orthogonal local basis with scale)
          m_aabb.midPoint = vectorSIMDf(
            slabs[0].min + slabs[0].max,
            slabs[1].min + slabs[1].max,
            slabs[2].min + slabs[2].max
          ) * 0.5f;

          m_aabb.dims = vectorSIMDf(
            slabs[0].max - slabs[0].min,
            slabs[1].max - slabs[1].min,
            slabs[2].max - slabs[2].min
          );

          m_aabb.area = m_aabb.dims.x * m_aabb.dims.y +
                        m_aabb.dims.x * m_aabb.dims.z +
                        m_aabb.dims.y * m_aabb.dims.z; // half box area

          bestVal = m_aabb.area;
        }

        /**
         * @brief calculate remaining points (vertices) of ditetrahedron
         *
         * @param[in] p
         * @param[in] n
         * @param[in] extremalPoints
         * @param[out] ditetraPoints
         */
        inline void findDitetraPoints(
          const vectorSIMDf& p,
          const vectorSIMDf& n,
          const VectorSIMDfArrayPair& extremalPoints,
          std::pair<vectorSIMDf, vectorSIMDf>& ditetraPoints
        ) noexcept
        {
          // find vertices that are furthest from the plane defined by p0, p1 and p2 (base triangle)
          // on the both positive and negative half space
          // n dot x = d
          const float& d = dot(p, n).x;
          const auto& [minVtxPos, maxVtxPos] = extremalPoints;
          auto& [t0, t1] = ditetraPoints;
          float minDistance;
          float maxDistance;

          minDistance = maxDistance = dot(m_baseVtxPos, n).x - d;
          t0 = t1 = m_baseVtxPos;

          for(size_t i = 0; i < projCount; i++)
          {
            const auto& maxVtxPosition = minVtxPos[i];
            const auto& minVtxPosition = maxVtxPos[i];
            float maxDist = dot(maxVtxPosition, n).x - d;
            float minDist = dot(minVtxPosition, n).x - d;

            if(maxDist > maxDistance) { maxDistance = maxDist; t1 = maxVtxPosition; }
            if(minDist < minDistance) { minDistance = minDist; t0 = minVtxPosition; }
          }

          // Detrimental Case 3 Check: Degenerate Tetrahedron
          if(distancesquared(t0, t1).x < LBT::s_threshold)
          {

          }
        }

        /**
         *
         * @param[in] dir
         * @param[out] minSlab
         * @param[out] maxSlab
         */
        inline void calcExtremalProjections(const vectorSIMDf& dir, float& minSlab, float& maxSlab) noexcept
        {
          minSlab = maxSlab = dot(m_baseVtxPos, dir).x;

          for(size_t i = 1u; i < m_totalVtxCount; i++)
          {
            if(isIdxInvalid(i)) break; // cap to max

            float proj = dot(m_selVtxArray[i], dir).x;

            if(proj > maxSlab) maxSlab = proj;
            if(proj < minSlab) minSlab = proj;
          }
        }

        /**
         *
         * @param[in] v0
         * @param[in] v1
         * @param[in] v2
         * @param[in] n
         * @param[out] b
         * @param[out] bestVal
         */
        inline void calcImprovedAxes(
          const vectorSIMDf& v0, const vectorSIMDf& v1, const vectorSIMDf& v2, const vectorSIMDf& n,
          matrix3x4SIMD& b,
          float& bestVal
        ) noexcept
        {
          vectorSIMDf dMin, dMax, len;

          vectorSIMDf m0 = cross(v0, n);
          vectorSIMDf m1 = cross(v1, n);
          vectorSIMDf m2 = cross(v2, n);

          calcExtremalProjections(v0, dMin.x, dMax.x);
          calcExtremalProjections(n, dMin.y, dMax.y);
          calcExtremalProjections(m0, dMin.z, dMax.z);

          len = dMax - dMin;
          float quality = len.x * len.y + len.x * len.z + len.y * len.z;

          if(quality < bestVal)
          {
            bestVal = quality;
            b = matrix3x4SIMD(v0, n, m0);
          }

          calcExtremalProjections(v1, dMin.x, dMax.x);
          calcExtremalProjections(m1, dMin.z, dMax.z);

          len = dMax - dMin;
          quality = len.x * len.y + len.x * len.z + len.y * len.z;

          if(quality < bestVal)
          {
            bestVal = quality;
            b = matrix3x4SIMD(v1, n, m1);
          }

          calcExtremalProjections(v2, dMin.x, dMax.x);
          calcExtremalProjections(m2, dMin.z, dMax.z);

          len = dMax - dMin;
          quality = len.x * len.y + len.x * len.z + len.y * len.z;

          if(quality < bestVal)
          {
            bestVal = quality;
            b = matrix3x4SIMD(v2, n, m2);
          }
        }

        /**
         *
         * @param[in] baseTri
         * @param[in] extremalPoints
         * @param[out] b
         * @param[out] bestVal
         */
        inline void calcOBBAxes(
          const typename LBT::Data& baseTri,
          const VectorSIMDfArrayPair& extremalPoints,
          matrix3x4SIMD &b,
          float &bestVal
        ) noexcept
        {
          auto& p0 = baseTri.p0; auto& p1 = baseTri.p1; auto& p2 = baseTri.p2;
          auto& e0 = baseTri.e0; auto& e1 = baseTri.e1; auto& e2 = baseTri.e2;
          auto& n  = baseTri.n;

          std::pair<vectorSIMDf, vectorSIMDf> ditetraPoints; // two connected tetrahedron's vertices
          findDitetraPoints(p0, n, extremalPoints, ditetraPoints);

          auto& [t0, t1] = ditetraPoints;

          // from base triangle
          calcImprovedAxes(e0, e1, e2, n, b, bestVal);

          // from top tetrahedra
          vectorSIMDf f0 = normalize(t0 - p0);
          vectorSIMDf f1 = normalize(t0 - p1);
          vectorSIMDf f2 = normalize(t0 - p2);
          vectorSIMDf n0 = normalize(cross(f1, e0));
          vectorSIMDf n1 = normalize(cross(f2, e1));
          vectorSIMDf n2 = normalize(cross(f0, e2));

//          calcImprovedAxes(e0, f1, f0, n0, b, bestVal);
//          calcImprovedAxes(e1, f2, f1, n1, b, bestVal);
//          calcImprovedAxes(e2, f0, f2, n2, b, bestVal);

          // from bottom tetrahedra
          f0 = normalize(t1 - p0);
          f1 = normalize(t1 - p1);
          f2 = normalize(t1 - p2);
          n0 = normalize(cross(f1, e0));
          n1 = normalize(cross(f2, e1));
          n2 = normalize(cross(f0, e2));

//          calcImprovedAxes(e0, f1, f0, n0, b, bestVal);
//          calcImprovedAxes(e1, f2, f1, n1, b, bestVal);
//          calcImprovedAxes(e2, f0, f2, n2, b, bestVal);

          matrix3x4SIMD resultRotation;
          resultRotation[0] = vectorSIMDf(b[0].x, b[1].x, b[2].x);
          resultRotation[1] = vectorSIMDf(b[0].y, b[1].y, b[2].y);
          resultRotation[2] = vectorSIMDf(b[0].z, b[1].z, b[2].z);
          b = resultRotation;
        }

        /**
         *
         * @param[in] b
         * @param[out] bMin
         * @param[out] bMax
         * @param[out] dims
         */
        inline void calcOBBDimensions(
          const matrix3x4SIMD& b,
          vectorSIMDf& bMin, vectorSIMDf& bMax, vectorSIMDf& dims
        ) noexcept
        {
          // b is an orthonormal matrix, which represent rotation of the bounding box
          bMin.x = bMax.x = dot(m_baseVtxPos, b[0]).x;
          bMin.y = bMax.y = dot(m_baseVtxPos, b[1]).x;
          bMin.z = bMax.z = dot(m_baseVtxPos, b[2]).x;

          for(size_t i = 1u; i < m_totalVtxCount; i++)
          {
            if(isIdxInvalid(i)) break; // cap to max

            const auto vtxPos = m_selVtxArray[i];

            for(uint32_t j = 0u; j < 3u; j++)
            {
              float proj = dot(vtxPos, b[j]).x;

              if(proj > bMax[j]) bMax[j] = proj;
              if(proj < bMin[j]) bMin[j] = proj;
            }
          }

          dims = bMax - bMin;
        }

        /**
         *
         * @param[in] obbType
         * @param[in,out] obb
         */
        inline void finalizeOBB(OBB& obb, const OBBType& obbType = OBBType::DEFAULT)
        {
          switch (obbType)
          {
            case OBBType::AA: finalizeAAOBB(obb); break;
            case OBBType::LA: finalizeLAOBB(obb); break;
            default:
            {
              vectorSIMDf dims  = obb.dims;
              float bestVal     = obb.bestVal = dims.x * dims.y +
                                                dims.x * dims.z +
                                                dims.y * dims.z;

              const vectorSIMDf   &scale    = bestVal < m_aabb.area ? dims   : m_aabb.dims;
              const matrix3x4SIMD &rotation = bestVal < m_aabb.area ? obb.b  : matrix3x4SIMD();
              const vectorSIMDf   &mid      = vectorSIMDf();

              obb.init(scale / 2.0f, rotation, mid);
            }
            break;
          }
        }

        /**
         * @brief finalize Axis Aligned OBB (when it's degenerate edge case)
         *
         * @param obb
         */
        inline void finalizeAAOBB(OBB& obb)
        {
          const vectorSIMDf   &scale     = m_aabb.dims;
          const matrix3x4SIMD &rotation  = matrix3x4SIMD();
          const vectorSIMDf   &mid       = vectorSIMDf();

          obb.init(scale / 2.0f, rotation, mid);
        }

        /**
         * @brief finalize Line Aligned OBB (when it's degenerate triangle case)
         *
         * @param obb
         */
        inline void finalizeLAOBB(OBB& obb)
        {
          auto diff = [=](vectorSIMDf& a, vectorSIMDf& b)
          {
            return vectorSIMDf(a.x - b.x, a.y - b.y, a.z - b.z);
          };

          obb.dims = diff(obb.bMax, obb.bMin);
        }

      private:
        static_assert(projCount == 7 || projCount == 13, "size of sample directions (k) should only be either 7 or 13!");
        static_assert(maxPointCount == projCount * 2, "maximum number of points (np) should be twice the size of the sample directions!");

        const VectorSIMDfFullArr m_selVtxArray; // pre-selected vertices or total vertices (if fewer than max) of the mesh
        size_t m_totalVtxCount;
        vectorSIMDf m_baseVtxPos;

        AABB m_aabb;
    };
  }
}

#endif