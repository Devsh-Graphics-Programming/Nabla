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
     *
     * @note OBB is currently computed only by DiTO-k algorithms
     * if need PCA or BF methods may need to update to include
     * eigen vectors, etc.
     *
     * PCA method example for Maya: https://obb.readthedocs.io/en/latest/index.html
     */
    struct OBB final
    {
      matrix3x4SIMD asMat3x4; // equivalent of glsl's mat4x3

      float bestVal = 0;  // the best obb quality value

      vectorSIMDf bMin;
      vectorSIMDf bMax;

      vectorSIMDf origin;
      vectorSIMDf midPoint;
      matrix3x4SIMD b;      // The currently best found OBB orientation (transposed)
      vectorSIMDf extents;  // (bLen)

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

        extents   = scale * 0.5f;
        midPoint  = mid;

        scaleMat.setScale(extents);

        asMat3x4 = concatenateBFollowedByA(rotation, scaleMat);
        asMat3x4.setTranslation(midPoint);
      }
    };

    /**
     * @class DiTOBase Base class for Ditetrahedron OBB Algorithms
     *
     * @tparam projCount        (k)  Number of sample directions (projections)
     * @tparam maxPointCount    (np) Maximum Number of points selected along the sample directions
     *
     * @brief holds common functionality and data across LBT, KDOP,
     * and other DiTO-k algos with uniformly distributed normal sets
     */
    template<uint8_t projCount, uint8_t maxPointCount = projCount * 2>
    class DiTOBase
    {
      protected:
        using VectorSIMDfHalfArr    = std::array<vectorSIMDf, projCount>;
        using VectorSIMDfFullArr    = std::array<vectorSIMDf, maxPointCount>;
        using VectorSIMDfArrayPair  = std::pair<VectorSIMDfHalfArr, VectorSIMDfHalfArr>;

        enum class DegenCase
        {
          NONE,
          EDGE, // max|p0 - p1| < threshold (eps)
          TRI,  // max|p0 - e0| < threshold (eps)
          TETRA //
        };

      protected:
        const float m_threshold = 0.000001f; // epsilon (eps)
    };

    /**
     * @class LBT (Large Base Triangle)
     *
     * @tparam projCount        (k)  Number of sample directions (projections)
     * @tparam maxPointCount    (np) Maximum Number of points selected along the sample directions
     */
    template<uint8_t projCount, uint8_t maxPointCount = projCount * 2>
    class LBT final : private DiTOBase<projCount, maxPointCount>
    {
      using VectorSIMDfFullArr    = typename DiTOBase<projCount>::VectorSIMDfFullArr;
      using VectorSIMDfArrayPair  = typename DiTOBase<projCount>::VectorSIMDfArrayPair;
      using DegenCase             = typename DiTOBase<projCount>::DegenCase;

      public:
        struct Data final
        {
          vectorSIMDf p0, p1, p2; // Vertices of the large base triangle
          vectorSIMDf e0, e1, e2; // Edge vectors of the large base triangle
          vectorSIMDf n;          // Unit normal of the large base triangle
        };

      public:
        inline Data         getData()           const noexcept { return m_data; }
        inline vectorSIMDf  getOffendingEdge()  const noexcept { return m_offendingEdge; }
        inline DegenCase    getDegenerateCase() const noexcept { return m_degenerate; }

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

          auto maxDistance = distancesquared(minVtxPos[0], maxVtxPos[0]).x;
          uint32_t farthestPairIdx = 0u;

          for(uint32_t k = 1u; k < projCount; k++)
          {
            auto distance = distancesquared(minVtxPos[k], maxVtxPos[k]).x;

            if(distance > maxDistance) { maxDistance = distance; farthestPairIdx = k; }
          }

          m_data.p0 = minVtxPos[farthestPairIdx];
          m_data.p1 = maxVtxPos[farthestPairIdx];

          // Detrimental Case 1 Check: Degenerate Edge
          // If the found furthest points are located very close, return OBB aligned with the initial AABB
          if(distancesquared(m_data.p0, m_data.p1).x < this->m_threshold) m_degenerate = DegenCase::EDGE;
        }

        /**
         * @brief calculate third vertex furthest away from line given by p0, e0
         *
         * @param[in] selVtxArr
         * @param[in] vtxCount
         */
        inline void calcThirdPoint(
          const VectorSIMDfFullArr& selVtxArr,
          const uint8_t vtxCount
        ) noexcept
        {
          // TODO: Need to be consistent, if it's vector from p0 then the directions all need to be -p0 for oriented lines
          m_data.e0 = normalize(m_data.p0 - m_data.p1);

          auto vtxPos = selVtxArr[0];

          auto maxDistance = calcPointToLineDistance(m_data.p0, m_data.e0, vtxPos);
          m_data.p2 = vtxPos;

          for(auto i = 1u; i < vtxCount; i++)
          {
            vtxPos = selVtxArr[i];
            const auto distance = calcPointToLineDistance(m_data.p0, m_data.e0, vtxPos);

            if(distance > maxDistance) { maxDistance = distance; m_data.p2 = vtxPos; }
          }

          // Detrimental Case 2 Check: Degenerate Triangle
          // If the third point is located very close to the line, return an OBB aligned with the line
          if(maxDistance < this->m_threshold) { m_degenerate = DegenCase::TRI; m_offendingEdge = m_data.e0; }
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

      private:
        /**
         *
         * @param[in] p
         * @param[in] dir
         * @param[in] vtx
         *
         * @return point to line distance
         */
        inline float calcPointToLineDistance(
          const vectorSIMDf& p,
          const vectorSIMDf& dir,
          const vectorSIMDf& vtx
        ) const noexcept
        {
          _NBL_DEBUG_BREAK_IF(length(dir).x < this->m_threshold);

          // TODO: make sure this calculation is correct

//          auto u0 = vtx - p;
//          auto t = dot(dir, u0);
//          auto dirLen = length(dir);
//          auto dist = length(u0) - t * t / dirLen;

//          return dist.x;

          return length(cross(vtx - p, dir)).x;
        }

      private:
        Data m_data;
        DegenCase m_degenerate = DegenCase::NONE;
        vectorSIMDf m_offendingEdge;
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
    class KDOP final : private DiTOBase<projCount, maxPointCount>
    {
      using VectorSIMDfHalfArr    = typename DiTOBase<projCount>::VectorSIMDfHalfArr;
      using VectorSIMDfFullArr    = typename DiTOBase<projCount>::VectorSIMDfFullArr;
      using VectorSIMDfArrayPair  = typename DiTOBase<projCount>::VectorSIMDfArrayPair;
      using DegenCase             = typename DiTOBase<projCount>::DegenCase;
      using LBTData               = typename LBT<projCount>::Data;

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

      struct Tetra final
      {
        vectorSIMDf vtxPos;
        DegenCase degenerate = DegenCase::NONE;
      };

      struct AABB final
      {
        vectorSIMDf origin;
        vectorSIMDf extents;  // axis aligned dimensions of the vertices (alLen)
        vectorSIMDf midPoint; // axis aligned mid point of the vertices (alMid)
        float area = 0;       // quality measure of the axis-aligned box (alVal)
      };

      /**
       * @struct EtaNormal
       * @brief constructs efficient normal sets (predefined slab directions)
       * for DiTO-14 and DiTO-26 with k-dop computation
       */
      struct EtaNormal final
      {
        VectorSIMDfHalfArr sets;

        EtaNormal()
        {
          // DiTO-14 Normal sets
          sets = {
            vectorSIMDf(1.0f, 0.0f, 0.0f),
            vectorSIMDf(0.0f, 1.0f, 0.0f),
            vectorSIMDf(0.0f, 0.0f, 1.0f),
            vectorSIMDf(1.0f, 1.0f, 1.0f),
            vectorSIMDf(1.0f, 1.0f, -1.0f),
            vectorSIMDf(1.0f, -1.0f, 1.0f),
            vectorSIMDf(1.0f, -1.0f, -1.0f)
          };

          // DiTO-26 Normal sets
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

      using LBT                   = LBT<projCount>;
      using VtxPosGetCallback     = std::function<vectorSIMDf(size_t)>;
      using SlabArr               = std::array<Slab, projCount>;

      public:
        /**
         *
         * @param[in] selVtxArray fixed array of pre-selected vertices (capped to maxPointCount)
         * @param[in] totalVtxCount total number of vertices
         */
        KDOP(VectorSIMDfFullArr selVtxArray, const size_t totalVtxCount)
        : m_selVtxArray   (std::move(selVtxArray))
        , m_totalVtxCount (getCappedVtxCount(totalVtxCount))
        , m_baseVtxPos    (m_selVtxArray[0]) {}

        /**
         * @note this ctor may be slightly slower due to lambda's heap allocation but more convenient
         *
         * TODO: figure if can be further optimized
         *
         * @param[in] vtxPosGetCallback lambda expression returning vertex position by specified index
         * @param[in] totalVtxCount total number of vertices
         */
        KDOP(const VtxPosGetCallback& vtxPosGetCallback, const size_t totalVtxCount)
        : m_selVtxArray   (getSelectedVtxArray(vtxPosGetCallback, totalVtxCount))
        , m_totalVtxCount (getCappedVtxCount(totalVtxCount))
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

          findPredefinedExtremalValues(extremalPoints, slabs);
          calcAABBOrientationForStdBase(slabs, obb.bestVal);

          LBT lbt;
          lbt.calcFarthestPointPair(extremalPoints);

          if(m_totalVtxCount > maxPointCount)
          {
            for(auto k = 0u; k < projCount; k++)
            {
              m_selVtxArray[k]              = extremalPoints.first[k];  // minVtxPos
              m_selVtxArray[k + projCount]  = extremalPoints.second[k]; // maxVtxPos
            }
          }

          if(lbt.getDegenerateCase() == DegenCase::EDGE)
          { finalizeOBB(obb, OBBType::AA, true); return; }

          lbt.calcThirdPoint(m_selVtxArray, m_totalVtxCount);

          if(lbt.getDegenerateCase() == DegenCase::TRI)
          { finalizeOBB(obb, OBBType::LA, true, lbt.getOffendingEdge()); return; }

          lbt.calcEdges();
          lbt.calcNormal();

          calcOBBAxes(lbt.getData(), obb.b, obb.bestVal);
          calcOBBDimensions(obb.b, obb.bMin, obb.bMax, obb.extents);

          finalizeOBB(obb);
        }

      private:
         /**
          * @brief calculate/initialize max & min points and predefined projections of the meshBuffer
          *
          * @param[out] extremalPoints
          * @param[out] slabs
          */
        inline void findPredefinedExtremalValues(
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
        inline void calcAABBOrientationForStdBase(
          const SlabArr& slabs,
          float& bestVal
        ) noexcept
        {
          // TODO: keep track of origin instead (OBB forms an orthogonal local basis with scale)
          m_aabb.origin = vectorSIMDf(

          );

          m_aabb.midPoint = vectorSIMDf(
            slabs[0].min + slabs[0].max,
            slabs[1].min + slabs[1].max,
            slabs[2].min + slabs[2].max
          ) * 0.5f;

          m_aabb.extents = vectorSIMDf(
            slabs[0].max - slabs[0].min,
            slabs[1].max - slabs[1].min,
            slabs[2].max - slabs[2].min
          );

          m_aabb.area = getQualityValue(m_aabb.extents);

          bestVal = m_aabb.area;
        }

        /**
         *
         * @param[in] baseTri
         * @param[out] b
         * @param[out] bestVal
         */
        inline void calcOBBAxes(
          const LBTData& baseTri,
          matrix3x4SIMD &b,
          float &bestVal
        ) noexcept
        {
          // Find improved OBB axes from base triangle
          calcTetraEdgesImprovedAxes(baseTri.e0, baseTri.e1, baseTri.e2, baseTri.n, b, bestVal);

          Tetra lowerTetra, upperTetra; // two connected tetrahedron
          findDitetraPoints(baseTri.p0, baseTri.n, lowerTetra, upperTetra);

          // Find improved OBB axes from ditetrahedra
          if(lowerTetra.degenerate != DegenCase::TETRA)
          { findTetraImprovedAxes(lowerTetra.vtxPos, baseTri, b, bestVal); }

          if(upperTetra.degenerate != DegenCase::TETRA)
          { findTetraImprovedAxes(upperTetra.vtxPos, baseTri, b, bestVal); }

          // TODO: is this transpose correct?
          matrix3x4SIMD resultRotation;
          resultRotation[0] = vectorSIMDf(b[0].x, b[1].x, b[2].x);
          resultRotation[1] = vectorSIMDf(b[0].y, b[1].y, b[2].y);
          resultRotation[2] = vectorSIMDf(b[0].z, b[1].z, b[2].z);

          b = resultRotation;
        }

        /**
         * @brief calculate remaining points (vertices) of ditetrahedron
         *
         * @param[in] p
         * @param[in] normal
         * @param[out] lowerTetra
         * @param[out] upperTetra
         */
        inline void findDitetraPoints(
          const vectorSIMDf& p,
          const vectorSIMDf& normal,
          Tetra& lowerTetra,
          Tetra& upperTetra
        ) noexcept
        {
          float minSlab, maxSlab; // min & max tetra's projection slabs
          const auto& triProj = dot(p, normal).x;

          minSlab = maxSlab = dot(m_baseVtxPos, normal).x;
          lowerTetra.vtxPos = upperTetra.vtxPos = m_baseVtxPos; // q0 = q1

          for(auto i = 1u; i < m_totalVtxCount; i++)
          {
            const auto& vtxPos = m_selVtxArray[i];
            auto proj = dot(vtxPos, normal).x;

            if(proj < minSlab) { minSlab = proj; lowerTetra.vtxPos = vtxPos; }
            if(proj > maxSlab) { maxSlab = proj; upperTetra.vtxPos = vtxPos; }
          }

          const auto threshold = this->m_threshold;
          if(maxSlab - threshold <= triProj) lowerTetra.degenerate = DegenCase::TETRA;
          if(minSlab + threshold >= triProj) upperTetra.degenerate = DegenCase::TETRA;
        }

        /**
         *
         * @param[in] vtxPos top/bottom tetrahedron vertex position
         * @param[in] baseTri
         * @param[out] b
         * @param[out] bestVal
         */
        inline void findTetraImprovedAxes(
          const vectorSIMDf& vtxPos,
          const LBTData& baseTri,
          matrix3x4SIMD& b,
          float& bestVal
        )
        {
          const auto& p0 = baseTri.p0; const auto& p1 = baseTri.p1; const auto& p2 = baseTri.p2;
          const auto& e0 = baseTri.e0; const auto& e1 = baseTri.e1; const auto& e2 = baseTri.e2;

          vectorSIMDf f0, f1, f2; // Edges towards tetra
          vectorSIMDf n0, n1, n2; // Normals of tetra triangles

          f0 = normalize(vtxPos - p0);
          f1 = normalize(vtxPos - p1);
          f2 = normalize(vtxPos - p2);

          n0 = normalize(cross(f1, e0));
          n1 = normalize(cross(f2, e1));
          n2 = normalize(cross(f0, e2));

          calcTetraEdgesImprovedAxes(e0, f1, f0, n0, b, bestVal);
          calcTetraEdgesImprovedAxes(e1, f2, f1, n1, b, bestVal);
          calcTetraEdgesImprovedAxes(e2, f0, f2, n2, b, bestVal);
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
        inline void calcTetraEdgesImprovedAxes(
          const vectorSIMDf& v0, const vectorSIMDf& v1, const vectorSIMDf& v2,
          const vectorSIMDf& n,
          matrix3x4SIMD& b,
          float& bestVal
        ) noexcept
        {
          vectorSIMDf dMin, dMax, len;

          findExtremalProjections(n, dMin.y, dMax.y);
          len.y = dMax.y - dMin.y;

          calcTriEdgeImprovedAxes(v0, n, dMin, dMax, len, b, bestVal);
          calcTriEdgeImprovedAxes(v1, n, dMin, dMax, len, b, bestVal);
          calcTriEdgeImprovedAxes(v2, n, dMin, dMax, len, b, bestVal);
        }

        /**
         *
         * @param[in] edge
         * @param[in] normal
         * @param[in,out] dMin
         * @param[in,out] dMax
         * @param[in,out] len
         * @param[out] b
         * @param[out] bestVal
         */
        inline void calcTriEdgeImprovedAxes(
          const vectorSIMDf& edge,
          const vectorSIMDf& normal,
          vectorSIMDf& dMin, vectorSIMDf& dMax, vectorSIMDf& len,
          matrix3x4SIMD& b,
          float& bestVal
        ) noexcept
        {
          vectorSIMDf dir = cross(edge, normal);

          findExtremalProjections(edge, dMin.x, dMax.x);
          findExtremalProjections(dir, dMin.z, dMax.z);

          len.x = dMax.x - dMin.x;
          len.z = dMax.z - dMin.z;

          auto quality = getQualityValue(len);

          if(quality < bestVal) { bestVal = quality; b = matrix3x4SIMD(edge, normal, dir); }
        }

        /**
         *
         * @param[in] dir
         * @param[in,out] minSlab
         * @param[in,out] maxSlab
         */
        inline void findExtremalProjections(
          const vectorSIMDf& dir,
          float& minSlab,
          float& maxSlab
        ) noexcept
        {
          minSlab = maxSlab = dot(m_baseVtxPos, dir).x;

          for(auto i = 1u; i < m_totalVtxCount; i++)
          {
            // TODO: does selVtxArray here need to be of selected extremalPoints (ordered) if vtxCount > maxPointCount?
            auto proj = dot(m_selVtxArray[i], dir).x;

            if(proj > maxSlab) maxSlab = proj;
            if(proj < minSlab) minSlab = proj;
          }
        }

        /**
         *
         * @param[in] b
         * @param[out] bMin
         * @param[out] bMax
         * @param[out] extents
         */
        inline void calcOBBDimensions(
          const matrix3x4SIMD& b,
          vectorSIMDf& bMin, vectorSIMDf& bMax, vectorSIMDf& extents
        ) noexcept
        {
          findExtremalProjections(b[0], bMin.x, bMax.x);
          findExtremalProjections(b[1], bMin.x, bMax.x);
          findExtremalProjections(b[2], bMin.x, bMax.x);

          extents = bMax - bMin;
        }

        /**
         *
         * @param[in,out] obb
         * @param[in] obbType
         * @param[in] isDegenerate
         * @param[in] offendingEdge line-aligned edge position of the degenerate triangle
         */
        inline void finalizeOBB(
          OBB& obb,
          const OBBType& obbType            = OBBType::DEFAULT,
          const bool isDegenerate           = false,
          const vectorSIMDf& offendingEdge  = vectorSIMDf()
        )
        {
          switch(obbType)
          {
            case OBBType::AA: finalizeAAOBB(isDegenerate, obb); break;
            case OBBType::LA: finalizeLAOBB(offendingEdge, obb); break;
            default:
            {
              obb.bestVal = getQualityValue(obb.extents);

              const vectorSIMDf   &scale    = obb.bestVal < m_aabb.area ? obb.extents : m_aabb.extents;
              const matrix3x4SIMD &rotation = obb.bestVal < m_aabb.area ? obb.b       : matrix3x4SIMD();

              // q is the midpoint expressed in the OBB's local coordinate system
              vectorSIMDf q = (obb.bMin + obb.bMax) * 0.5f;

              // Compute midpoint expressed in the standard base
              vectorSIMDf mid = obb.b[0] * q.x;
              mid = mid + obb.b[1] * q.y;
              mid = mid + obb.b[2] * q.z;

              obb.init(scale, rotation, mid);
            }
            break;
          }
        }

        /**
         * @brief finalize Axis Aligned OBB (when it's degenerate edge case or totalVtxCount <= 0)
         *
         * @param[in] isDegenerate
         * @param[in,out] obb
         */
        inline void finalizeAAOBB(const bool isDegenerate, OBB& obb)
        {
          const vectorSIMDf   &scale    = isDegenerate ? m_aabb.extents : vectorSIMDf();
          const matrix3x4SIMD &rotation = matrix3x4SIMD();
          const vectorSIMDf   &mid      = isDegenerate ? m_aabb.midPoint : vectorSIMDf();

          obb.init(scale, rotation, mid);
        }

        /**
         * @brief finalize Line Aligned OBB (when it's degenerate triangle case)
         *
         * @param[in] edge
         * @param[in,out] obb
         */
        inline void finalizeLAOBB(const vectorSIMDf& edge, OBB& obb)
        {
          vectorSIMDf r = edge;

          // Make sure r is not equal to edge
          if(fabs(edge.x) > fabs(edge.y) && fabs(edge.x) > fabs(edge.z))
          { r.x = 0; }
          else if(fabs(edge.y) > fabs(edge.z) )
          { r.y = 0; }
          else
          { r.z = 0; }

          if(length(r).x < this->m_threshold) { r.x = r.y = r.z = 1; }

          obb.b[0] = edge;
          obb.b[1] = normalize(cross(edge, r));
          obb.b[2] = normalize(cross(edge, obb.b[1]));

          calcOBBDimensions(obb.b,obb.bMin, obb.bMax, obb.extents);

          const vectorSIMDf   &scale    = obb.extents;
          const matrix3x4SIMD &rotation = obb.b;

          // q is the midpoint expressed in the OBB's local coordinate system
          vectorSIMDf q = (obb.bMin + obb.bMax) * 0.5f;

          // Compute midpoint expressed in the standard base
          vectorSIMDf mid = obb.b[0] * q.x;
          mid = mid + obb.b[1] * q.y;
          mid = mid + obb.b[2] * q.z;

          obb.init(scale, rotation, mid);
        }

      private:
        /**
         *
         * @param[in] vtxPosGetCallback
         * @param[in] totalVtxCount
         * @return fixed array of pre-selected vertices (capped to maxPointCount)
         */
        inline VectorSIMDfFullArr getSelectedVtxArray(
          const VtxPosGetCallback& vtxPosGetCallback,
          const size_t totalVtxCount
        ) const noexcept
        {
          VectorSIMDfFullArr vtxArray;

          for(auto i = 0u; i < getCappedVtxCount(totalVtxCount); i++)
          {
            vtxArray[i] = vtxPosGetCallback(i);
          }

          return vtxArray;
        }

        /**
         * @brief to cap the iteration going beyond maxPointCount
         *
         * @param[in] totalVtxCount
         * @return bool
         */
        inline uint8_t getCappedVtxCount(const size_t totalVtxCount) const noexcept
        { return totalVtxCount > maxPointCount ? maxPointCount : totalVtxCount; }

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
          const uint8_t idx
        ) const noexcept
        {
          const auto& slab = dot(vtxPos, normalSet);
          return idx < 3
              ? (idx == 0 ? slab.x : (idx == 1 ? slab.y : slab.z)) // slabs 0, 1 and 2 define AABB
              : slab.x + slab.y + slab.z;
        }

        /**
         *
         * @param[in] len
         * @return quality value - half box area (float)
         */
        float getQualityValue(const vectorSIMDf& len)
        {
          return  len.x * len.y +
                  len.x * len.z +
                  len.y * len.z;
        }

      private:
        static_assert(projCount == 7 || projCount == 13, "size of sample directions (k) should only be either 7 or 13!");
        static_assert(maxPointCount == projCount * 2, "maximum number of points (np) should be twice the size of the sample directions!");

        VectorSIMDfFullArr m_selVtxArray; // pre-selected vertices or total vertices (if fewer than max) of the mesh
        uint8_t m_totalVtxCount;
        vectorSIMDf m_baseVtxPos;

        AABB m_aabb;
    };
  }
}

#endif