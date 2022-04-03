// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_KDOP_OBB_H_INCLUDED__
#define __NBL_KDOP_OBB_H_INCLUDED__

#include "matrix3x4SIMD.h"
#include "nbl/asset/ICPUMeshBuffer.h"

namespace nbl
{
  namespace core
  {
    /**
     * @struct OBB (Oriented Bounding Box)
     */
    struct OBB
    {
      matrix3x4SIMD asMat3x4;

      matrix3x4SIMD b;    // The currently best found OBB orientation (transposed)
      float bestVal = 0;  // the best obb quality value
      vectorSIMDf bMin, bMax, bLen;

      /**
       * TODO: Store world-space coordinate of the local OBB origin
       * Construct from offset (local origin) and full extent, then OBB can have the form:
       *    column0 = localXaxis
       *    column1 = localYaxis
       *    column2 = localZaxis
       *    column3 = offset
       * no concatenation of matrices necessary
       */
      inline void init(
        const vectorSIMDf& scale = vectorSIMDf(),
        const matrix3x4SIMD& rotation = matrix3x4SIMD(),
        const vectorSIMDf& mid = vectorSIMDf()
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
     */
    class LBT
    {
      public:
        struct Data
        {
          vectorSIMDf p0, p1, p2; // Vertices of the large base triangle
          vectorSIMDf e0, e1, e2; // Edge vectors of the large base triangle
          vectorSIMDf n;          // Unit normal of the large base triangle
        };

        /**
         *
         * @param[in] maxDistance
         */
        explicit LBT(float maxDistance)
        {
          m_maxDistance = maxDistance;
        }

        inline Data getData() const noexcept { return m_data; }
        inline bool isValid() const noexcept { return m_isValid; }

        /**
         * @brief find first 2 vertices
         * @param[in] projCount
         * @param[in] m_minVtxPos
         * @param[in] m_maxVtxPos
         */
        inline void calcPointPair(
          uint32_t projCount,
          const vectorSIMDf* minVtxPos,
          const vectorSIMDf* maxVtxPos
        ) noexcept
        {
          uint32_t bestPairIdx = 0u;

          for(uint32_t i = 1u; i < projCount; i++)
          {
            float distance = distancesquared(minVtxPos[i], maxVtxPos[i]).x;

            if (distance > m_maxDistance)
            {
              m_maxDistance = distance;
              bestPairIdx = i;
            }
          }

          m_data.p0 = minVtxPos[bestPairIdx];
          m_data.p1 = maxVtxPos[bestPairIdx];

          if(hasValidDistance(m_data.p0, m_data.p1)) m_isValid = true;
        }

        inline void calcThirdPoint(vectorSIMDf* m_selVtxPos, int m_numPoints) noexcept
        {
          // TODO: Need to be consistent, if it's vector from p0 then the directions all need to be -p0 for oriented lines
          m_data.e0 = normalize(m_data.p0 - m_data.p1);

          // Find a third vertex furthest away from line given by p0, e0
          m_maxDistance = calcPointToLineDistance(m_data.p0, m_data.e0, m_selVtxPos[0]);
          m_data.p2 = m_selVtxPos[0];
          for(uint32_t i = 1u; i < m_numPoints; i++)
          {
            float distance = calcPointToLineDistance(m_data.p0, m_data.e0, m_selVtxPos[i]);

            if(distance > m_maxDistance)
            {
              m_data.p2 = m_selVtxPos[i];
              m_maxDistance = distance;
            }
          }

          // TODO: construct OBB with local X axis along the p1-p0 line and any random rotation
          if(m_maxDistance < s_threshold)
          {
            _NBL_DEBUG_BREAK_IF(true);
          }
        }

        /**
         * @brief calculate edges the base triangle
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
         * @param[in] p0
         * @param[in] p1
         * @return bool
         */
        inline static bool hasValidDistance(const vectorSIMDf& p0, const vectorSIMDf& p1) noexcept
        {
          return distancesquared(p0, p1).x >= s_threshold;
        }

        /**
         *
         * @param[in] p
         * @param[in] dir
         * @param[in] v
         * @return point to line distance
         */
        inline static float calcPointToLineDistance(
          const vectorSIMDf& p,
          const vectorSIMDf& dir,
          const vectorSIMDf& v
        ) noexcept
        {
          _NBL_DEBUG_BREAK_IF(length(dir).x < s_threshold);

          const vectorSIMDf u = v - p;
          return length(cross(u, dir)).x;
        }

      private:
        static constexpr float s_threshold = 0.000001f;

        Data m_data;

        float m_maxDistance = 0;
        bool m_isValid = false; // max|p0 - p1| < threshold
    };

    /**
     * @class KDOP (k-dimensional Discrete Oriented Polytope)
     * @tparam projCount (k)
     *
     * @brief DiTO 26
     */
    template<uint8_t projCount = 13>
    class KDOP
    {
      struct Slab
      {
        float min = FLT_MAX;
        float max = -FLT_MAX;
      };

      struct AABB
      {
        vectorSIMDf dims;     // axis aligned dimensions of the vertices
        vectorSIMDf midPoint; // axis aligned mid point of the vertices
        float area = 0;       // quality measure of the axis-aligned box
      };

      public:
        KDOP(const asset::ICPUMeshBuffer* meshBuffer, size_t vertexCount)
        : m_meshBuffer(meshBuffer)
        , m_vtxCount  (vertexCount)
        , m_minVtxPos (m_selVtxPos) // Pointer to first half of selVert where the min points are placed
        , m_maxVtxPos (m_selVtxPos + projCount) // Pointer to the second half of selVert where the max points are placed
        {}

      public:
        inline vectorSIMDf getVertexPosition(size_t idx) const noexcept { return m_meshBuffer->getPosition(idx); }
        inline vectorSIMDf getBaseVertexPosition() const noexcept { return m_meshBuffer->getPosition(0u); }

      public:
        inline void compute(OBB& obb) noexcept
        {
          if (m_vtxCount == 0u)
          {
            _NBL_DEBUG_BREAK_IF(true);
            return;
          }

          calcMinMaxProjections();
          calcAABBSizeAsStdBase(obb.bestVal);

          LBT lbt(distancesquared(m_minVtxPos[0], m_maxVtxPos[0]).x);

          lbt.calcPointPair(projCount, m_minVtxPos, m_maxVtxPos);

          if(!lbt.isValid())
          {
            // return AABB
            obb.init(m_aabb.dims / 2.0f);
            return;
          }

          lbt.calcThirdPoint(m_selVtxPos, m_numPoints);
          lbt.calcEdges();
          lbt.calcNormal();

          const auto &lbtData = lbt.getData();
          std::pair<vectorSIMDf, vectorSIMDf> ditetraVertices; // two connected tetrahedra's vertices
          calcDitetrahedronVertices(lbtData.p0, lbtData.n, m_minVtxPos, m_maxVtxPos, ditetraVertices);
          calcOBBAxes(lbtData, ditetraVertices, obb.b, obb.bestVal);
          calcOBBDimensions(obb);
        }

      private:
        /**
         * @brief calculate max and min projections
         */
        inline void calcMinMaxProjections() noexcept
        {
          const vectorSIMDf sampleDirs[projCount] = {
            normalize(vectorSIMDf(1.0f, 0.0f, 0.0f)),
            normalize(vectorSIMDf(0.0f, 1.0f, 0.0f)),
            normalize(vectorSIMDf(0.0f, 0.0f, 1.0f)),
            normalize(vectorSIMDf(1.0f, 1.0f, 1.0f)),
            normalize(vectorSIMDf(1.0f, 1.0f, -1.0f)),
            normalize(vectorSIMDf(1.0f, -1.0f, 1.0f)),
            normalize(vectorSIMDf(1.0f, -1.0f, -1.0f)),
            normalize(vectorSIMDf(1.0f, 1.0f, 0.0f)),
            normalize(vectorSIMDf(1.0f, -1.0f, 0.0f)),
            normalize(vectorSIMDf(1.0f, 0.0f, 1.0f)),
            normalize(vectorSIMDf(1.0f, 0.0f, -1.0f)),
            normalize(vectorSIMDf(0.0f, 1.0f, 1.0f)),
            normalize(vectorSIMDf(1.0f, 1.0f, -1.0f))
          };
          const auto baseVtxPos = getBaseVertexPosition();

          for(uint32_t i = 0u; i < projCount; i++)
          {
            // should be better to compute it manually..
            m_slabs[i].min = m_slabs[i].max = dot(baseVtxPos, sampleDirs[i]).x;
          }

          std::fill(m_minVtxPos, m_minVtxPos + projCount, baseVtxPos);
          std::fill(m_maxVtxPos, m_maxVtxPos + projCount, baseVtxPos);

          for (size_t i = 1u; i < m_vtxCount; i++)
          {
            const auto vtxPos = getVertexPosition(i);

            for (uint32_t j = 0u; j < projCount; j++)
            {
              auto vtxProj = dot(vtxPos, sampleDirs[j]).x;
              auto& maxSlab = m_slabs[j].max;
              auto& minSlab = m_slabs[j].min;

              if(vtxProj > maxSlab) maxSlab = vtxProj; m_maxVtxPos[j] = vtxPos;
              if(vtxProj < minSlab) minSlab = vtxProj; m_minVtxPos[j] = vtxPos;
            }
          }
        }

        /**
         * @brief calculate size of AABB (m_slabs 0, 1 and 2 define AABB)
         * @param[in] bestVal
         */
        inline void calcAABBSizeAsStdBase(float& bestVal) noexcept
        {
          // TODO: keep track of origin instead (OBB forms an orthogonal local basis with scale)
          m_aabb.midPoint = vectorSIMDf(
            m_slabs[0].min + m_slabs[0].max,
            m_slabs[1].min + m_slabs[1].max,
            m_slabs[2].min + m_slabs[2].max
          ) * 0.5f;

          m_aabb.dims = vectorSIMDf(
            m_slabs[0].max - m_slabs[0].min,
            m_slabs[1].max - m_slabs[1].min,
            m_slabs[2].max - m_slabs[2].min
          );

          m_aabb.area = m_aabb.dims.x * m_aabb.dims.y +
                        m_aabb.dims.x * m_aabb.dims.z +
                        m_aabb.dims.y * m_aabb.dims.z; // half box area

          // Initialize the best found orientation so far to be the standard base
          bestVal = m_aabb.area;
        }

        /**
         * @brief find remaining vertices of ditetrahedron
         * @param[in] p
         * @param[in] n
         * @param[out] ditetraVertices
         */
        inline void calcDitetrahedronVertices(
          const vectorSIMDf& p,
          const vectorSIMDf& n,
          const vectorSIMDf* minVtxPos,
          const vectorSIMDf* maxVtxPos,
          std::pair<vectorSIMDf, vectorSIMDf>& ditetraVertices
        ) noexcept
        {
          // find vertices that are furthest from the plane defined by p0, p1 and p2 (base triangle)
          // on the both positive and negative half space
          // n dot x = d
          const float &d = dot(p, n).x;
          auto &[t0, t1] = ditetraVertices;
          auto baseVtxPos = minVtxPos[0];
          float minDistance;
          float maxDistance;

          minDistance = maxDistance = dot(baseVtxPos, n).x - d;
          t0 = t1 = baseVtxPos;

          for(size_t i = 0; i < m_vtxCount; i++)
          {
            const auto& maxVtxPosition = minVtxPos[i];
            const auto& minVtxPosition = maxVtxPos[i];
            float maxDist = dot(maxVtxPosition, n).x - d;
            float minDist = dot(minVtxPosition, n).x - d;

            if(maxDist > maxDistance) maxDistance = maxDist; t1 = maxVtxPosition;
            if(minDist < minDistance) minDistance = minDist; t0 = minVtxPosition;
          }
        }

        /**
         *
         * @param[in] dir
         * @param[out] minPoint
         * @param[out] maxPoint
         */
        inline void calcExternalPointProjection(const vectorSIMDf& dir, float& minPoint, float& maxPoint) noexcept
        {
          minPoint = maxPoint = dot(m_selVtxPos[0], dir).x;

          for(size_t i = 1u; i < m_numPoints; i++)
          {
            float proj = dot(m_selVtxPos[i], dir).x;

            if(proj > maxPoint) maxPoint = proj;
            if(proj < minPoint) minPoint = proj;
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

          calcExternalPointProjection(v0, dMin.x, dMax.x);
          calcExternalPointProjection(n, dMin.y, dMax.y);
          calcExternalPointProjection(m0, dMin.z, dMax.z);

          len = dMax - dMin;
          float quality = len.x * len.y + len.x * len.z + len.y * len.z;

          if(quality < bestVal)
          {
            bestVal = quality;
            b = matrix3x4SIMD(v0, n, m0);
          }

          calcExternalPointProjection(v1, dMin.x, dMax.x);
          calcExternalPointProjection(m1, dMin.z, dMax.z);

          len = dMax - dMin;
          quality = len.x * len.y + len.x * len.z + len.y * len.z;

          if(quality < bestVal)
          {
            bestVal = quality;
            b = matrix3x4SIMD(v1, n, m1);
          }

          calcExternalPointProjection(v2, dMin.x, dMax.x);
          calcExternalPointProjection(m2, dMin.z, dMax.z);

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
         * @param[in] ditetraVerts
         * @param[out] b
         * @param[out] bestVal
         */
        inline void calcOBBAxes(
          const LBT::Data& baseTri,
          const std::pair<vectorSIMDf, vectorSIMDf>& ditetraVerts,
          matrix3x4SIMD &b,
          float &bestVal
        ) noexcept
        {
          auto& p0 = baseTri.p0; auto& p1 = baseTri.p1; auto& p2 = baseTri.p2;
          auto& e0 = baseTri.e0; auto& e1 = baseTri.e1; auto& e2 = baseTri.e2;
          auto& n  = baseTri.n;

          auto [t0, t1] = ditetraVerts;

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
        }

        /**
         *
         * @param[in] obb
         */
        inline void calcOBBDimensions(OBB& obb) noexcept
        {
          auto &b = obb.b; auto &bestVal = obb.bestVal;
          auto &bMin = obb.bMin; auto &bMax = obb.bMax; auto &bLen = obb.bLen;
          auto baseVtxPos = getBaseVertexPosition();

          matrix3x4SIMD resultRotation;
          resultRotation[0] = vectorSIMDf(b[0].x, b[1].x, b[2].x);
          resultRotation[1] = vectorSIMDf(b[0].y, b[1].y, b[2].y);
          resultRotation[2] = vectorSIMDf(b[0].z, b[1].z, b[2].z);
          b = resultRotation;

          // b is an orthonormal matrix, which represent rotation of the bounding box
          bMin.x = bMax.x = dot(baseVtxPos, b[0]).x;
          bMin.y = bMax.y = dot(baseVtxPos, b[1]).x;
          bMin.z = bMax.z = dot(baseVtxPos, b[2]).x;

          for(size_t i = 1u; i < m_vtxCount; i++)
          {
            const auto vtxPos = getVertexPosition(i);

            for(uint32_t j = 0u; j < 3u; j++)
            {
              float proj = dot(vtxPos, b[j]).x;

              if(proj > bMax[j]) bMax[j] = proj;
              if(proj < bMin[j]) bMin[j] = proj;
            }
          }

          bLen = bMax - bMin;

          bestVal = bLen.x * bLen.y + bLen.x * bLen.z + bLen.y * bLen.z;

          const vectorSIMDf   &scale     = bestVal < m_aabb.area ? bLen  : m_aabb.dims;
          const matrix3x4SIMD &rotation  = bestVal < m_aabb.area ? b     : matrix3x4SIMD();
          const vectorSIMDf   &mid       = vectorSIMDf();

          obb.init(scale / 2.0f, rotation, mid);
        }

      private:
        static_assert(projCount == 13, "size of sample directions (k) should be 13!");
        int m_numPoints = projCount * 2; // Number of points selected along the sample directions
        vectorSIMDf m_selVtxPos[projCount * 2];
        Slab m_slabs[projCount];

        AABB m_aabb;
        size_t m_vtxCount;

        vectorSIMDf* m_minVtxPos;
        vectorSIMDf* m_maxVtxPos;

        const asset::ICPUMeshBuffer* m_meshBuffer;
    };
  }
}

#endif