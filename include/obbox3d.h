// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_KDOP_OBB_H_INCLUDED__
#define __NBL_KDOP_OBB_H_INCLUDED__

namespace nbl
{
  namespace asset { class IMeshManipulator; }
  namespace ext { namespace DebugDraw { class CDraw3DLine; } } // for debug/testing only

  namespace core
  {
    /**
     * @struct OBB (Oriented Bounding Box) POD
     *
     * @note OBB is currently computed only by DiTO-k algorithms
     * if need PCA or BF methods may need to update to include
     * eigen vectors, etc.
     *
     * @note PCA method example for Maya: https://obb.readthedocs.io/en/latest/index.html
     *
     * @note if need access to superclass functions/members, should
     * explicitly friend the corresponding structures as OBB
     * end-use is of a POD
     */
    struct OBB final : private aabbox3dsf
    {
      template<typename T, typename UVector>
      friend class aabbox3d; // to allow access minEdge & maxEdge in obb's addInternalBox
      template<uint8_t, uint8_t>
      friend class KDOP;
      friend class OBBHelper;
      friend class asset::IMeshManipulator;
      friend class ext::DebugDraw::CDraw3DLine; // for debug/testing only

      enum class Type
      {
        DEFAULT,
        AA, // Axis Aligned
        LA  // Line Aligned
      };

      matrix3x4SIMD asMat3x4;
      matrix3x4SIMD orientation;  // (b - u,v,w) best found OBB orientation (transposed)

      vectorSIMDf origin;
      vectorSIMDf midpoint; // midpoint of the vertices
      vectorSIMDf extent;   // dimensions of the vertices

      float bestVal = 0;  // the best obb quality value
    };

    class OBBHelper final
    {
      template<uint8_t, uint8_t>
      friend class KDOP;

      private:
        explicit OBBHelper(OBB obb) : m_obb(std::move(obb)) {}

        /**
         *
         * @param[in] scale
         * @param[in] rotation
         * @param[in] mid
         *
         */
        inline OBB& init(
          const vectorSIMDf&    scale     = vectorSIMDf(),
          const matrix3x4SIMD&  rotation  = matrix3x4SIMD(),
          const vectorSIMDf&    mid       = vectorSIMDf()
        ) noexcept
        {
          m_obb.midpoint  = mid;
          m_obb.extent    = scale * 0.5f;

          matrix3x4SIMD scaleMat;
          scaleMat.setScale(m_obb.extent);

          m_obb.asMat3x4 = concatenateBFollowedByA(rotation, scaleMat);
//          m_obb.asMat3x4.setTranslation(m_obb.midpoint);

          return m_obb;
        }

        /**
         * @brief
         *
         * @param[in] scale
         * @param[in] rotation
         * @param[in] min
         * @param[in] max
         *
         * TODO: Store world-space coordinate of the local OBB origin
         * Construct from offset (local origin) and full extent, then OBB can have the form:
         *    column0 = localXaxis
         *    column1 = localYaxis
         *    column2 = localZaxis
         *    column3 = offset
         * no concatenation of matrices necessary
         */
        inline OBB& init(
          const vectorSIMDf& scale, const matrix3x4SIMD& rotation,
          const vectorSIMDf& min,   const vectorSIMDf&   max
        ) noexcept
        {
          m_obb.midpoint  = getMidpoint(rotation, min, max);
          m_obb.extent    = scale * .5f;

          matrix3x4SIMD mat;
          mat.setScale(m_obb.extent);

          m_obb.asMat3x4 = concatenateBFollowedByA(rotation, mat);
          m_obb.asMat3x4.setTranslation(m_obb.midpoint);

          return m_obb;
        }

        /**
         *
         * @param[in] rotation
         * @param[in] min
         * @param[in] max
         * @return midpoint (center point) position of kdop obb
         */
        inline static vectorSIMDf getMidpoint(
          const matrix3x4SIMD&  rotation,
          const vectorSIMDf&    min,
          const vectorSIMDf&    max
        ) noexcept
        {
          vectorSIMDf mid;

          // q is the midpoint expressed in the OBB's local coordinate system
          const auto &q = (max + min) * 0.5f; // (l + s) / 2

          // Compute midpoint expressed in the standard base
          mid  = rotation[0] * q.x;
          mid += rotation[1] * q.y;
          mid += rotation[2] * q.z;

          return mid;
        }

      private:
        OBB m_obb;
    };

    /**
     * @class DiTOBase Base class for Ditetrahedron OBB Algorithms
     *
     * @tparam projCount        (ns) Number of sample directions (slabs/projections) = k/2
     * @tparam maxPointCount    (np) Maximum Number of points selected along the sample directions
     *
     * @brief holds common functionality and data across LBT, KDOP,
     * and other DiTO-k algos with uniformly distributed normal sets
     */
    template<uint8_t projCount, uint8_t maxPointCount = projCount * 2>
    class DiTOBase
    {
      static_assert(
        maxPointCount == projCount * 2,
        "maximum number of points (np) should be twice the size of the sample slab directions!"
      );

      protected:
        using VectorSIMDfList   = std::vector<core::vectorSIMDf>;
        using VectorSIMDfArray  = std::array<vectorSIMDf, projCount>;

        enum class DegenCase
        {
          NONE,
          EDGE, // the two points are located at the same spot (max|p0 - p1| < threshold (eps))
          TRI,  // the point (p2) is collinear with the endpoints of the first long edge (e0)
          TETRA // the computed fourth point (q0,q1) lies in the plane of the already-found base triangle
        };

      protected:
        const float m_threshold = 0.000001f; // epsilon (eps)
    };

    /**
     * @class LBT (Large Base Triangle)
     *
     * @tparam projCount        (ns) Number of sample directions (slabs/projections) = k/2
     */
    template<uint8_t projCount>
    class LBT final : private DiTOBase<projCount>
    {
      using VectorSIMDfList   = typename DiTOBase<projCount>::VectorSIMDfList;
      using VectorSIMDfArray  = typename DiTOBase<projCount>::VectorSIMDfArray;
      using DegenCase         = typename DiTOBase<projCount>::DegenCase;

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
         * @param[in] minVerts
         * @param[in] maxVerts
         */
        inline void calcFarthestPointPair(
          const VectorSIMDfArray& minVerts,
          const VectorSIMDfArray& maxVerts
        ) noexcept
        {
          auto maxDistance = distancesquared(minVerts[0], maxVerts[0]).x;
          uint32_t farthestPairIdx = 0u;

          for(uint32_t k = 1u; k < projCount; k++)
          {
            auto distance = distancesquared(minVerts[k], maxVerts[k]).x;

            if(distance > maxDistance) { maxDistance = distance; farthestPairIdx = k; }
          }

          m_data.p0 = minVerts[farthestPairIdx];
          m_data.p1 = maxVerts[farthestPairIdx];

          // Detrimental Case 1 Check: Degenerate Edge
          // If the found furthest points are located very close, return OBB aligned with the initial AABB
          if(distancesquared(m_data.p0, m_data.p1).x < this->m_threshold) m_degenerate = DegenCase::EDGE;
        }

        /**
         * @brief calculate third vertex furthest away from line given by p0, e0
         *
         * @param[in] selVtxList
         * @param[in] vtxCount
         */
        inline void calcThirdPoint(
          const VectorSIMDfList& selVtxList,
          const uint8_t vtxCount
        ) noexcept
        {
          // TODO: Need to be consistent, if it's vector from p0 then the directions all need to be -p0 for oriented lines
          m_data.e0 = normalize(m_data.p0 - m_data.p1);

          auto vtxPos = selVtxList[0];

          auto maxDistance = findPointToLineDistance(m_data.p0, m_data.e0, vtxPos);
          m_data.p2 = vtxPos;

          for(auto i = 1u; i < vtxCount; i++)
          {
            vtxPos = selVtxList[i];
            const auto distance = findPointToLineDistance(m_data.p0, m_data.e0, vtxPos);

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
        { m_data.n  = normalize(cross(m_data.e1, m_data.e0)); }

      private:
        /**
         *
         * @param[in] p
         * @param[in] dir
         * @param[in] vtx
         *
         * @return point to line distance
         */
        inline float findPointToLineDistance(
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
     * @tparam projCount        (ns) Number of sample directions (slabs/projections) = k/2
     * @tparam maxPointCount    (np) Maximum Number of points selected along the sample directions
     *
     * @brief DiTO-14 and DiTO-26 (14-DOP, 26-DOP) are currently supported only.
     */
    template<uint8_t projCount = 13, uint8_t maxPointCount = projCount * 2>
    class KDOP final : private DiTOBase<projCount, maxPointCount>
    {
      static_assert(
        projCount == 7 || projCount == 13,
        "number of sample slab directions (ns) should only be either 7 or 13 (projCount = k/2)!"
      );

      using VectorSIMDfList   = typename DiTOBase<projCount, maxPointCount>::VectorSIMDfList;
      using VectorSIMDfArray  = typename DiTOBase<projCount>::VectorSIMDfArray;
      using DegenCase         = typename DiTOBase<projCount>::DegenCase;
      using LBTData           = typename LBT<projCount>::Data;

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

      /**
       * @struct EtaNormal
       * @brief constructs efficient normal sets (predefined slab directions)
       * for DiTO-14 and DiTO-26 with k-dop computation (14-DOP, 26-DOP)
       */
      struct EtaNormal final
      {
        VectorSIMDfArray sets;

        EtaNormal()
        {
          // DiTO-14 Normal sets (AABB + Corners)
          sets = {
            vectorSIMDf(1.0f, 0.0f, 0.0f),  // slab 0 face normal (6-DOP)
            vectorSIMDf(0.0f, 1.0f, 0.0f),  // slab 1 face normal (6-DOP)
            vectorSIMDf(0.0f, 0.0f, 1.0f),  // slab 2 face normal (6-DOP)

            vectorSIMDf(1.0f, 1.0f, 1.0f),  // slab 3 corner normal (14-DOP)
            vectorSIMDf(1.0f, 1.0f, -1.0f), // slab 4 corner normal (14-DOP)
            vectorSIMDf(1.0f, -1.0f, 1.0f), // slab 5 corner normal (14-DOP)
            vectorSIMDf(1.0f, -1.0f, -1.0f) // slab 6 corner normal (14-DOP)
          };

          // DiTO-26 Normal sets (AABB + Corners + Edges)
          if(projCount == 13)
          {
            sets[7]   = vectorSIMDf(1.0f, 1.0f, 0.0f);  // slab 7  edge normal (26-DOP)
            sets[8]   = vectorSIMDf(1.0f, -1.0f, 0.0f); // slab 8  edge normal (26-DOP)
            sets[9]   = vectorSIMDf(1.0f, 0.0f, 1.0f);  // slab 9  edge normal (26-DOP)
            sets[10]  = vectorSIMDf(1.0f, 0.0f, -1.0f); // slab 10 edge normal (26-DOP)
            sets[11]  = vectorSIMDf(0.0f, 1.0f, 1.0f);  // slab 11 edge normal (26-DOP)
            sets[12]  = vectorSIMDf(0.0f, 1.0f, -1.0f); // slab 12 edge normal (26-DOP)
          }
        }
      };

      using OBBType             = OBB::Type;
      using LBT                 = LBT<projCount>;
      using VtxPosGetCallback   = std::function<vectorSIMDf(size_t)>;
      using SlabArray           = std::array<Slab, projCount>;

      public:
        /**
         *
         * @param[in] totalVtxList vector containing total vertices
         * @param[in] totalVtxCount total number of vertices
         */
        KDOP(const VectorSIMDfList& totalVtxList, const size_t totalVtxCount)
        : m_totalVtxList  (totalVtxList)
        , m_totalVtxCount (totalVtxCount)
        , m_selVtxList    (VectorSIMDfList(maxPointCount))
        , m_selPointCount (maxPointCount) {}

        /**
         * @note this ctor may be slightly slower due to lambda's heap allocation but more convenient
         *
         * TODO: figure if can be further optimized
         *
         * @param[in] vtxPosGetCallback lambda expression returning vertex position by specified index
         * @param[in] totalVtxCount total number of vertices
         */
        KDOP(const VtxPosGetCallback& vtxPosGetCallback, const size_t totalVtxCount)
        : m_totalVtxList  (getTotalVtxList(vtxPosGetCallback, totalVtxCount))
        , m_totalVtxCount (totalVtxCount)
        , m_selVtxList    (VectorSIMDfList(maxPointCount))
        , m_selPointCount (maxPointCount) {}

      public:
        /**
         *
         * @param[out] obb
         */
        inline void compute(OBB& obb) noexcept
        {
          if(m_totalVtxCount <= 0)
          { finalizeOBB(obb, OBBType::AA); return; }

          VectorSIMDfArray minVerts, maxVerts;
          SlabArray slabs;

          calcInitialExtremalValues(minVerts, maxVerts, slabs);
          calcAABBOrientationForStdBase(slabs, obb.bestVal);

          if(m_totalVtxCount > maxPointCount)
          {
            m_selVtxList.reserve(maxPointCount);

            for(auto k = 0u; k < projCount; k++)
            {
              m_selVtxList[k]              = minVerts[k];
              m_selVtxList[k + projCount]  = maxVerts[k];
            }
          }
          else
          {
            m_selPointCount = m_totalVtxCount;
            m_selVtxList    = m_totalVtxList;
          }

          LBT lbt;
          lbt.calcFarthestPointPair(minVerts, maxVerts);

          if(lbt.getDegenerateCase() == DegenCase::EDGE)
          { finalizeOBB(obb, OBBType::AA, true); return; }

          lbt.calcThirdPoint(m_selVtxList, m_selPointCount);

          if(lbt.getDegenerateCase() == DegenCase::TRI)
          { finalizeOBB(obb, OBBType::LA, true, lbt.getOffendingEdge()); return; }

          lbt.calcEdges();
          lbt.calcNormal();

          calcOBBAxes(lbt.getData(), obb.orientation, obb.bestVal);
          calcOBBDimensions(obb.orientation, obb.MinEdge, obb.MaxEdge, obb.extent);

          finalizeOBB(obb);
        }

      private:
        /**
         * @brief calculate/initialize max & min points and projections of the meshBuffer by predefined slab directions
         *
         * @param[out] minVerts
         * @param[out] maxVerts
         * @param[out] slabs
         */
        inline void calcInitialExtremalValues(
          VectorSIMDfArray& minVerts,
          VectorSIMDfArray& maxVerts,
          SlabArray& slabs
        ) noexcept
        {
          EtaNormal normal;
          const auto& baseVtxPos = m_totalVtxList[0];

          for(auto k = 0u; k < projCount; k++)
          { slabs[k].min = slabs[k].max = getNormalizedSlab(baseVtxPos, normal.sets[k], k); }

          // TODO: implement a SIMD memset to zero out faster
          std::fill(std::begin(minVerts), std::end(minVerts), baseVtxPos);
          std::fill(std::begin(maxVerts), std::end(maxVerts), baseVtxPos);

          for(auto i = 1u; i < m_totalVtxCount; i++)
          {
            const auto& vtxPos = m_totalVtxList[i];

            for(auto j = 0u; j < projCount; j++)
            {
              auto& minSlab = slabs[j].min;
              auto& maxSlab = slabs[j].max;
              auto vtxProj = getNormalizedSlab(vtxPos, normal.sets[j], j);

              if(vtxProj < minSlab) { minSlab = vtxProj; minVerts[j] = vtxPos; }
              if(vtxProj > maxSlab) { maxSlab = vtxProj; maxVerts[j] = vtxPos; }
            }
          }
        }

        /**
         * @brief calculate size of AABB (slabs 0, 1 and 2 define AABB)
         * and initialize the best found orientation so far to be the standard base
         *
         * @note three face normals are used to define AABB (6-DOP)
         *
         * @param[in] slabs
         * @param[out] bestVal
         */
        inline void calcAABBOrientationForStdBase(
          const SlabArray& slabs,
          float& bestVal
        ) noexcept
        {
          m_aabb = aabbox3dsf(
            slabs[0].min, slabs[1].min, slabs[2].min, // min edge
            slabs[0].max, slabs[1].max, slabs[2].max // max edge
          );

          // TODO: keep track of origin instead (OBB forms an orthogonal local basis with scale)

          bestVal = m_aabb.Area = getQualityValue(m_aabb.getExtent());
        }

        /**
         *
         * @param[in] baseTri
         * @param[out] orientation
         * @param[out] bestVal
         */
        inline void calcOBBAxes(
          const LBTData& baseTri,
          matrix3x4SIMD &orientation,
          float &bestVal
        ) noexcept
        {
          // Find improved OBB axes from base triangle
          findTetraEdgesImprovedAxes(baseTri.e0, baseTri.e1, baseTri.e2, baseTri.n, orientation, bestVal);

          Tetra lowerTetra, upperTetra; // two connected tetrahedron
          findDitetraPoints(baseTri.p0, baseTri.n, lowerTetra, upperTetra);

          // Find improved OBB axes from ditetrahedra
          if(upperTetra.degenerate != DegenCase::TETRA)
          { findTetraImprovedAxes(upperTetra.vtxPos, baseTri, orientation, bestVal); }

          if(lowerTetra.degenerate != DegenCase::TETRA)
          { findTetraImprovedAxes(lowerTetra.vtxPos, baseTri, orientation, bestVal); }

          // TODO: is this transpose correct? (simd impl seems to be forcing row-major for matrices and coords)
//          auto& a0 = orientation[0];
//          auto& a1 = orientation[1];
//          auto& a2 = orientation[2];
//
//          a0 = vectorSIMDf(a0.x, a1.x, a2.x);
//          a1 = vectorSIMDf(a0.y, a1.y, a2.y);
//          a2 = vectorSIMDf(a0.z, a1.z, a2.z);
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
          const auto& triProj     = dot(p, normal).x;
          const auto& baseVtxPos  = m_selVtxList[0];

          minSlab = maxSlab = dot(baseVtxPos, normal).x;
          lowerTetra.vtxPos = upperTetra.vtxPos = baseVtxPos; // q0 = q1

          for(auto i = 1u; i < m_selPointCount; i++)
          {
            const auto& vtxPos = m_selVtxList[i];
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
         * @param[out] orientation
         * @param[out] bestVal
         */
        inline void findTetraImprovedAxes(
          const vectorSIMDf& vtxPos,
          const LBTData& baseTri,
          matrix3x4SIMD& orientation,
          float& bestVal
        ) noexcept
        {
          const auto& e0 = baseTri.e0; const auto& e1 = baseTri.e1; const auto& e2 = baseTri.e2;

          // Edges towards tetra
          const auto& f0 = normalize(vtxPos - baseTri.p0);
          const auto& f1 = normalize(vtxPos - baseTri.p1);
          const auto& f2 = normalize(vtxPos - baseTri.p2);

          // Normals of tetra triangles
          const auto& n0 = normalize(cross(f1, e0));
          const auto& n1 = normalize(cross(f2, e1));
          const auto& n2 = normalize(cross(f0, e2));

          findTetraEdgesImprovedAxes(e0, f1, f0, n0, orientation, bestVal);
          findTetraEdgesImprovedAxes(e1, f2, f1, n1, orientation, bestVal);
          findTetraEdgesImprovedAxes(e2, f0, f2, n2, orientation, bestVal);
        }

        /**
         *
         * @param[in] v0
         * @param[in] v1
         * @param[in] v2
         * @param[in] normal
         * @param[out] orientation
         * @param[out] bestVal
         */
        inline void findTetraEdgesImprovedAxes(
          const vectorSIMDf& v0, const vectorSIMDf& v1, const vectorSIMDf& v2,
          const vectorSIMDf& normal,
          matrix3x4SIMD& orientation,
          float& bestVal
        ) noexcept
        {
          vectorSIMDf dMin, dMax, len;

          findExtremalProjections(normal, dMin.y, dMax.y);
          len.y = dMax.y - dMin.y;

          for(const auto& edge : { v0, v1, v2 })
          { findTriEdgeImprovedAxes(edge, normal, dMin, dMax, len, orientation, bestVal); }
        }

        /**
         *
         * @param[in] edge
         * @param[in] normal
         * @param[in,out] dMin
         * @param[in,out] dMax
         * @param[in,out] len
         * @param[out] orientation
         * @param[out] bestVal
         */
        inline void findTriEdgeImprovedAxes(
          const vectorSIMDf& edge,
          const vectorSIMDf& normal,
          vectorSIMDf& dMin, vectorSIMDf& dMax, vectorSIMDf& len,
          matrix3x4SIMD& orientation,
          float& bestVal
        ) noexcept
        {
          vectorSIMDf dir = cross(edge, normal);

          findExtremalProjections(edge, dMin.x, dMax.x);
          findExtremalProjections(dir, dMin.z, dMax.z);

          len.x = dMax.x - dMin.x;
          len.z = dMax.z - dMin.z;

          auto quality = getQualityValue(len);

          if(quality < bestVal) { bestVal = quality; orientation = matrix3x4SIMD(edge, normal, dir); }
        }

        /**
         *
         * @param[in] dir
         * @param[in,out] minSlab
         * @param[in,out] maxSlab
         * @param[in] hasAllVerts
         */
        inline void findExtremalProjections(
          const vectorSIMDf& dir,
          float& minSlab,
          float& maxSlab,
          bool hasAllVerts = false
        ) noexcept
        {
          const auto  vtxCount  = hasAllVerts ? m_totalVtxCount  : m_selPointCount;
          const auto& vtxList   = hasAllVerts ? m_totalVtxList   : m_selVtxList;

          minSlab = maxSlab = dot(vtxList[0], dir).x;

          for(auto i = 1u; i < vtxCount; i++)
          {
            auto proj = dot(vtxList[i], dir).x;

            if(proj > maxSlab) maxSlab = proj;
            if(proj < minSlab) minSlab = proj;
          }
        }

        /**
         *
         * @param[in] orientation
         * @param[out] minEdge
         * @param[out] maxEdge
         * @param[out] extent
         */
        inline void calcOBBDimensions(
          const matrix3x4SIMD& orientation,
          vectorSIMDf& minEdge, vectorSIMDf& maxEdge, vectorSIMDf& extent
        ) noexcept
        {
          findExtremalProjections(orientation[0], minEdge.x, maxEdge.x, true);
          findExtremalProjections(orientation[1], minEdge.y, maxEdge.y, true);
          findExtremalProjections(orientation[2], minEdge.z, maxEdge.z, true);

          extent = maxEdge - minEdge;
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
          const OBBType&      obbType       = OBBType::DEFAULT,
          const bool          isDegenerate  = false,
          const vectorSIMDf&  offendingEdge = vectorSIMDf()
        ) noexcept
        {
          switch(obbType)
          {
            case OBBType::AA: finalizeAAOBB(isDegenerate, obb); break;
            case OBBType::LA: finalizeLAOBB(offendingEdge, obb); break;
            default:
            {
              obb.bestVal = getQualityValue(obb.extent);
              obb.Area    = obb.bestVal * 2;

              const auto isOBBQualified = obb.bestVal < m_aabb.Area;
              const auto &scale         = isOBBQualified ? obb.extent      : m_aabb.getExtent();
              const auto &rotation      = isOBBQualified ? obb.orientation : matrix3x4SIMD();

              obb = OBBHelper(obb).init(scale, rotation, obb.MinEdge, obb.MaxEdge);
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
        inline void finalizeAAOBB(const bool isDegenerate, OBB& obb) noexcept
        {
          const vectorSIMDf   &scale    = isDegenerate ? m_aabb.getExtent() : vectorSIMDf();
          const matrix3x4SIMD &rotation = matrix3x4SIMD();
          const vectorSIMDf   &mid      = isDegenerate ? m_aabb.getCenter() : vectorSIMDf();

          obb = OBBHelper(obb).init(scale, rotation, mid);
        }

        /**
         * @brief finalize Line Aligned OBB (when it's degenerate triangle case)
         *
         * @param[in] edge
         * @param[in,out] obb
         */
        inline void finalizeLAOBB(const vectorSIMDf& edge, OBB& obb) noexcept
        {
          vectorSIMDf r = edge;

          // Make sure r is not equal to edge
          if(fabs(edge.x) > fabs(edge.y) && fabs(edge.x) > fabs(edge.z))
          { r.x = 0; }
          else if(fabs(edge.y) > fabs(edge.z) )
          { r.y = 0; }
          else
          { r.z = 0; }

          if(length(r).x < this->m_threshold)
          { r.x = r.y = r.z = 1; }

          obb.orientation[0] = edge;
          obb.orientation[1] = normalize(cross(edge, r));
          obb.orientation[2] = normalize(cross(edge, obb.orientation[1]));

          calcOBBDimensions(obb.orientation,obb.MinEdge, obb.MaxEdge, obb.extent);

          const auto &scale    = obb.extent;
          const auto &rotation = obb.orientation;

          obb = OBBHelper(obb).init(scale, rotation, obb.MinEdge, obb.MaxEdge);
        }

      private:
        /**
         *
         * @param[in] vtxPosGetCallback
         * @param[in] totalVtxCount
         * @return vector of total vertices
         */
        inline VectorSIMDfList getTotalVtxList(
          const VtxPosGetCallback& vtxPosGetCallback,
          const size_t totalVtxCount
        ) const noexcept
        {
          VectorSIMDfList vtxList;
          vtxList.reserve(totalVtxCount);

          for(auto i = 0u; i < totalVtxCount; i++)
          { vtxList.push_back(vtxPosGetCallback(i)); }

          return vtxList;
        }

        /**
         *
         * @param[in] vtxPos
         * @param[in] normalSet
         * @param[in] idx
         * @return normalized projection/slab (float)
         */
        inline float getNormalizedSlab(
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
        inline float getQualityValue(const vectorSIMDf& len) const noexcept
        { return (len.x * len.y) + (len.x * len.z) + (len.y * len.z); }

      private:
        VectorSIMDfList m_selVtxList;
        VectorSIMDfList m_totalVtxList;

        uint8_t m_selPointCount;
        uint32_t m_totalVtxCount;

        aabbox3dsf m_aabb;
    };
  }
}

#endif