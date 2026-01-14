
// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_OBB_GENERATOR_H_INCLUDED_
#define _NBL_ASSET_C_OBB_GENERATOR_H_INCLUDED_

#include "nbl/builtin/hlsl/shapes/obb.hlsl"

namespace nbl::asset
{

class COBBGenerator
{
  private:
    template<typename T, size_t CountV>
    struct Extremals
    {
      std::array<T, CountV * 2> values;

      T* minPtr()
      {
        return values.data();
      }

      const T* minPtr() const
      {
        return values.data();
      }

      T* maxPtr()
      {
        return values.data() + CountV;
      }

      const T* maxPtr() const
      {
        return values.data() + CountV;
      }

    };
  public:

    template <typename FetchVertexFn> 
      requires (std::same_as<std::invoke_result_t<FetchVertexFn, size_t>, hlsl::float32_t3>)
    static hlsl::shapes::OBB<> compute(size_t vertexCount, FetchVertexFn&& fetchFn)
    {
      // Algorithm from Game Engine Gems 2, Fast Computation of Tight-Fitting Oriented Bounding Box
      // Credit to Thomas Larsson and Linus Källberg

      constexpr size_t SAMPLE_DIR_COUNT = 7;		// Number of sample directions
      constexpr size_t SAMPLE_COUNT = SAMPLE_DIR_COUNT * 2;

      struct VertexCollection
      {
        using FetchFn = std::function<hlsl::float32_t3(size_t vertexIndex)>;
        FetchFn fetch;
        size_t size;

        static auto fromSpan(std::span<const hlsl::float32_t3> vertices) -> VertexCollection
        {
          return VertexCollection{
            .fetch = [data = vertices.data()](size_t vertexIndex)-> hlsl::float32_t3
            {
              return data[vertexIndex];
            },
            .size = vertices.size()
          };
        }

        hlsl::float32_t3 operator[](size_t index) const { return fetch(index); }
      };

      VertexCollection vertices = {
        .fetch = std::forward<FetchVertexFn>(fetchFn),
        .size = vertexCount,
      };

      if (vertices.size <= 0)
      {
        return hlsl::shapes::OBB<>::createAxisAligned({}, {});
      }

      static auto getQualityValue = [](hlsl::float32_t3 len) -> hlsl::float32_t
      {
        return len.x * len.y + len.x * len.z + len.y * len.z; //half box area
      };

      using ExtremalVertices = Extremals<hlsl::float32_t3, SAMPLE_DIR_COUNT>;
      using ExtremalProjections = Extremals<hlsl::float32_t, SAMPLE_DIR_COUNT>;
      using Axes = std::array<hlsl::float32_t3, 3>;
      using Edges = std::array<hlsl::float32_t3, 3>;

      struct ExtremalSamples
      {
        ExtremalVertices vertices;
        ExtremalProjections projections;
      };

      struct LargeBaseTriangle
      {
        hlsl::float32_t3 normal = {};
        Axes vertices = {};
        Edges edges = {};
        enum Flag
        {
          NORMAL,
          SECOND_POINT_CLOSE,
          THIRD_POINT_CLOSE
        } flag;
      };

      static auto findExtremals_7FixedDirs = [](const VertexCollection& vertices)-> ExtremalSamples
      {
        ExtremalSamples result;
        hlsl::float32_t proj;

        const auto firstVertex = vertices.fetch(0);

        auto* minProjections = result.projections.minPtr();
        auto* maxProjections = result.projections.maxPtr();

        auto* minVertices = result.vertices.minPtr();
        auto* maxVertices = result.vertices.maxPtr();

        // Slab 0: dir {1, 0, 0}
        proj = firstVertex.x;
        minProjections[0] = maxProjections[0] = proj;
        minVertices[0] = firstVertex; maxVertices[0] = firstVertex;
        // Slab 1: dir {0, 1, 0}
        proj = firstVertex.y;
        minProjections[1] = maxProjections[1] = proj;
        minVertices[1] = firstVertex; maxVertices[1] = firstVertex;
        // Slab 2: dir {0, 0, 1}
        proj = firstVertex.z;
        minProjections[2] = maxProjections[2] = proj;
        minVertices[2] = firstVertex; maxVertices[2] = firstVertex;
        // Slab 3: dir {1, 1, 1}
        proj = firstVertex.x + firstVertex.y + firstVertex.z;
        minProjections[3] = maxProjections[3] = proj;
        minVertices[3] = firstVertex; maxVertices[3] = firstVertex;
        // Slab 4: dir {1, 1, -1}
        proj = firstVertex.x + firstVertex.y - firstVertex.z;
        minProjections[4] = maxProjections[4] = proj;
        minVertices[4] = firstVertex; maxVertices[4] = firstVertex;
        // Slab 5: dir {1, -1, 1}
        proj = firstVertex.x - firstVertex.y + firstVertex.z; 
        minProjections[5] = maxProjections[5] = proj;
        minVertices[5] = firstVertex; maxVertices[5] = firstVertex;
        // Slab 6: dir {1, -1, -1}
        proj = firstVertex.x - firstVertex.y - firstVertex.z;
        minProjections[6] = maxProjections[6] = proj;
        minVertices[6] = firstVertex; maxVertices[6] = firstVertex;

        for (size_t vertex_i = 1; vertex_i < vertices.size; vertex_i++)
        {
          const auto vertex = vertices.fetch(vertex_i);
          // Slab 0: dir {1, 0, 0}
          proj = vertices.fetch(vertex_i).x;
          if (proj < minProjections[0]) { minProjections[0] = proj; minVertices[0] = vertices.fetch(vertex_i); }
          if (proj > maxProjections[0]) { maxProjections[0] = proj; maxVertices[0] = vertices.fetch(vertex_i); }
          // Slab 1: dir {0, 1, 0}
          proj = vertices.fetch(vertex_i).y;
          if (proj < minProjections[1]) { minProjections[1] = proj; minVertices[1] = vertices.fetch(vertex_i); }
          if (proj > maxProjections[1]) { maxProjections[1] = proj; maxVertices[1] = vertices.fetch(vertex_i); }
          // Slab 2: dir {0, 0, 1}
          proj = vertices.fetch(vertex_i).z;
          if (proj < minProjections[2]) { minProjections[2] = proj; minVertices[2] = vertices.fetch(vertex_i); }
          if (proj > maxProjections[2]) { maxProjections[2] = proj; maxVertices[2] = vertices.fetch(vertex_i); }
          // Slab 3: dir {1, 1, 1}
          proj = vertices.fetch(vertex_i).x + vertices.fetch(vertex_i).y + vertices.fetch(vertex_i).z;
          if (proj < minProjections[3]) { minProjections[3] = proj; minVertices[3] = vertices.fetch(vertex_i); }
          if (proj > maxProjections[3]) { maxProjections[3] = proj; maxVertices[3] = vertices.fetch(vertex_i); }
          // Slab 4: dir {1, 1, -1}
          proj = vertices.fetch(vertex_i).x + vertices.fetch(vertex_i).y - vertices.fetch(vertex_i).z;
          if (proj < minProjections[4]) { minProjections[4] = proj; minVertices[4] = vertices.fetch(vertex_i); }
          if (proj > maxProjections[4]) { maxProjections[4] = proj; maxVertices[4] = vertices.fetch(vertex_i); }
          // Slab 5: dir {1, -1, 1}
          proj = vertices.fetch(vertex_i).x - vertices.fetch(vertex_i).y + vertices.fetch(vertex_i).z; 		
          if (proj < minProjections[5]) { minProjections[5] = proj; minVertices[5] = vertices.fetch(vertex_i); }
          if (proj > maxProjections[5]) { maxProjections[5] = proj; maxVertices[5] = vertices.fetch(vertex_i); }
          // Slab 6: dir {1, -1, -1}
          proj = vertices.fetch(vertex_i).x - vertices.fetch(vertex_i).y - vertices.fetch(vertex_i).z;
          if (proj < minProjections[6]) { minProjections[6] = proj; minVertices[6] = vertices.fetch(vertex_i); }
          if (proj > maxProjections[6]) { maxProjections[6] = proj; maxVertices[6] = vertices.fetch(vertex_i); }
        }

        return result;
      };

      static auto getSqDist = [](hlsl::float32_t3 a, hlsl::float32_t3 b) -> hlsl::float32_t
      {
        return hlsl::dot(a - b, a - b);
      };

      static auto findFurthestPointPair = [](const ExtremalVertices& extremalVertices) -> std::pair<hlsl::float32_t3, hlsl::float32_t3>
      {
        int indexFurthestPair = 0;
        auto maxSqDist = getSqDist(extremalVertices.maxPtr()[0], extremalVertices.minPtr()[0]);
        for (int k = 1; k < SAMPLE_DIR_COUNT; k++)
        {
          const auto sqDist = getSqDist(extremalVertices.maxPtr()[k], extremalVertices.minPtr()[k]);
          if (sqDist > maxSqDist) { maxSqDist = sqDist; indexFurthestPair = k; }
        }
        return {
          extremalVertices.minPtr()[indexFurthestPair],
          extremalVertices.maxPtr()[indexFurthestPair]
        };
      };

      static auto getSqDistPointInfiniteEdge = [](const hlsl::float32_t3& q, const hlsl::float32_t3& p0, const hlsl::float32_t3& v) -> hlsl::float32_t
      {
        const auto u0 = q - p0;
        const auto t = dot(v, u0);
        const auto sqLen_v = hlsl::dot(v, v);
        return hlsl::dot(u0, u0) - (t * t) / sqLen_v;
      };

      static auto findFurthestPointFromInfiniteEdge = [](const hlsl::float32_t3& p0, const hlsl::float32_t3& e0, const VertexCollection& vertices)
      {
        auto maxSqDist = getSqDistPointInfiniteEdge(vertices[0], p0, e0);
        int maxIndex = 0;
        for (size_t i = 1; i < vertices.size; i++)
        {
          const auto sqDist = getSqDistPointInfiniteEdge(vertices[i], p0, e0);
          if (sqDist > maxSqDist)
          {	
            maxSqDist = sqDist;
            maxIndex = i;
          }
        }

        struct Result
        {
          hlsl::float32_t3 point;
          hlsl::float32_t sqDist;
        };
        return Result{
          vertices[maxIndex],
          maxSqDist
        };
      };

      static auto findExtremalProjs_OneDir = [](const hlsl::float32_t3& normal, const VertexCollection& vertices)
        {
          const auto firstProj = hlsl::dot(vertices[0], normal);
          auto tMinProj = firstProj, tMaxProj = firstProj;

          for (int i = 1; i < vertices.size; i++)
          {
            const auto proj = hlsl::dot(vertices[i], normal);
            if (proj < tMinProj) { tMinProj = proj; }
            if (proj > tMaxProj) { tMaxProj = proj; }
          }

          struct Result
          {
            hlsl::float32_t minProj;
            hlsl::float32_t maxProj;
          };
          return Result{ tMinProj, tMaxProj };
        };

      static auto findExtremalPoints_OneDir = [](const hlsl::float32_t3& normal, const VertexCollection& vertices)
        {
          const auto firstProj = dot(vertices[0], normal);

          auto tMinProj = firstProj, tMaxProj = firstProj;
          auto tMinVert = vertices[0], tMaxVert = vertices[0];

          for (int i = 1; i < vertices.size; i++)
          {
            const auto proj = hlsl::dot(vertices[i], normal);
            if (proj < tMinProj) { tMinProj = proj; tMinVert = vertices[i]; }
            if (proj > tMaxProj) { tMaxProj = proj; tMaxVert = vertices[i]; }
          }

          struct Result
          {
            hlsl::float32_t minProj;
            hlsl::float32_t maxProj;
            hlsl::float32_t3 minVert;
            hlsl::float32_t3 maxVert;
          };
          return Result{ tMinProj, tMaxProj, tMinVert, tMaxVert };
        };

      static auto findUpperLowerTetraPoints = [](
        const hlsl::float32_t3& n,
        const VertexCollection& vertices,
        const hlsl::float32_t3& p0)
        {
          const auto eps = 0.000001f;
          const auto extremalPoints = findExtremalPoints_OneDir(n, vertices);
          const auto triProj = hlsl::dot(p0, n);

          const auto maxVert = extremalPoints.maxProj - eps > triProj ? std::optional(extremalPoints.maxVert) : std::nullopt;
          const auto minVert = extremalPoints.minProj + eps < triProj ? std::optional(extremalPoints.minVert) : std::nullopt;

          struct Result
          {
            std::optional<hlsl::float32_t3> minVert;
            std::optional<hlsl::float32_t3> maxVert;
          };
          return Result{
            minVert,
            maxVert
          };
        };

      static auto findBestObbAxesFromTriangleNormalAndEdgeVectors = [](
        const VertexCollection& vertices,
        const hlsl::float32_t3 normal,
        const std::array<hlsl::float32_t3, 3>& edges,
        Axes& bestAxes, 
        hlsl::float32_t& bestVal)
        {	
          // The operands are assumed to be orthogonal and unit normals	
          const auto yExtremeProjs = findExtremalProjs_OneDir(normal, vertices);
          const auto yLen = yExtremeProjs.maxProj - yExtremeProjs.minProj;

          for (const auto& edge : edges)
          {
            const auto binormal = hlsl::cross(edge, normal);

            const auto xExtremeProjs = findExtremalProjs_OneDir(edge, vertices);
            const auto xLen = xExtremeProjs.maxProj - xExtremeProjs.minProj;

            const auto zExtremeProjs = findExtremalProjs_OneDir(binormal, vertices);
            const auto zLen = zExtremeProjs.maxProj - zExtremeProjs.minProj;

            const auto quality = getQualityValue({xLen, yLen, zLen});
            if (quality < bestVal)
            {
              bestVal = quality;
              bestAxes = {
                edge,
                normal,
                binormal
              };
            }
          }

        };


      static auto findBaseTriangle = [](const ExtremalVertices& extremalVertices, const VertexCollection& vertices)-> LargeBaseTriangle
        {
          constexpr hlsl::float32_t eps = 0.000001f;

          std::array<hlsl::float32_t3, 3> baseTriangleVertices = {}; // p0, p1, p2
          Edges edges;

          // Find the furthest point pair among the selected min and max point pairs
          std::tie(baseTriangleVertices[0], baseTriangleVertices[1]) = findFurthestPointPair(extremalVertices);

          // Degenerate case 1:
          // no need to compute third vertices, since base triangle is invalid
          if (getSqDist(baseTriangleVertices[0], baseTriangleVertices[1]) < eps) 
          {
            return {
              .vertices = baseTriangleVertices,
              .flag = LargeBaseTriangle::SECOND_POINT_CLOSE
            };
          }

          // Compute edge vector of the line segment p0, p1 		
          edges[0] = hlsl::normalize(baseTriangleVertices[0] - baseTriangleVertices[1]);

          // Find a third point furthest away from line given by p0, e0 to define the large base triangle
          const auto furthestPointRes = findFurthestPointFromInfiniteEdge(baseTriangleVertices[0], edges[0], vertices);
          baseTriangleVertices[2] = furthestPointRes.point;

          // Degenerate case 2:
          if (furthestPointRes.sqDist < eps)
          {
            return {
              .vertices = baseTriangleVertices,
              .edges = edges,
              .flag = LargeBaseTriangle::THIRD_POINT_CLOSE
            };
          }

          // Compute the two remaining edge vectors and the normal vector of the base triangle				
          edges[1] = hlsl::normalize(baseTriangleVertices[1] - baseTriangleVertices[2]);
          edges[2] = hlsl::normalize(baseTriangleVertices[2] - baseTriangleVertices[0]);
          const auto normal = hlsl::normalize(hlsl::cross(edges[1], edges[0]));

          return {
            .normal = normal,
            .vertices = baseTriangleVertices,
            .edges = edges,
            .flag = LargeBaseTriangle::NORMAL
          };
        };

      auto findImprovedObbAxesFromUpperAndLowerTetrasOfBaseTriangle = [](const VertexCollection& vertices,
        const LargeBaseTriangle& baseTriangle,
        Axes& bestAxes, hlsl::float32_t& bestVal)
        {

          // Find furthest points above and below the plane of the base triangle for tetra constructions 
          // For each found valid point, search for the best OBB axes based on the 3 arising triangles
          const auto upperLowerTetraVertices = findUpperLowerTetraPoints(baseTriangle.normal, vertices, baseTriangle.vertices[0]);
          if (upperLowerTetraVertices.minVert)
          {
            const auto minVert = *upperLowerTetraVertices.minVert;
            const auto f0 = normalize(minVert - baseTriangle.vertices[0]);
            const auto f1 = normalize(minVert - baseTriangle.vertices[1]);
            const auto f2 = normalize(minVert - baseTriangle.vertices[2]);
            const auto n0 = normalize(cross(f1, baseTriangle.edges[0]));
            const auto n1 = normalize(cross(f2, baseTriangle.edges[1]));
            const auto n2 = normalize(cross(f0, baseTriangle.edges[2]));
            findBestObbAxesFromTriangleNormalAndEdgeVectors(vertices, n0, { baseTriangle.edges[0], f1, f0 }, bestAxes, bestVal);
            findBestObbAxesFromTriangleNormalAndEdgeVectors(vertices, n1, { baseTriangle.edges[1], f2, f1 }, bestAxes, bestVal);
            findBestObbAxesFromTriangleNormalAndEdgeVectors(vertices, n2, { baseTriangle.edges[2], f0, f2 }, bestAxes, bestVal);
          }
          if (upperLowerTetraVertices.maxVert)
          {
            const auto maxVert = *upperLowerTetraVertices.maxVert;
            const auto f0 = normalize(maxVert - baseTriangle.vertices[0]);
            const auto f1 = normalize(maxVert - baseTriangle.vertices[1]);
            const auto f2 = normalize(maxVert - baseTriangle.vertices[2]);
            const auto n0 = normalize(cross(f1, baseTriangle.edges[0]));
            const auto n1 = normalize(cross(f2, baseTriangle.edges[1]));
            const auto n2 = normalize(cross(f0, baseTriangle.edges[2]));
            findBestObbAxesFromTriangleNormalAndEdgeVectors(vertices, n0, { baseTriangle.edges[0], f1, f0 }, bestAxes, bestVal);
            findBestObbAxesFromTriangleNormalAndEdgeVectors(vertices, n1, { baseTriangle.edges[1], f2, f1 }, bestAxes, bestVal);
            findBestObbAxesFromTriangleNormalAndEdgeVectors(vertices, n2, { baseTriangle.edges[2], f0, f2 }, bestAxes, bestVal);
          }
        };

      static auto buildObbFromAxesAndLocalMinMax = [](
        const Axes& axes, 
        const hlsl::float32_t3& localMin, 
        const hlsl::float32_t3& localMax) -> hlsl::shapes::OBB<3, hlsl::float32_t>
        {
          const auto localMid = 0.5f * (localMin + localMax);
          const hlsl::float32_t3 axesArray[3] = {axes[0], axes[1], axes[2]};
          return hlsl::shapes::OBB<3, hlsl::float32_t>::create(
            axes[0] * localMid.x + axes[1] * localMid.y + axes[2] * localMid.z,
            0.5f * (localMax - localMin),
            axesArray
          );
        };

      static auto computeObb = [](const Axes& axes, const VertexCollection& vertices, hlsl::float32_t& quality)
        {
          const auto extremalX = findExtremalProjs_OneDir(axes[0], vertices);
          const auto extremalY = findExtremalProjs_OneDir(axes[1], vertices);
          const auto extremalZ = findExtremalProjs_OneDir(axes[2], vertices);
          const auto localMin = hlsl::float32_t3{ extremalX.minProj, extremalY.minProj, extremalZ.minProj };
          const auto localMax = hlsl::float32_t3{ extremalX.maxProj, extremalY.maxProj, extremalZ.maxProj };
          quality = getQualityValue(localMax - localMin);
          return buildObbFromAxesAndLocalMinMax(axes, localMin, localMax);
        };

      static auto computeLineAlignedObb = [](const hlsl::float32_t3& u, const VertexCollection& vertices)
      {
        // Given u, build any orthonormal base u, v, w 

        // Make sure r is not equal to u
        auto r = u;
        if (fabs(u.x) > fabs(u.y) && fabs(u.x) > fabs(u.z)) { r.x = 0; }
        else if (fabs(u.y) > fabs(u.z)) { r.y = 0; }
        else { r.z = 0; }

        const auto sqLen = hlsl::dot(r, r);
        if (sqLen < FLT_EPSILON) { r.x = r.y = r.z = 1; }

        const auto v = normalize(cross(u, r));
        const auto w = normalize(cross(u, v));
        hlsl::float32_t quality;
        return computeObb({ u, v, w }, vertices, quality);
      };

      const auto extremals = findExtremals_7FixedDirs(vertices);

      const auto* minProj = extremals.projections.minPtr();
      const auto* maxProj = extremals.projections.maxPtr();

      // Determine which points to use in the iterations below 
      const auto selectedVertices = [&]
      {
          if (vertices.size < SAMPLE_COUNT) { return vertices; }
          return VertexCollection::fromSpan(extremals.vertices.values);
      }();

      // Compute size of AABB (max and min projections of vertices are already computed as slabs 0-2)
      auto alMid = hlsl::float32_t3((minProj[0] + maxProj[0]) * 0.5f, (minProj[1] + maxProj[1]) * 0.5f, (minProj[2] + maxProj[2]) * 0.5f);
      auto alLen = hlsl::float32_t3(maxProj[0] - minProj[0], maxProj[1] - minProj[1], maxProj[2] - minProj[2]);
      auto alVal = getQualityValue(alLen);


      const auto baseTriangle = findBaseTriangle(extremals.vertices, vertices);

      // Degenerate case 1:
      // If the found furthest points are located very close, return OBB aligned with the initial AABB 
      if (baseTriangle.flag == LargeBaseTriangle::SECOND_POINT_CLOSE)
        return hlsl::shapes::OBB<>::createAxisAligned(alMid, alLen);

      // Degenerate case 2:
      // If the third point is located very close to the line, return an OBB aligned with the line 
      if (baseTriangle.flag == LargeBaseTriangle::THIRD_POINT_CLOSE)
        return computeLineAlignedObb(baseTriangle.edges[0], vertices);


      Axes bestAxes = {
        hlsl::float32_t3{1.f, 0.f, 0.f},
        {0.f, 1.f, 0.f},
        {0.f, 0.f, 1.f},
      };
      auto bestVal = alVal;
      // Find best OBB axes based on the base triangle
      findBestObbAxesFromTriangleNormalAndEdgeVectors(selectedVertices, baseTriangle.normal, baseTriangle.edges, bestAxes, bestVal);

      // Find improved OBB axes based on constructed di-tetrahedral shape raised from base triangle
      findImprovedObbAxesFromUpperAndLowerTetrasOfBaseTriangle(selectedVertices, baseTriangle, bestAxes, bestVal);

      hlsl::float32_t improvedObbQuality;
      const auto obb = computeObb(bestAxes, vertices, improvedObbQuality);

      // Check if the OBB extent is still smaller than the intial AABB
      if (improvedObbQuality < alVal)
        return obb;
      return hlsl::shapes::OBB<>::createAxisAligned(alMid, alLen);

    }

};

}

#endif
