// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_
#define _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_

#include "nbl/asset/utils/CVertexHashGrid.h"


namespace nbl::asset 
{

class CSmoothNormalGenerator
{
	public:
		CSmoothNormalGenerator() = delete;
		~CSmoothNormalGenerator() = delete;

    struct SSNGVertexData
    {
      uint32_t index;									     //offset of the vertex into index buffer
			// TODO: check whether separating hash and position into its own vector or even rehash the position everytime we need will result in VertexHashGrid become faster.
			uint32_t hash;
      hlsl::float32_t3 weightedNormal;
      hlsl::float32_t3 position;							   //position of the vertex in 3D space

			hlsl::float32_t3 getPosition() const
			{
				return position;
			}

			void setHash(uint32_t hash)
			{
				this->hash = hash;
			}

			uint32_t getHash() const
			{
				return hash;
			};

    };

		using VxCmpFunction = std::function<bool(const SSNGVertexData&, const SSNGVertexData&, const ICPUPolygonGeometry*)>;

		using VertexHashMap = CVertexHashGrid<SSNGVertexData>;

	  struct Result
	  {
			VertexHashMap vertexHashGrid;
			core::smart_refctd_ptr<ICPUPolygonGeometry> geom;
	  };
		static Result calculateNormals(const ICPUPolygonGeometry* polygon, float epsilon, VxCmpFunction function);

	private:

		static VertexHashMap setupData(const ICPUPolygonGeometry* polygon, float epsilon);
		static core::smart_refctd_ptr<ICPUPolygonGeometry> processConnectedVertices(const ICPUPolygonGeometry* polygon, VertexHashMap& vertices, float epsilon, VxCmpFunction vxcmp);
};

}
#endif