// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_
#define _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_

#include "nbl/asset/utils/CVertexHashGrid.h"


namespace nbl::asset 
{

// TODO: implement a class template that take position type(either float32_t3 or float64_t3 as template argument
class CSmoothNormalGenerator final
{
	public:
		CSmoothNormalGenerator() = delete;
		~CSmoothNormalGenerator() = delete;

		struct VertexData
		{
			//offset of the vertex into index buffer
			uint32_t index;
			uint32_t hash;
			hlsl::float32_t3 weightedNormal;
			//position of the vertex in 3D space
			hlsl::float32_t3 position;

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

		using VxCmpFunction = std::function<bool(const VertexData&, const VertexData&, const ICPUPolygonGeometry*)>;

		using VertexHashMap = CVertexHashGrid<VertexData>;

		struct Result
		{
			VertexHashMap vertexHashGrid;
			core::smart_refctd_ptr<ICPUPolygonGeometry> geom;
		};
		static Result calculateNormals(const ICPUPolygonGeometry* polygon, float epsilon, VxCmpFunction function, const bool recomputeHash=true);

	private:
		static VertexHashMap setupData(const ICPUPolygonGeometry* polygon, float epsilon);
		static core::smart_refctd_ptr<ICPUPolygonGeometry> processConnectedVertices(const ICPUPolygonGeometry* polygon, VertexHashMap& vertices, float epsilon, VxCmpFunction vxcmp);
};

}
#endif