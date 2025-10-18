// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_
#define _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_


#include "nbl/asset/utils/CPolygonGeometryManipulator.h"

namespace nbl::asset 
{

class CSmoothNormalGenerator
{
	public:
		CSmoothNormalGenerator() = delete;
		~CSmoothNormalGenerator() = delete;

		using VertexHashMap = CVertexHashGrid<CPolygonGeometryManipulator::SSNGVertexData>;

	  struct Result
	  {
			VertexHashMap vertexHashGrid;
			core::smart_refctd_ptr<ICPUPolygonGeometry> geom;
	  };
		static Result calculateNormals(const ICPUPolygonGeometry* polygon, float epsilon, CPolygonGeometryManipulator::VxCmpFunction function);

	private:

		static VertexHashMap setupData(const ICPUPolygonGeometry* polygon, float epsilon);
		static core::smart_refctd_ptr<ICPUPolygonGeometry> processConnectedVertices(const ICPUPolygonGeometry* polygon, VertexHashMap& vertices, float epsilon, CPolygonGeometryManipulator::VxCmpFunction vxcmp);
};

}
#endif