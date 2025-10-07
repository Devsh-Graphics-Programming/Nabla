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

		static core::smart_refctd_ptr<ICPUPolygonGeometry> calculateNormals(const ICPUPolygonGeometry* polygon, bool enableWelding, float epsilon, CPolygonGeometryManipulator::VxCmpFunction function);

	private:
		using VertexHashMap = CVertexHashMap;

		static VertexHashMap setupData(const ICPUPolygonGeometry* polygon, float epsilon);
		static core::smart_refctd_ptr<ICPUPolygonGeometry> processConnectedVertices(const ICPUPolygonGeometry* polygon, VertexHashMap& vertices, float epsilon, CPolygonGeometryManipulator::VxCmpFunction vxcmp);
		static core::smart_refctd_ptr<ICPUPolygonGeometry> weldVertices(const ICPUPolygonGeometry* polygon, VertexHashMap& vertices, float epsilon);
};

}
#endif