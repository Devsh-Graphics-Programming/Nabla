// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_POLYGON_VERTEX_WELDER_H_INCLUDED_
#define _NBL_ASSET_C_POLYGON_VERTEX_WELDER_H_INCLUDED_

#include "nbl/asset/utils/CPolygonGeometryManipulator.h"

namespace nbl::asset {

class CVertexWelder {
	
  template <typename AccelStructureT>
  static core::smart_refctd_ptr<ICPUPolygonGeometry> weldVertices(const ICPUPolygonGeometry* polygon, const AccelStructureT& vertices, float epsilon);
};

}

#endif
