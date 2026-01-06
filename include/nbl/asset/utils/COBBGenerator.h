
// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_OBB_GENERATOR_H_INCLUDED_
#define _NBL_ASSET_C_OBB_GENERATOR_H_INCLUDED_

#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
#include "nbl/builtin/hlsl/shapes/obb.hlsl"

namespace nbl::asset
{

class COBBGenerator
{
  public:

    using VertexCollection = CPolygonGeometryManipulator::VertexCollection;

    static hlsl::shapes::OBB<> compute(const VertexCollection& vertices);

};

}

#endif
