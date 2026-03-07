// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_PLY_POLYGON_GEOMETRY_AUX_LAYOUT_H_INCLUDED_
#define _NBL_ASSET_S_PLY_POLYGON_GEOMETRY_AUX_LAYOUT_H_INCLUDED_

namespace nbl::asset
{

// Private PLY loader/writer contract for reserved aux slots stored in ICPUPolygonGeometry.
class SPLYPolygonGeometryAuxLayout
{
    public:
        static inline constexpr unsigned int UV0 = 0u;
};

}

#endif
