// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_POLYGON_GEOMETRY_H_INCLUDED_
#define _NBL_ASSET_I_POLYGON_GEOMETRY_H_INCLUDED_


#include "nbl/asset/IGeometry.h"
#include "nbl/asset/IAccelerationStructure.h"


namespace nbl::asset
{
// Don't want to overengineer, support for variable vertex count (order) polgyon meshes is not encouraged or planned.
// If you want different polygon types in same model, bucket the polgyons into separate geometries and reunite in a single collection.
template<class BufferType>
class NBL_API2 IPolygonGeometry : public IIndexableGeometry<BufferType>
{
    public:
        //
        constexpr static inline EPrimitiveType PrimitiveType = EPrimitiveType::Polygon;
        inline EPrimitiveType getPrimitiveType() const override final {return PrimitiveType;}

        //
        inline uint64_t getPrimitiveCount() const override final
        {
            const auto vertexReferences = getIndexView() ? getIndexCount():m_positionView.getElementCount();
            if (vertexReferences<m_verticesForFirst)
                return 0;
            return (vertexReferences-m_verticesForFirst)/m_verticesPerSupplementary;
        }

        //
        inline uint32_t getJointCount() const override final
        {
            return m_jointCount;
        }

        //
        inline uint16_t getVerticesForFirst() const {return m_verticesForFirst;}
        inline uint16_t getVerticesPerSupplementary() const {return m_verticesPerSupplementary;}

    protected:
        virtual ~IPolygonGeometry() = default;

        uint32_t m_jointCount = 0;
        // Simple geometries like point, line, triangle and patch have `m_verticesForFirst==m_verticesPerSupplementary`
        // Note that this is also true for adjacency lists, but they're the reason `m_verticesForFirst` is not named `m_order` or `m_n`
        // While strips and fans have `m_verticesForFirst>m_verticesPerSupplementary`
        uint16_t m_verticesForFirst = 1;
        uint16_t m_verticesPerSupplementary = 1;
};

}
#endif