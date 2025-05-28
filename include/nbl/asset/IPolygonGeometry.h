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
        using SDataView = IGeometry<BufferType>::SDataView;

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

        // Its also a Max Joint ID that `m_jointWeightViews` can reference
        inline uint32_t getJointCount() const override final
        {
            return m_jointCount;
        }

        // SoA instead of AoS, first component is the first bone influece, etc.
        struct SJointWeight
        {
            SDataView indices;
            // Assumption is that only non-zero weights are present, which is why the joints are indexed (sparseness)
            // Zero weights are acceptable but only if they form a contiguous block including the very last component of the very last weight view
            SDataView weights;
        };
        // It's a vector in case you need more than 4 influences per bone
        inline const core::vector<SJointWeight>& getJointWeightViews() const {return m_jointWeightViews;}

        // For when the geometric normal of the patch isn't enough and you want interpolated custom normals
        inline SDataView& getNormalView() const {return m_normalView;}

        // For User defined semantics
        inline const core::vector<SDataView>& getAuxAttributeViews() const {return m_auxAttributeViews;}

        // Simple geometries like point, line, triangle and patch have `m_verticesForFirst==m_verticesPerSupplementary`
        // Note that this is also true for adjacency lists, but they're the reason `m_verticesForFirst` is not named `m_order` or `m_n`
        // While strips and fans have `m_verticesForFirst>m_verticesPerSupplementary`
        inline uint16_t getVerticesForFirst() const {return m_verticesForFirst;}
        inline uint16_t getVerticesPerSupplementary() const {return m_verticesPerSupplementary;}

    protected:
        virtual ~IPolygonGeometry() = default;

        //
        core::vector<SJointWeight> m_jointWeightViews = {};
        //
        core::vector<SDataView> m_auxAttributeViews = {};
        //
        SDataView m_normalView = {};
        //
        uint32_t m_jointCount = 0;
        //
        uint16_t m_verticesForFirst = 1;
        uint16_t m_verticesPerSupplementary = 1;
};

}
#endif