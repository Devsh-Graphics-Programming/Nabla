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
        inline const SAABBStorage& getAABB() const override final {return m_positionView.encodedDataRange;}

        //
        inline uint64_t getVertexReferenceCount() const {return getIndexView() ? getIndexCount():m_positionView.getElementCount();}

        //
        inline uint64_t getPrimitiveCount() const override final
        {
            const auto vertexReferenceCount = getVertexReferenceCount();
            if (vertexReferenceCount<m_verticesForFirst)
                return 0;
            return (vertexReferenceCount-m_verticesForFirst)/m_verticesPerSupplementary;
        }

        // For when the geometric normal of the patch isn't enough and you want interpolated custom normals
        inline const SDataView& getNormalView() const {return m_normalView;}

        // Its also a Max Joint ID that `m_jointWeightViews` can reference
        inline uint32_t getJointCount() const override final
        {
            return m_jointCount;
        }

        // SoA instead of AoS, first component is the first bone influece, etc.
        struct SJointWeight
        {
            // one thing this doesn't check is whether every vertex has a weight and index
            inline operator bool() const {return indices && isIntegerFormat(indices.format) && weights && weights.isFormatted() && indices.getElementCount()==weights.getElementCount();}

            SDataView indices;
            // Assumption is that only non-zero weights are present, which is why the joints are indexed (sparseness)
            // Zero weights are acceptable but only if they form a contiguous block including the very last component of the very last weight view
            SDataView weights;
        };
        // It's a vector in case you need more than 4 influences per bone
        inline const core::vector<SJointWeight>& getJointWeightViews() const {return m_jointWeightViews;}

        // For User defined semantics
        inline const core::vector<SDataView>& getAuxAttributeViews() const {return m_auxAttributeViews;}

        // Simple geometries like point, line, triangle and patch have `m_verticesForFirst==m_verticesPerSupplementary`
        // Note that this is also true for adjacency lists, but they're the reason `m_verticesForFirst` is not named `m_order` or `m_n`
        // While strips and fans have `m_verticesForFirst>m_verticesPerSupplementary`
        inline uint16_t getVerticesForFirst() const {return m_verticesForFirst;}
        inline uint16_t getVerticesPerSupplementary() const {return m_verticesPerSupplementary;}

        //
        inline bool valid() const
        {
            // things that make no sense
            if (m_verticesPerSupplementary==0 || m_verticesPerSupplementary>m_verticesForFirst)
                return false;
            // for polygons we must have the vertices formatted
            if (!m_positionView.isFormatted())
                return false;
            //
            const auto vertexCount = m_positionView.getElementCount();
            if (m_normalView && m_normalView.getElementCount()<vertexCount)
                return false;
            // the variable length vectors must be filled with valid views
            for (const auto& pair : m_jointWeightViews)
            if (!pair || pair.weights.getElementCount()<vertexCount)
                return false;
            for (const auto& view : m_auxAttributeViews)
            if (!view)
                return false;
            return true;
        }

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
        uint16_t m_verticesPerSupplementary = 0;
};

}
#endif