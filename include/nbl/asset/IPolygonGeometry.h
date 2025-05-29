// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_POLYGON_GEOMETRY_H_INCLUDED_
#define _NBL_ASSET_I_POLYGON_GEOMETRY_H_INCLUDED_


#include "nbl/asset/IGeometry.h"
#include "nbl/asset/IAccelerationStructure.h"


namespace nbl::asset
{
//
class NBL_API2 IPolygonGeometryBase : public virtual core::IReferenceCounted
{
    public:
        //
        class NBL_API2 IIndexingCallback
        {
            public:
                // how many vertices per polygon
                inline uint8_t degree() const
                {
                    const auto retval = degree_impl();
                    assert(retval>0);
                    return retval;
                }
                //
                inline uint8_t reuseCount() const
                {
                    const auto retval = reuseCount_impl();
                    assert(retval<degree());
                    return retval;
                }

                struct SContext
                {
                    template<typename Index> requires std::is_integral_v<Index>
                    inline Index indexSize() const {return Index(1)<<indexSizeLog2;}

                    template<typename Index> requires std::is_integral_v<Index>
                    inline void setOutput(const Index value) const
                    {
                        switch (indexSizeLog2)
                        {
                            case 0:
                                *reinterpret_cast<uint8_t*>(out) = value;
                                break;
                            case 1:
                                *reinterpret_cast<uint16_t*>(out) = value;
                                break;
                            case 2:
                                *reinterpret_cast<uint32_t*>(out) = value;
                                break;
                            case 3:
                                *reinterpret_cast<uint64_t*>(out) = value;
                                break;
                            default:
                                assert(false);
                                break;
                        }
                    }

                    const uint8_t* const indexBuffer;
                    // no point making them smaller cause of padding
                    const uint32_t inSizeLog2;
                    const uint32_t outSizeLog2;
                    uint8_t* out;
                    uint32_t newIndexID;
                    // if `reuseCount()==0` then should be ignored
                    uint32_t restartValue = ~0ull;
                };
                inline void operator()(SContext& ctx) const
                {
                    const auto deg = degree();
                    if (ctx.newIndexID<deg || primitiveRestart)
                    {
// straight up copy
                    }
                    operator_impl(ctx);
                    ctx.newIndexID += reuseCount();
                    ctx.out += deg<<ctx.outSizeLog2;
                }

                // default is unknown
                virtual inline E_PRIMITIVE_TOPOLOGY knownTopology() const {return static_cast<E_PRIMITIVE_TOPOLOGY>(~0);}

            protected:
                virtual uint8_t degree_impl() const = 0;
                virtual uint8_t reuseCount_impl() const = 0;
                // needs to deal with being handed `!ctx.indexBuffer`
                virtual void operator_impl(const SContext& ctx) const = 0;
        };
        //
        static IIndexingCallback* PointList();
        static IIndexingCallback* LineList();
        static IIndexingCallback* TriangleList();
        static IIndexingCallback* QuadList();
        //
        static IIndexingCallback* TriangleStrip();
        static IIndexingCallback* TriangleFan();

        //
        inline const IIndexingCallback* getIndexingCallback() const {return m_indexing;}

    protected:
        virtual ~IPolygonGeometryBase() = default;

        //
        const IIndexingCallback* m_indexing = nullptr;
};

// Don't want to overengineer, support for variable vertex count (order) polgyon meshes is not encouraged or planned.
// If you want different polygon types in same model, bucket the polgyons into separate geometries and reunite in a single collection.
template<class BufferType>
class NBL_API2 IPolygonGeometry : public IIndexableGeometry<BufferType>, public IPolygonGeometryBase
{
        using base_t = IIndexableGeometry<BufferType>;

    protected:
        using EPrimitiveType = base_t::EPrimitiveType;
        using SDataView = base_t::SDataView;
        using BLASTriangles = IBottomLevelAccelerationStructure::Triangles<std::remove_const_t<BufferType>>;

    public:
        //
        virtual inline bool valid() const override
        {
            if (!base_t::valid())
                return false;
            if (!m_indexing)
                return false;
            // things that make no sense
            if (m_verticesPerSupplementary==0 || m_verticesPerSupplementary>m_verticesForFirst)
                return false;
            // there needs to be at least one vertex to reference (it also needs to be formatted)
            const auto& positionBase = base_t::m_positionView.composed;
            const auto vertexCount = positionBase.getElementCount();
            if (vertexCount==0 || !positionBase.isFormatted())
                return false;
            if (m_normalView && m_normalView.getElementCount()<vertexCount)
                return false;
            // the variable length vectors must be filled with valid views
            for (const auto& pair : m_jointWeightViews)
            if (!pair || !pair.weights.getElementCount()<vertexCount)
                return false;
            for (const auto& view : m_auxAttributeViews)
            if (!view)
                return false;
            return true;
        }

        //
        constexpr static inline EPrimitiveType PrimitiveType = EPrimitiveType::Polygon;
        inline EPrimitiveType getPrimitiveType() const override final {return PrimitiveType;}

        //
        inline const IGeometryBase::SAABBStorage& getAABB() const override final {return base_t::m_positionView.encodedDataRange;}

        //
        inline uint64_t getVertexReferenceCount() const {return base_t::getIndexView() ? base_t::getIndexCount():base_t::m_positionView.getElementCount();}

        //
        inline uint64_t getPrimitiveCount() const override final
        {
            if (!m_indexing)
                return 0;
            const auto vertexReferenceCount = getVertexReferenceCount();
            const auto verticesForFirst = m_indexing->degree();
            if (vertexReferenceCount<verticesForFirst)
                return 0;
            return (vertexReferenceCount-verticesForFirst)/(verticesForFirst-m_indexing->reuseCount());
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


        // Does not set the `transform` or `geometryFlags` fields, because it doesn't care about it.
        // Also won't set second set of vertex data, opacity mipmaps, etc.
        inline BLASTriangles exportForBLAS() const
        {
            BLASTriangles retval = {};
            // must be a triangle list 
            if (m_verticesForFirst==3 && m_verticesPerSupplementary==3)
            {
                auto indexType = EIT_UNKNOWN;
                // disallowed index format
                if (base_t::m_indexView)
                {
                    switch (base_t::m_indexView.format)
                    {
                        case EF_R16_UINT:
                            indexType = EIT_16BIT;
                            break;
                        case EF_R32_UINT: [[fallthrough]];
                            indexType = EIT_32BIT;
                            break;
                        default:
                            break;
                    }
                    if (indexType==EIT_UNKNOWN)
                        return retval;
                }
                retval.vertexData[0] = base_t::m_positionView;
                retval.indexData = base_t::m_indexView;
                retval.maxVertex = base_t::m_positionView.getElementCount();
                retval.vertexStride = base_t::m_positionView.getStride();
                retval.vertexFormat = base_t::m_positionView.format;
                retval.indexType = indexType;
            }
            return retval;
        }

    protected:
        virtual ~IPolygonGeometry() = default;

        //
        inline bool setIndexView(SDataView&& view)
        {
            if (!view || view.isFormattedScalarInteger())
            {
                m_indexView = std::move(view);
                return true;
            }
            return false;
        }

        //
        core::vector<SJointWeight> m_jointWeightViews = {};
        //
        core::vector<SDataView> m_auxAttributeViews = {};
        //
        SDataView m_normalView = {};
        //
        uint32_t m_jointCount = 0;
};

}
#endif