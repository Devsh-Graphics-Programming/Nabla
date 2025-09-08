// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_POLYGON_GEOMETRY_H_INCLUDED_
#define _NBL_ASSET_I_POLYGON_GEOMETRY_H_INCLUDED_


#include "nbl/asset/IGeometry.h"
#include "nbl/asset/RasterizationStates.h"
#include "nbl/asset/IAccelerationStructure.h"

#include <span>

namespace nbl::asset
{
//
class IPolygonGeometryBase : public virtual core::IReferenceCounted
{
    public:
        //
        class IIndexingCallback
        {
            public:
                // how many vertices per polygon
                inline uint8_t degree() const
                {
                    const auto retval = degree_impl();
                    assert(retval>0);
                    return retval;
                }
                // at which we consume indices for each new polygon
                inline uint8_t rate() const
                {
                    const auto retval = rate_impl();
                    assert(retval>0 && retval<=degree());
                    return retval;
                }

                template<typename OutT> requires (sizeof(OutT)<8 && hlsl::concepts::UnsignedIntegralScalar<OutT>)
                struct SContext final
                {
                    // `indexOfIndex` is somewhat of a baseIndex
                    template<typename Range>
                    inline void streamOut(const uint32_t indexOfIndex, const Range& permutation)
                    {
                        auto& typedOut = reinterpret_cast<OutT*&>(out);
                        if (indexBuffer)                  
                        switch (indexSize)
                        {
                            case 1:
                                for (const auto relIx : permutation)
                                    *(typedOut++) = reinterpret_cast<const uint8_t*>(indexBuffer)[indexOfIndex+relIx];
                                break;
                            case 2:
                                for (const auto relIx : permutation)
                                    *(typedOut++) = reinterpret_cast<const uint16_t*>(indexBuffer)[indexOfIndex+relIx];
                                break;
                            case 4:
                                for (const auto relIx : permutation)
                                    *(typedOut++) = reinterpret_cast<const uint32_t*>(indexBuffer)[indexOfIndex+relIx];
                                break;
                            default:
                                assert(false);
                                break;
                        }
                        else
                        for (const auto relIx : permutation)
                            *(typedOut++) = indexOfIndex+relIx;
                    }

                    // always the base pointer, doesn't get advanced
                    const void* const indexBuffer;
                    const uint64_t indexSize : 3;
                    const uint64_t beginPrimitive : 30;
                    const uint64_t endPrimitive : 31;
                    void* out;
                };
                // could have been a static if not virtual
                virtual void operator()(SContext<uint8_t>& ctx) const = 0;
                virtual void operator()(SContext<uint16_t>& ctx) const = 0;
                virtual void operator()(SContext<uint32_t>& ctx) const = 0;

                // default is unknown
                virtual inline E_PRIMITIVE_TOPOLOGY knownTopology() const {return static_cast<E_PRIMITIVE_TOPOLOGY>(~0);}

            protected:
                virtual uint8_t degree_impl() const = 0;
                virtual uint8_t rate_impl() const = 0;
        };
        //
        NBL_API2 static IIndexingCallback* PointList();
        NBL_API2 static IIndexingCallback* LineList();
        NBL_API2 static IIndexingCallback* TriangleList();
        NBL_API2 static IIndexingCallback* QuadList();
        // TODO: Adjacency, Patch, etc.
        NBL_API2 static IIndexingCallback* TriangleStrip();
        NBL_API2 static IIndexingCallback* TriangleFan();

        // This should be a pointer to a stateless singleton (think of it more like a dynamic enum/template than anything else)
        inline const IIndexingCallback* getIndexingCallback() const {return m_indexing;}

    protected:
        virtual inline ~IPolygonGeometryBase() = default;

        // indexing callback cannot be cleared
        inline bool setIndexingCallback(IIndexingCallback* indexing)
        {
            if (!indexing)
                return false;
            const auto deg = m_indexing->degree();
            if (deg==0 || m_indexing->rate()==0 || m_indexing->rate()>deg)
                return false;
            m_indexing = indexing;
            return true;
        }

        //
        const IIndexingCallback* m_indexing = nullptr;
};

// Don't want to overengineer, support for variable vertex count (order) polgyon meshes is not encouraged or planned.
// If you want different polygon types in same model, bucket the polgyons into separate geometries and reunite in a single collection.
template<class BufferType>
class IPolygonGeometry : public IIndexableGeometry<BufferType>, public IPolygonGeometryBase
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
            // there needs to be at least one vertex to reference (it also needs to be formatted)
            const auto& positionBase = base_t::m_positionView.composed;
            const auto vertexCount = base_t::m_positionView.getElementCount();
            if (vertexCount==0 || !positionBase.isFormatted())
                return false;
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

        //
        constexpr static inline EPrimitiveType PrimitiveType = EPrimitiveType::Polygon;
        inline EPrimitiveType getPrimitiveType() const override final {return PrimitiveType;}

        //
        inline const IGeometryBase::SAABBStorage& getAABBStorage() const override final {return m_aabb;}

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
            return (vertexReferenceCount-verticesForFirst)/m_indexing->rate()+1;
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
            inline operator bool() const {return indices && isIntegerFormat(indices.composed.format) && weights && weights.composed.isFormatted() && indices.getElementCount()==weights.getElementCount();}

            SDataView indices;
            // Assumption is that only non-zero weights are present, which is why the joints are indexed (sparseness)
            // Zero weights are acceptable but only if they form a contiguous block including the very last component of the very last weight view
            SDataView weights;
        };
        // It's a vector in case you need more than 4 influences per bone
        inline const core::vector<SJointWeight>& getJointWeightViews() const {return m_jointWeightViews;}

        // For User defined semantics
        inline const core::vector<SDataView>& getAuxAttributeViews() const {return m_auxAttributeViews;}

        inline E_INDEX_TYPE getIndexType() const
        {
            auto indexType = EIT_UNKNOWN;
            // disallowed index format
            if (base_t::m_indexView)
            {
                switch (base_t::m_indexView.composed.format)
                {
                    case EF_R16_UINT:
                        indexType = EIT_16BIT;
                        break;
                    case EF_R32_UINT:
                        indexType = EIT_32BIT;
                        break;
                    default:
                        break;
                }
            }
            return indexType;
        }

        // Does not set the `transform` or `geometryFlags` fields, because it doesn't care about it.
        // Also won't set second set of vertex data, opacity mipmaps, etc.
        inline BLASTriangles exportForBLAS() const
        {
            BLASTriangles retval = {};
            // must be a triangle list, but don't want to compare pointers
            if (m_indexing && m_indexing->knownTopology()==EPT_TRIANGLE_LIST)// && m_indexing->degree() == TriangleList()->degree() && m_indexing->rate() == TriangleList->rate())
            {
                retval.vertexData[0] = base_t::m_positionView.src;
                retval.indexData = base_t::m_indexView.src;
                retval.maxVertex = base_t::m_positionView.getElementCount() - 1;
                retval.vertexStride = base_t::m_positionView.composed.getStride();
                retval.vertexFormat = base_t::m_positionView.composed.format;
                retval.indexType = getIndexType();
            }
            return retval;
        }

    protected:
        virtual ~IPolygonGeometry() = default;

        // 64bit indices are just too much to deal with in all the other code
        // Also if you have more than 2G vertex references in a single geometry there's something wrong with your architecture
        // Note that this still allows 6GB vertex attribute streams (assuming at least 3 bytes for a postion)
        inline bool setIndexView(SDataView&& view)
        {
            if (view)
            {
                const auto format = view.composed.format;
                if (!view.composed.isFormattedScalarInteger() || format == EF_R64_UINT || format == EF_R64_SINT)
                    return false;
                if (view.getElementCount()>(1u<<31))
                    return false;
            }
            base_t::m_indexView = std::move(view);
            return true;
        }

        //
        IGeometryBase::SAABBStorage m_aabb = {};
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