// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_GEOMETRY_COLLECTION_H_INCLUDED_
#define _NBL_ASSET_I_GEOMETRY_COLLECTION_H_INCLUDED_


#include "nbl/asset/IPolygonGeometry.h"


namespace nbl::asset
{
// Collection of geometries of the same type (but e.g. with different formats or transforms)
template<class BufferType>
class NBL_API2 IGeometryCollection : public virtual core::IReferenceCounted
{
    public:
        using SDataView = IGeometry<BufferType>::SDataView;

        //
        inline const auto& getAABB() const {return m_aabb;}

        //
        struct SGeometryReference final
        {
            inline operator bool() const
            {
                if (!geometry)
                    return false;
                if (jointRedirectView.src)
                {
                    if (!jointRedirectView.composed.isFormattedScalarInteger())
                        return false;
                    if (jointRedirectView.getElementCount()<geometry->getJointCount())
                        return false;
                }
                return true;
            }

            inline bool hasTransform() const {return !core::isnan(transform[0][0]);}

            hlsl::float32_t3x4 transform = hlsl::float32_t3x4(std::numeric_limits<float>::quiet_NaN());
            core::smart_refctd_ptr<IGeometry<BufferType>> geometry = {};
            // The geometry may be using a smaller set of joint/bone IDs which need to be remapped to a larger or common skeleton
            // Ignored if this geometry collection is not skinned or the `geometry` doesn't have a weight view.
            // If not provided, its treated as-if an iota {0,1,2,...} view was provided
            SDataView jointRedirectView = {};
        };
        inline const core::vector<SGeometryReference>& getGeometries() const {return m_geometries;}
        
        // This is for the whole geometry collection, geometries remap to those.
        // Each element is a row of an affine transformation matrix
        inline uint32_t getJointCount() const {return m_inverseBindPoseView.getElementCount()/3;}

        //
        inline bool isSkinned() const {return getJointCount()>0;}
        // View of matrices being the inverse bind pose
        inline const SDataView& getInverseBindPoseView() const {return m_inverseBindPoseView;}
        
        
        //
        class IBLASExporter
        {
            protected:
                using BLASTriangles = IBottomLevelAccelerationStructure::Triangles<std::remove_const_t<BufferType>>;

                inline IBLASExporter(const core::vector<SGeometryReference>& _geoms) : m_geoms(_geoms) {}
                virtual void setTransform(BLASTriangles& out, const uint32_t geomIndex) = 0;

                const core::vector<SGeometryReference>& m_geoms;

            public:
                template<typename TriIter, typename PrimCountIter> // requires (std::is_same_v<decltype(*declval<TriIter>()),decltype(BLASTriangles&)> && PrimCountIter is integral && OrdinalIter is also)
                inline TriIter operator()(TriIter outIt, PrimCountIter outPrimCount, uint32_t* pWrittenOrdinals=nullptr)
                {
                    for (const auto& ref : m_geoms)
                    {
                        // not a polygon geometry
                        const auto* geo = ref.geometry.get();
                        if (geo->getPrimitiveType()!=IGeometryBase::EPrimitiveType::Polygon)
                            continue;
                        const auto ordinal = std::distance(m_geoms.data(),&ref);
                        const auto* polyGeo = static_cast<const IPolygonGeometry<BufferType>*>(geo);
                        *outIt = polyGeo->exportForBLAS();
                        if (outIt->vertexData[0])
                        {
                            if (pWrittenOrdinals)
                                *(pWrittenOrdinals++) = ordinal;
                            *(outPrimCount++) = polyGeo->getPrimitiveCount();
                            if (ref.hasTransform())
                                setTransform(*outIt,ordinal);
                            outIt++;
                        }
                    }
                    return outIt;
                }
        };

    protected:
        virtual ~IGeometryCollection() = default;

        //
        inline core::vector<SGeometryReference>& getGeometries() {return m_geometries;}

        // returns whether skinning was enabled or disabled
        inline bool setSkin(SDataView&& inverseBindPoseView, SDataView&& jointAABBView)
        {
            // disable skinning
            m_inverseBindPoseView = {};
            m_jointAABBView = {};
            // need a format with one row per element
            const auto ibpFormat = inverseBindPoseView.composed.format;
            if (!inverseBindPoseView || !inverseBindPoseView.composed.isFormatted() || getFormatChannelCount(ibpFormat)!=4)
                return false;
            const auto matrixRowCount = inverseBindPoseView.getElementCount();
            // and 3 elements per matrix
            if (matrixRowCount==0 || (matrixRowCount%3)!=0)
                return false;
            const auto jointCount = matrixRowCount/3;
            // ok now check the AABB stuff
            if (jointAABBView)
            {
                // needs to be formatted
                if (!jointAABBView.composed.isFormatted())
                   return false;
                // each element is a AABB vertex, so need 2 per joint
                if (jointAABBView.getElementCount()!=jointCount*2)
                    return false;
            }
            m_inverseBindPoseView = std::move(inverseBindPoseView);
            m_jointAABBView = std::move(jointAABBView);
            return true;
        }


        // For the entire collection, as always it should NOT include any geometry which is affected by a joint.
        hlsl::shapes::AABB<3,hlsl::float64_t> m_aabb;
        SDataView m_inverseBindPoseView = {};
        // The AABBs gathered from all geometries (optional) and are in "bone-space" so there's no need for OBB option,
        // joint influence is usually aligned to the covariance matrix of geometry affected by it.
        SDataView m_jointAABBView = {};
        //
        core::vector<SGeometryReference> m_geometries;
};

}
#endif