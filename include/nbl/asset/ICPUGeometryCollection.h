// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_GEOMETRY_COLLECTION_H_INCLUDED_
#define _NBL_ASSET_I_CPU_GEOMETRY_COLLECTION_H_INCLUDED_


#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/IGeometryCollection.h"


namespace nbl::asset
{
//
class NBL_API2 ICPUGeometryCollection : public IAsset, public IGeometryCollection<ICPUBuffer>
{
        using base_t = IGeometryCollection<ICPUBuffer>;

    public:
        inline ICPUGeometryCollection() = default;
        
        constexpr static inline auto AssetType = ET_GEOMETRY_COLLECTION;
        inline E_TYPE getAssetType() const override {return AssetType;}

        //
        inline bool valid() const //override
        {
            for (const auto& ref : m_geometries)
            if (!ref.geometry->valid())
                return false;
            return true;
        }

        inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth=~0u) const
        {
            const auto nextDepth = _depth ? (_depth-1):0;
            auto retval = core::smart_refctd_ptr<ICPUGeometryCollection>();
            retval->m_aabb = m_aabb;
            retval->m_inverseBindPoseView = m_inverseBindPoseView.clone(nextDepth);
            retval->m_jointAABBView = m_jointAABBView.clone(nextDepth);
            retval->m_geometries.reserve(m_geometries.size());
            for (const auto& in : m_geometries)
            {
                auto& out = retval->m_geometries.emplace_back();
                out.transform = in.transform;
                out.geometry = core::smart_refctd_ptr_static_cast<IGeometry<ICPUBuffer>>(in.geometry->clone(nextDepth));
                out.jointRedirectView = in.jointRedirectView.clone(nextDepth);
            }
            return retval;
        }

        // 
        inline bool setAABB(const IGeometryBase::SAABBStorage& aabb)
        {
            if (isMutable())
            {
                m_aabb = aabb;
                return true;
            }
            return false;
        }

        //
        inline core::vector<SGeometryReference>* getGeometries()
        {
            if (isMutable())
                return &m_geometries;
            return nullptr;
        }

        //
        inline bool setSkin(SDataView&& inverseBindPoseView, SDataView&& jointAABBView)
        {
            if (isMutable())
                return setSkin(std::move(inverseBindPoseView),std::move(jointAABBView));
            return false;
        }

        //
        template<typename Iterator>// requires std::is_same_v<decltype(*declval<Iterator>()),decltype(ICPUBottomLevelAccelerationStructure::Triangles&)>
        inline Iterator exportForBLAS(Iterator out, uint32_t* pWrittenOrdinals=nullptr) const
        {
            return exportForBLAS(std::forward<Iterator>(out),[this, &pWrittenOrdinals](hlsl::float32_t3x4& lhs, const hlsl::float32_t3x4& rhs)->void
                {
                    lhs = rhs;
                    if (pWrittenOrdinals)
                        *(pWrittenOrdinals++) = (ptrdiff_t(&rhs)-offsetof(SGeometryReference,transform)-ptrdiff_t(base_t::m_geometries.data()))/sizeof(SGeometryReference);
                }
            );
        }

    protected:
        //
        inline void visitDependents_impl(std::function<bool(const IAsset*)> visit) const override
        {
            auto nonNullOnly = [&visit](const IAsset* dep)->bool
            {
                if (dep)
                    return visit(dep);
                return true;
            };
            if (!nonNullOnly(m_inverseBindPoseView.src.buffer.get())) return;
            if (!nonNullOnly(m_jointAABBView.src.buffer.get())) return;
            for (const auto& ref : m_geometries)
            {
                const auto* geometry = static_cast<const IAsset*>(ref.geometry.get());
                if (!nonNullOnly(geometry)) return;
            }
        }
};

}
#endif