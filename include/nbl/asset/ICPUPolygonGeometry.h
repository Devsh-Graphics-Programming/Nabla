// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_POLYGON_GEOMETRY_H_INCLUDED_
#define _NBL_ASSET_I_CPU_POLYGON_GEOMETRY_H_INCLUDED_


#include "nbl/asset/IPolygonGeometry.h"


namespace nbl::asset
{
//
class NBL_API2 ICPUPolygonGeometry : public IAsset, public IPolygonGeometry<ICPUBuffer>
{
        using base_t = IPolygonGeometry<ICPUBuffer>;

    public:
        inline ICPUPolygonGeometry() = default;
        
        constexpr static inline auto AssetType = ET_GEOMETRY;
        inline E_TYPE getAssetType() const override {return AssetType;}

        inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth=~0u) const
        {
            const auto nextDepth = _depth ? (_depth-1):0;
            auto retval = core::smart_refctd_ptr<ICPUPolygonGeometry>();
            retval->m_positionView = m_positionView.clone(nextDepth);
            retval->m_jointOBBView = m_jointOBBView.clone(nextDepth);
            retval->m_indexView = m_indexView.clone(nextDepth);
            retval->m_jointWeightViews.reserve(m_jointWeightViews.size());
            for (const auto& pair : m_jointWeightViews)
                retval->m_jointWeightViews.push_back({
                    .indices = pair.indices.clone(nextDepth),
                    .weights = pair.weights.clone(nextDepth)
                });
            retval->m_auxAttributeViews.reserve(m_auxAttributeViews.size());
            for (const auto& view : m_auxAttributeViews)
                retval->m_auxAttributeViews.push_back(view.clone(nextDepth));
            retval->m_normalView = m_normalView.clone(nextDepth);
            retval->m_jointCount = m_jointCount;
            retval->m_verticesForFirst = m_verticesForFirst;
            retval->m_verticesPerSupplementary = m_verticesPerSupplementary;
            return retval;
        }

        // TODO: remove after https://github.com/Devsh-Graphics-Programming/Nabla/pull/871 merge
        inline size_t getDependantCount() const override
        {
           size_t count = 0;
           visitDependents([&current](const IAsset* dep)->bool
               {
                   count++;
                   return true;
               }
           );
           return count;
        }

        // needs to be hidden because of mutability checking
        inline bool setPositionView(SDataView&& view)
        {
            if (isMutable() && view.composed.isFormatted())
                return base_t::setPositionView(std::move(view));
            return false;
        }

        // 
        inline bool setJointOBBView(SDataView&& view)
        {
            if (isMutable())
                return base_t::setJointOBBView(std::move(view));
            return false;
        }

        // Needs to be hidden because ICPU base class shall check mutability
        inline bool setIndexView(SDataView&& view)
        {
            if (isMutable())
                return base_t::setIndexView(std::move(view));
            return false;
        }

        //
        inline bool setVerticesForFirst(const uint16_t count)
        {
            if (isMutable())
            {
                m_verticesForFirst = count;
                return true;
            }
            return false;
        }

        //
        inline bool setVerticesPerSupplementary(const uint16_t count)
        {
            if (isMutable())
            {
                m_verticesPerSupplementary = count;
                return true;
            }
            return false;
        }

        //
        inline void setNormalView(SDataView&& view)
        {
            if (isMutable())
            {
                m_normalView = std::move(view);
                return true;
            }
            return false;
        }

        //
        inline bool setJointCount(const uint32_t count)
        {
            if (isMutable())
            {
                m_jointCount = count;
                return true;
            }
            return false;
        }

        //
        inline const core::vector<SJointWeight>* getJointWeightViews()
        {
            if (isMutable())
                return m_jointWeightViews;
            return nullptr;
        }

        //
        inline const core::vector<SJointWeight>* getAuxAttributeViews()
        {
            if (isMutable())
                return m_auxAttributeViews;
            return nullptr;
        }

    protected:
        //
        inline void visitDependents(std::function<bool(const IAsset*)>& visit) const //override
        {
            if (!visit(m_positionView.src.buffer.get())) return;
            if (!visit(m_jointOBBView.src.buffer.get())) return;
            if (!visit(m_indexView.src.buffer.get())) return;
            for (const auto& pair : m_jointWeightViews)
            {
                if (!visit(pair.indices.src.buffer.get())) return;
                if (!visit(pair.weights.src.buffer.get())) return;
            }
            for (const auto& view : m_auxAttributeViews)
                if (!visit(view.src.buffer.get())) return;
            if (!visit(m_normalView.src.buffer.get())) return;
        }
        // TODO: remove after https://github.com/Devsh-Graphics-Programming/Nabla/pull/871 merge
        inline IAsset* getDependant_impl(const size_t ix) override
        {
           const IAsset* retval = nullptr;
           size_t current = 0;
           visitDependents([&current](const IAsset* dep)->bool
               {
                   retval = dep;
                   return ix<current++;
               }
           );
           return const_cast<IAsset*>(retval);
        }
};

}
#endif