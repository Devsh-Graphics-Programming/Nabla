// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_POLYGON_GEOMETRY_H_INCLUDED_
#define _NBL_ASSET_I_CPU_POLYGON_GEOMETRY_H_INCLUDED_


#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/IPolygonGeometry.h"


namespace nbl::asset
{
//
class NBL_API2 ICPUPolygonGeometry final : public IAsset, public IPolygonGeometry<ICPUBuffer>
{
        using base_t = IPolygonGeometry<ICPUBuffer>;

    protected:
        using SDataView = base_t::SDataView;

    public:
        inline ICPUPolygonGeometry() = default;
        
        constexpr static inline auto AssetType = ET_GEOMETRY;
        inline E_TYPE getAssetType() const override {return AssetType;}

        inline bool valid() const override {return base_t::valid();}

        inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth=~0u) const override
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
            return retval;
        }

        // TODO: remove after https://github.com/Devsh-Graphics-Programming/Nabla/pull/871 merge
        inline size_t getDependantCount() const override
        {
           size_t count = 0;
           visitDependents([&count](const IAsset* dep)->bool
               {
                   count++;
                   return true;
               }
           );
           return count;
        }

        //
        inline bool setPositionView(SDataView&& view)
        {
            if (isMutable() && (!view || view.composed.isFormatted()))
            {
                m_positionView = std::move(view);
                return true;
            }
            return false;
        }

        // 
        inline bool setJointOBBView(SDataView&& view)
        {
            if (isMutable())
                return base_t::setJointOBBView(std::move(view));
            return false;
        }

        //
        inline bool setIndexView(SDataView&& view)
        {
            if (isMutable())
                return base_t::setIndexView(std::move(view));
            return false;
        }

        //
        inline bool setIndexing(const IIndexingCallback* indexing)
        {
            if (isMutable())
            {
                m_indexing = indexing;
                return true;
            }
            return false;
        }
        //
        inline bool setNormalView(SDataView&& view)
        {
            if (isMutable() && (!view || view.composed.getStride()>0))
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
        inline core::vector<SJointWeight>* getJointWeightViews()
        {
            if (isMutable())
                return &m_jointWeightViews;
            return nullptr;
        }

        //
        inline core::vector<SDataView>* getAuxAttributeViews()
        {
            if (isMutable())
                return &m_auxAttributeViews;
            return nullptr;
        }

        // We don't care about primitive restart, you need to check for it yourself.
        // Unlike OpenGL and other APIs we don't adjust the Primitive ID because that breaks parallel processing.
        // So a triangle strip `{ 0 1 2 3 RESTART 2 3 4 5 }` means 7 primitives, of which 3 are invalid (contain the restart index) 
        template<typename Out> requires hlsl::concepts::UnsignedIntegralScalar<Out>
        inline bool getPrimitiveIndices(Out* out, const uint32_t beginPrimitive, const uint32_t endPrimitive) const
        {
            if (!m_indexing)
                return false;
            IIndexingCallback::SContext ctx = {
                .indexBuffer = m_indexView.getPointer(),
                .indexSize = getTexelOrBlockBytesize(m_indexView.composed.format),
                .beginPrimitive = beginPrimitive,
                .endPrimitive = endPrimitive,
                .out = out
            };
            m_indexing->operator()(ctx);
            return true;
        }

        //
        template<typename Out>
        inline bool getPrimitiveIndices(Out* out, const uint32_t primitiveID) const
        {
            return getPrimitiveIndices(out,primitiveID,primitiveID+1);
        }

    protected:
        //
        inline void visitDependents(std::function<bool(const IAsset*)> visit) const //override
        {
            auto nonNullOnly = [&visit](const IAsset* dep)->bool
            {
                if (dep)
                    return visit(dep);
                return true;
            };
            if (!nonNullOnly(m_positionView.src.buffer.get())) return;
            if (!nonNullOnly(m_jointOBBView.src.buffer.get())) return;
            if (!nonNullOnly(m_indexView.src.buffer.get())) return;
            for (const auto& pair : m_jointWeightViews)
            {
                if (!nonNullOnly(pair.indices.src.buffer.get())) return;
                if (!nonNullOnly(pair.weights.src.buffer.get())) return;
            }
            for (const auto& view : m_auxAttributeViews)
                if (!nonNullOnly(view.src.buffer.get())) return;
            if (!nonNullOnly(m_normalView.src.buffer.get())) return;
        }
        // TODO: remove after https://github.com/Devsh-Graphics-Programming/Nabla/pull/871 merge
        inline IAsset* getDependant_impl(const size_t ix) override
        {
           const IAsset* retval = nullptr;
           size_t current = 0;
           visitDependents([&](const IAsset* dep)->bool
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