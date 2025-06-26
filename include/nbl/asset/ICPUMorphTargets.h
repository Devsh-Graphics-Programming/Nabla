// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_MORPH_TARGETS_H_INCLUDED_
#define _NBL_ASSET_I_CPU_MORPH_TARGETS_H_INCLUDED_


#include "nbl/asset/IAsset.h"
#include "nbl/asset/IMorphTargets.h"


namespace nbl::asset
{
//
class NBL_API2 ICPUMorphTargets : public IAsset, public IMorphTargets<ICPUGeometryCollection>
{
        using base_t = IMorphTargets<ICPUGeometryCollection>;

    public:
        inline ICPUMorphTargets() = default;
        
        constexpr static inline auto AssetType = ET_MORPH_TARGETS;
        inline E_TYPE getAssetType() const override {return AssetType;}

        //
        inline bool valid() const //override
        {
            for (const auto& target : m_targets)
            if (!target || !target.geoCollection->valid())
                return false;
            return true;
        }

        inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth=~0u) const
        {
            const auto nextDepth = _depth ? (_depth-1):0;
            auto retval = core::smart_refctd_ptr<ICPUMorphTargets>();
            retval->m_targets.reserve(m_targets.size());
            for (const auto& in : m_targets)
            {
                auto& out = retval->m_targets.emplace_back();
                out.geoCollection = core::smart_refctd_ptr_static_cast<ICPUGeometryCollection>(in.geoCollection->clone(nextDepth));
                out.jointRedirectView = in.jointRedirectView.clone(nextDepth);
            }
            return retval;
        }

        //
        inline core::vector<base_t::STarget>* getTargets()
        {
            if (isMutable())
                return &m_targets;
            return nullptr;
        }

    protected:
        //
        inline void visitDependents_impl(std::function<bool(const IAsset*)> visit) const //override
        {
            auto nonNullOnly = [&visit](const IAsset* dep)->bool
            {
                if (dep)
                    return visit(dep);
                return true;
            };
            for (const auto& ref : m_targets)
            if (!nonNullOnly(ref.geoCollection.get())) return;
        }
};

}
#endif