// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_SCENE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_SCENE_H_INCLUDED_


#include "nbl/asset/IScene.h"
// TODO: change to true IR later
#include "nbl/asset/material_compiler3/CFrontendIR.h"


namespace nbl::asset
{
// 
class NBL_API2 ICPUScene : public IAsset, public IScene
{
        using base_t = IScene;

    public:
        inline ICPUScene() = default;

        constexpr static inline auto AssetType = ET_SCENE;
        inline E_TYPE getAssetType() const override { return AssetType; }

        inline bool valid() const override
        {
            return true;
        }

        inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth=~0u) const
        {
            const auto nextDepth = _depth ? (_depth-1):0;
            auto retval = core::smart_refctd_ptr<ICPUScene>();
            return retval;
        }

    protected:
        //
        inline void visitDependents_impl(std::function<bool(const IAsset*)> visit) const override
        {
        }


        // suggested contents:
        // - morph target list
        // - material table
        // - instance list (morph target, keyframed transforms, material table indexings, FUTURE: reference skeleton)
        // - area light list (OBB decompositions, material table indexings)
        // - envlight data
};
}

#endif