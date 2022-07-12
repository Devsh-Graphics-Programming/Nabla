// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MITSUBA_LOADER_C_MATERIAL_COMPILER_FRONTEND_H_INCLUDED_
#define _NBL_EXT_MITSUBA_LOADER_C_MATERIAL_COMPILER_FRONTEND_H_INCLUDED_

#include "nbl/core/Types.h"

#include "nbl/asset/material_compiler/IR.h"

#include "nbl/ext/MitsubaLoader/CElementBSDF.h"

namespace nbl::ext::MitsubaLoader
{

struct SContext;

class CMaterialCompilerFrontend final
{
    public:
        using node_handle_t = asset::material_compiler::IR::node_handle_t;
        enum E_IMAGE_VIEW_SEMANTIC : uint8_t
        {
            EIVS_IDENTITIY,
            EIVS_BLEND_WEIGHT,
            EIVS_NORMAL_MAP,
            EIVS_BUMP_MAP,
            EIVS_COUNT
        };

        // TODO: embed hash val in the element for speed
        struct MerkleTree
        {
            const CElementBSDF* bsdf;
            bool frontface;

            struct hash
            {
                std::size_t operator()(const MerkleTree& node) const;
            };
            struct equal_to
            {
                bool operator()(const MerkleTree& lhs, const MerkleTree& rhs) const;
            };
        };
        using HashCons = core::unordered_map<MerkleTree,node_handle_t,MerkleTree::hash,MerkleTree::equal_to>;

        struct front_and_back_t
        {
            node_handle_t front;
            node_handle_t back;
        };
        static front_and_back_t compileToIRTree(SContext& ctx, const CElementBSDF* _root);

    private:
        static bool unwindTwosided(const CElementBSDF* &bsdf)
        {
            const auto orig_bsdf = bsdf;
            while (bsdf->type==CElementBSDF::TWO_SIDED)
            {
                // sanity checks
                static_assert(bsdf->twosided.MaxChildCount == 1);
                assert(bsdf->meta_common.childCount==1);
                assert(bsdf->twosided.childCount==1);

                bsdf = bsdf->meta_common.bsdf[0];
            }
            return bsdf!=orig_bsdf;
        }
        static node_handle_t createIRNode(SContext& ctx, const CElementBSDF* _bsdf, const bool frontface);

        using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>, core::smart_refctd_ptr<asset::ICPUSampler>, float>;
        static tex_ass_type getTexture(const SContext& _loaderContext, const CElementTexture* _element, const E_IMAGE_VIEW_SEMANTIC semantic=EIVS_IDENTITIY);
        static tex_ass_type getErrorTexture(const SContext& _loaderContext);
};

}

#endif