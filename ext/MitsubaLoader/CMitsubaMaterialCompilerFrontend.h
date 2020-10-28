#ifndef __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__
#define __C_MITSUBA_MATERIAL_COMPILER_FRONTEND_H_INCLUDED__

#include "../../ext/MitsubaLoader/CElementBSDF.h"
#include <irr/asset/material_compiler/IR.h>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{
    struct SContext;

class CMitsubaMaterialCompilerFrontend
{
    using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>, core::smart_refctd_ptr<asset::ICPUSampler>, float>;
    using deriv_map_cache_key_t = std::pair<core::smart_refctd_ptr<asset::ICPUImage>, core::smart_refctd_ptr<asset::ICPUSampler>>;
    using deriv_map_cache_val_t = core::smart_refctd_ptr<asset::ICPUImageView>;
    struct deriv_map_cache_hash_t
    {
        std::size_t operator()(const deriv_map_cache_key_t& key) const
        {
            return deriv_map_cache_key_t::first_type::hash{}(key.first) ^ deriv_map_cache_key_t::second_type::hash{}(key.second);
        }
    };
    using deriv_map_cache_t = std::unordered_map<deriv_map_cache_key_t, deriv_map_cache_val_t, deriv_map_cache_hash_t>;

    const SContext* m_loaderContext;
    deriv_map_cache_t m_derivMapCache;

    deriv_map_cache_val_t getDerivMap(asset::ICPUImage* _heightMap, asset::ICPUSampler* _smplr);

    tex_ass_type getTexture(const CElementTexture* _element) const;

public:
    CMitsubaMaterialCompilerFrontend(const SContext* _ctx) : m_loaderContext(_ctx) {}

    asset::material_compiler::IR::INode* compileToIRTree(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf);
};

}}}

#endif