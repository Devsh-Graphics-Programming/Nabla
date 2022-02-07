// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_LOADER_H_INCLUDED__
#define __NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_LOADER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
#include "nbl/asset/interchange/IAssetLoader.h"

namespace nbl::asset
{
class IRenderpassIndependentPipelineLoader : public IAssetLoader
{
public:
    virtual void initialize() override;

protected:
    IAssetManager* m_assetMgr;
    core::smart_refctd_dynamic_array<IRenderpassIndependentPipelineMetadata::ShaderInputSemantic> m_basicViewParamsSemantics;

    inline IRenderpassIndependentPipelineLoader(IAssetManager* _am)
        : m_assetMgr(_am) {}
    virtual ~IRenderpassIndependentPipelineLoader() = 0;

    // samplers
    static inline std::string genSamplerCacheKey(const asset::ICPUSampler::SParams& params)
    {
        // TODO: Change the HASH, ACTUALLY BUILD IT OUT OF ALL THE PARAMETERS, THERE CANNOT BE ANY COLLISIONS!
        const std::size_t hash = std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(&params), sizeof(params)));
        return "nbl/builtin/sampler/" + std::to_string(hash);
    }
    static inline core::smart_refctd_ptr<ICPUSampler> getSampler(asset::ICPUSampler::SParams&& params, const IAssetLoader::SAssetLoadContext& context, IAssetLoaderOverride* _override)
    {
        const auto cacheKey = genSamplerCacheKey(params);

        auto found = _override->findDefaultAsset<ICPUSampler>(cacheKey, context, 0u).first;  // cached builtins have level 0
        if(found)
            return found;

        auto sampler = core::make_smart_refctd_ptr<ICPUSampler>(std::move(params));
        SAssetBundle samplerBundle = SAssetBundle(nullptr, {sampler});
        _override->insertAssetIntoCache(samplerBundle, cacheKey, context, 0u);  // cached builtins have level 0
        return sampler;
    }

private:
};

}

#endif
