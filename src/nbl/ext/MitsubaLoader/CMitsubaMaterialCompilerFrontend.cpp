// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/MitsubaLoader/CMitsubaMaterialCompilerFrontend.h"
#include "nbl/ext/MitsubaLoader/SContext.h"

namespace nbl::ext::MitsubaLoader
{

auto CMitsubaMaterialCompilerFrontend::getTexture(const ext::MitsubaLoader::SContext* _loaderContext, const CElementTexture* _element, const E_IMAGE_VIEW_SEMANTIC semantic) -> tex_ass_type
{
    // first unwind the texture Scales
    float scale = 1.f;
    while (_element && _element->type==CElementTexture::SCALE)
    {
        scale *= _element->scale.scale;
        _element = _element->scale.texture;
    }
    _NBL_DEBUG_BREAK_IF(_element && _element->type!=CElementTexture::BITMAP);
    if (!_element)
    {
        os::Printer::log("[ERROR] Could Not Find Texture, dangling reference after scale unroll, substituting 2x2 Magenta Checkerboard Error Texture.", ELL_ERROR);
        return getErrorTexture(_loaderContext);
    }

    asset::IAsset::E_TYPE types[2]{ asset::IAsset::ET_IMAGE_VIEW, asset::IAsset::ET_TERMINATING_ZERO };
    const auto key = _loaderContext->imageViewCacheKey(_element->bitmap,semantic);
    auto viewBundle = _loaderContext->override_->findCachedAsset(key,types,_loaderContext->inner,0u);
    if (!viewBundle.getContents().empty())
    {
        auto view = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(viewBundle.getContents().begin()[0]);

        // TODO: here for the bumpmap bug
        auto found = _loaderContext->derivMapCache.find(view->getCreationParameters().image);
        if (found!=_loaderContext->derivMapCache.end())
        {
            const float normalizationFactor = found->second;
            scale *= normalizationFactor;
        }

        types[0] = asset::IAsset::ET_SAMPLER;
        const std::string samplerKey = _loaderContext->samplerCacheKey(_loaderContext->computeSamplerParameters(_element->bitmap));
        auto samplerBundle = _loaderContext->override_->findCachedAsset(samplerKey, types, _loaderContext->inner, 0u);
        assert(!samplerBundle.getContents().empty());
        auto sampler = core::smart_refctd_ptr_static_cast<asset::ICPUSampler>(samplerBundle.getContents().begin()[0]);

        return {view, sampler, scale};
    }
    return { nullptr, nullptr, core::nan<float>()};
}

auto CMitsubaMaterialCompilerFrontend::getErrorTexture(const ext::MitsubaLoader::SContext* _loaderContext) -> tex_ass_type
{
    constexpr const char* ERR_TEX_CACHE_NAME = "nbl/builtin/image_view/dummy2d";
    constexpr const char* ERR_SMPLR_CACHE_NAME = "nbl/builtin/sampler/default";

    asset::IAsset::E_TYPE types[2]{ asset::IAsset::ET_IMAGE_VIEW, asset::IAsset::ET_TERMINATING_ZERO };
    auto bundle = _loaderContext->override_->findCachedAsset(ERR_TEX_CACHE_NAME, types, _loaderContext->inner, 0u);
    assert(!bundle.getContents().empty()); // this shouldnt ever happen since ERR_TEX_CACHE_NAME is builtin asset
        
    auto view = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(bundle.getContents().begin()[0]);

    types[0] = asset::IAsset::ET_SAMPLER;
    auto sbundle = _loaderContext->override_->findCachedAsset(ERR_SMPLR_CACHE_NAME, types, _loaderContext->inner, 0u);
    assert(!sbundle.getContents().empty()); // this shouldnt ever happen since ERR_SMPLR_CACHE_NAME is builtin asset

    auto smplr = core::smart_refctd_ptr_static_cast<asset::ICPUSampler>(sbundle.getContents().begin()[0]);

    return { view, smplr, 1.f };
}

auto CMitsubaMaterialCompilerFrontend::createIRNode(SContext& ctx, const CElementBSDF* _bsdf, const bool frontface) -> node_handle_t
{
    using namespace asset::material_compiler;


    auto getFloatOrTexture = [&ctx](const CElementTexture::FloatOrTexture& src) -> IR::INode::SParameter<float>
    {
        if (src.value.type == SPropertyElementData::INVALID)
        {
            IR::INode::STextureSource tex;
            std::tie(tex.image, tex.sampler, tex.scale) = getTexture(ctx.m_loaderContext,src.texture);
            return tex;
        }
        else
            return src.value.fvalue;
    };
    auto getSpectrumOrTexture = [&ctx](const CElementTexture::SpectrumOrTexture& src, const E_IMAGE_VIEW_SEMANTIC semantic=EIVS_IDENTITIY) -> IR::INode::SParameter<IR::INode::color_t>
    {
        if (src.value.type == SPropertyElementData::INVALID)
        {
            IR::INode::STextureSource tex;
            std::tie(tex.image, tex.sampler, tex.scale) = getTexture(ctx.m_loaderContext,src.texture,semantic);
            return tex;
        }
        else
            return src.value.vvalue;
    };

    auto setAlpha = [&getFloatOrTexture](IR::IMicrofacetBSDFNode* node, const bool rough, const auto& _bsdfEl) -> void
    {
        if (rough)
        {
            using bsdf_t = std::remove_const_t<std::remove_reference_t<decltype(_bsdfEl)>>;
            if constexpr (std::is_same_v<bsdf_t,CElementBSDF::AllDiffuse> || std::is_same_v<bsdf_t,CElementBSDF::DiffuseTransmitter>)
                node->alpha_u = getFloatOrTexture(_bsdfEl.alpha);
            else
                node->alpha_u = getFloatOrTexture(_bsdfEl.alphaU);
            node->alpha_v = node->alpha_u;
            if constexpr (std::is_base_of_v<CElementBSDF::RoughSpecularBase,bsdf_t>)
            {
                constexpr IR::CMicrofacetSpecularBSDFNode::E_NDF ndfMap[4] =
                {
                    IR::CMicrofacetSpecularBSDFNode::ENDF_BECKMANN,
                    IR::CMicrofacetSpecularBSDFNode::ENDF_GGX,
                    IR::CMicrofacetSpecularBSDFNode::ENDF_PHONG,
                    IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY
                };
                auto& ndf = static_cast<IR::ICookTorranceBSDFNode*>(node)->ndf;
                ndf = ndfMap[_bsdfEl.distribution];
                if (ndf==IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                    node->alpha_v = getFloatOrTexture(_bsdfEl.alphaV);
            }
        }
        else
            node->setSmooth();
    };

    auto& hashCons = ctx.m_hashCons;
    auto ir_node = IR::invalid_node;
    auto findAndSetChild = [&hashCons,_bsdf,frontface](IR::INode* pNode, const uint32_t childIx) -> node_handle_t
    {
        const CElementBSDF* child = _bsdf->type==CElementBSDF::COATING ? _bsdf->coating.bsdf[childIx]:_bsdf->meta_common.bsdf[childIx];
        const bool twosided = unwindTwosided(child);
        auto found = hashCons.find({child,frontface||twosided});
        assert(found!=hashCons.end());
        return pNode->getChildrenArray()[childIx] = std::get<node_handle_t>(*found);
    };
    auto* ir = ctx.m_ir;
    const auto type = _bsdf->type;
    switch (type)
    {
        case CElementBSDF::MASK:
        {
            IR::INode::SParameter<IR::INode::color_t> opacity = getSpectrumOrTexture(_bsdf->mask.opacity,EIVS_BLEND_WEIGHT);
            ir_node = ir->allocNode<IR::COpacityNode>(1);
            auto pNode = ir->getNode<IR::COpacityNode>(ir_node);
            pNode->opacity = std::move(opacity);
            findAndSetChild(pNode,0u);
            break;
        }
        case CElementBSDF::DIFFUSE:
        case CElementBSDF::ROUGHDIFFUSE:
        {
            assert(frontface); // hash consing should have replaced this one with black diffuse
            ir_node = ir->allocNode<IR::CMicrofacetDiffuseBSDFNode>(0);
            auto pNode = ir->getNode<IR::CMicrofacetDiffuseBSDFNode>(ir_node);
            setAlpha(pNode,type==CElementBSDF::ROUGHDIFFUSE,_bsdf->diffuse);
            pNode->reflectance = getSpectrumOrTexture(_bsdf->diffuse.reflectance);
            break;
        }
        case CElementBSDF::CONDUCTOR:
        case CElementBSDF::ROUGHCONDUCTOR:
        {
            assert(frontface); // hash consing should have replaced this one with black diffuse
            ir_node = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>(0);
            auto pNode = ir->getNode<IR::CMicrofacetSpecularBSDFNode>(ir_node);
            setAlpha(pNode,type==CElementBSDF::ROUGHCONDUCTOR,_bsdf->conductor);
            const float extEta = _bsdf->conductor.extEta;
            pNode->eta = _bsdf->conductor.eta.vvalue/extEta;
            pNode->etaK = _bsdf->conductor.k.vvalue/extEta;
            // IR TODO: first check if eta=1 or etaK=INF (no idea actually what would cause pitch black fresnel), then replace with NOOP
            break;
        }
        case CElementBSDF::DIFFUSE_TRANSMITTER:
        {
            ir_node = ir->allocNode<IR::CMicrofacetDifftransBSDFNode>(0);
            auto pNode = ir->getNode<IR::CMicrofacetDifftransBSDFNode>(ir_node);
            pNode->setSmooth();
            pNode->transmittance = getSpectrumOrTexture(_bsdf->difftrans.transmittance);
            break;
        }
        case CElementBSDF::PLASTIC:
        case CElementBSDF::ROUGHPLASTIC:
        {
            assert(frontface); // hash consing should have replaced this one with black diffuse
            const bool rough = type==CElementBSDF::ROUGHPLASTIC;

            ir_node = ir->allocNode<IR::CMicrofacetCoatingBSDFNode>(1);
            auto coat = ir->getNode<IR::CMicrofacetCoatingBSDFNode>(ir_node);
            setAlpha(coat,rough,_bsdf->plastic);
            coat->eta = IR::INode::color_t(_bsdf->plastic.intIOR/_bsdf->plastic.extIOR);

            {
                CElementBSDF tmp("impl_tmp_diffuse_element");
                tmp.type = rough ? CElementBSDF::ROUGHDIFFUSE:CElementBSDF::DIFFUSE;
                tmp.diffuse.alpha = _bsdf->plastic.alpha;
                tmp.diffuse.reflectance = _bsdf->plastic.diffuseReflectance;
                coat->getChildrenArray()[0] = createIRNode(ctx,&tmp,frontface);
            }
            break;
        }
        case CElementBSDF::DIELECTRIC:
        case CElementBSDF::THINDIELECTRIC:
        case CElementBSDF::ROUGHDIELECTRIC:
        {
            ir_node = ir->allocNode<IR::CMicrofacetDielectricBSDFNode>(0);
            auto dielectric = ir->getNode<IR::CMicrofacetDielectricBSDFNode>(ir_node);
            setAlpha(dielectric,type==CElementBSDF::ROUGHDIELECTRIC,_bsdf->dielectric);
            dielectric->thin = type==CElementBSDF::THINDIELECTRIC;
            if (frontface || dielectric->thin)
                dielectric->eta = IR::INode::color_t(_bsdf->dielectric.intIOR/_bsdf->dielectric.extIOR);
            else
                dielectric->eta = IR::INode::color_t(_bsdf->dielectric.extIOR/_bsdf->dielectric.intIOR);
            break;
        }
        case CElementBSDF::BUMPMAP:
        {
            ir_node = ir->allocNode<IR::CGeomModifierNode>(1,IR::CGeomModifierNode::ET_DERIVATIVE);
            auto* pNode = ir->getNode<IR::CGeomModifierNode>(ir_node);
            //no other source supported for now (uncomment in the future) [far future TODO]
            //node->source = IR::CGeomModifierNode::ESRC_TEXTURE;

            std::tie(pNode->texture.image,pNode->texture.sampler,pNode->texture.scale) =
                getTexture(ctx.m_loaderContext,_bsdf->bumpmap.texture,_bsdf->bumpmap.wasNormal ? EIVS_NORMAL_MAP:EIVS_BUMP_MAP);
            
            findAndSetChild(pNode,0);
            break;
        }
        case CElementBSDF::COATING:
        case CElementBSDF::ROUGHCOATING:
        {
            assert(frontface); // hash consing should have replaced this one with black diffuse

            const bool rough = type==CElementBSDF::ROUGHDIELECTRIC;

            ir_node = ir->allocNode<IR::CMicrofacetCoatingBSDFNode>(1u);
            auto* coat = ir->getNode<IR::CMicrofacetCoatingBSDFNode>(ir_node);
            setAlpha(coat,rough,_bsdf->coating);

            coat->eta = IR::INode::color_t(_bsdf->dielectric.intIOR/_bsdf->dielectric.extIOR);

            const float thickness = _bsdf->coating.thickness;
            coat->thicknessSigmaA = getSpectrumOrTexture(_bsdf->coating.sigmaA);
            if (coat->thicknessSigmaA.isConstant())
                coat->thicknessSigmaA.constant *= thickness;
            else
                coat->thicknessSigmaA.texture.scale *= thickness;
            
            findAndSetChild(coat,0);
            break;
        }
        break;
        case CElementBSDF::BLEND_BSDF:
        {
            ir_node = ir->allocNode<IR::CBSDFBlendNode>(2);
            auto* pNode = ir->getNode<IR::CBSDFBlendNode>(ir_node);
            if (_bsdf->blendbsdf.weight.value.type == SPropertyElementData::INVALID)
            {
                std::tie(pNode->weight.texture.image,pNode->weight.texture.sampler,pNode->weight.texture.scale) =
                    getTexture(ctx.m_loaderContext,_bsdf->blendbsdf.weight.texture,EIVS_BLEND_WEIGHT);
                assert(!core::isnan(pNode->weight.texture.scale));
            }
            else
                pNode->weight = IR::INode::color_t(_bsdf->blendbsdf.weight.value.fvalue);
            findAndSetChild(pNode,0);
            findAndSetChild(pNode,1);
        }
        break;
        case CElementBSDF::MIXTURE_BSDF:
        {
            ir_node = ir->allocNode<IR::CBSDFMixNode>(_bsdf->mixturebsdf.childCount);
            auto* pNode = ir->getNode<IR::CBSDFMixNode>(ir_node);
            const auto* weightIt = _bsdf->mixturebsdf.weights;
            for (size_t i=0u; i<pNode->getChildCount(); i++)
            {
                pNode->weights[i] = *(weightIt++);
                findAndSetChild(pNode,i);
            }
            break;
        }
        default:
            break;
    }
    assert(ir_node!=IR::invalid_node);
    const bool inserted = std::get<bool>(hashCons.insert({{_bsdf,frontface},ir_node}));
    assert(inserted);
    return ir_node;
}

auto CMitsubaMaterialCompilerFrontend::compileToIRTree(SContext& ctx, const CElementBSDF* _root) -> front_and_back_t
{
    using namespace asset::material_compiler;

    auto traverseForSide = [&ctx,_root](const bool frontface) -> node_handle_t
    {
        auto root = IR::invalid_node;
        //create frontface IR
        struct DFSData
        {
            const CElementBSDF* bsdf;
            // to do Post-Order DFS we need to visit parent twice
            uint8_t visited : 1;

            CElementBSDF::Type type() const { return bsdf->type; }
        };
        core::stack<DFSData> dfs;
        auto pre = [&](const CElementBSDF* _bsdf)
        {
            DFSData el = {_bsdf};
            const bool twosided = unwindTwosided(el.bsdf);
            //
            auto found = ctx.m_hashCons.find({el.bsdf,frontface||twosided});
            if (found!=ctx.m_hashCons.end())
                root = std::get<node_handle_t>(*found);
            else
            {
                // we traversed the tree before, we must be able to find an entry for a front facing branch!
                assert(!twosided);
                // only meta nodes get pushed onto stack
                if (el.bsdf->isMeta())
                {
                    el.visited = false;
                    dfs.push(std::move(el));
                }
                else
                    root = createIRNode(ctx,el.bsdf,frontface);
            }
        };
        pre(_root);
        while (!dfs.empty())
        {
            auto& parent = dfs.top();
            assert(parent.type()!=CElementBSDF::TWO_SIDED);
            assert(parent.bsdf->isMeta());

            if (parent.visited)
            {
                dfs.pop();
                auto found = ctx.m_hashCons.find({parent.bsdf,frontface});
                if (found!=ctx.m_hashCons.end())
                    root = std::get<node_handle_t>(*found);
                else
                    root = createIRNode(ctx,parent.bsdf,frontface);
            }
            else
            {
                parent.visited = true;

                const bool isCoating = parent.type()==CElementBSDF::COATING;
                const auto childCount = isCoating ? parent.bsdf->coating.childCount:parent.bsdf->meta_common.childCount;
                for (auto i=0u; i<childCount; i++)
                {
                    const auto child = isCoating ? parent.bsdf->coating.bsdf[i]:parent.bsdf->meta_common.bsdf[i];
                    pre(child);
                }
            }
        }
        ctx.m_ir->addRootNode(root);
        return root;
    };

    // set up an invalid/vanta-black/NOOP node first
    {
        CElementBSDF invalid("invalid_vanta_black_noop");
        invalid.type = CElementBSDF::DIFFUSE;
        invalid.diffuse.reflectance = 0.f;
        createIRNode(ctx,&invalid,true);
    }

    // its important to generate the IR for the front-face first, so that backface can reuse
    // any twosided or orientation agnostic nodes which are in the consing cache
    front_and_back_t retval;
    retval.front = traverseForSide(true);
    retval.back = traverseForSide(false);
    return retval;
}

bool CMitsubaMaterialCompilerFrontend::MerkleTree::equal_to::operator()(const MerkleTree& lhs, const MerkleTree& rhs) const
{
    return lhs.bsdf==rhs.bsdf && lhs.frontface==rhs.frontface;
}

std::size_t CMitsubaMaterialCompilerFrontend::MerkleTree::hash::operator()(const MerkleTree& node) const
{
    return ptrdiff_t(node.bsdf);
}

}