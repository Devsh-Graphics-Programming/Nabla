#include "CMitsubaMaterialCompilerFrontend.h"

#include <irr/core/Types.h>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

asset::material_compiler::IR::INode* CMitsubaMaterialCompilerFrontend::compileToIRTree(asset::material_compiler::IR* ir, const CElementBSDF* _bsdf)
{
    using namespace asset;
    using namespace material_compiler;

    IR::INode* root = ir->allocNode<IR::CMaterialNode>();
    bool twosided = false;
    IR::INode::SParameter<IR::INode::color_t> opacity;
    {
        opacity.source = IR::INode::EPS_CONSTANT;
        opacity.value.constant = IR::INode::color_t(1.f);
    }
    const CElementBSDF* current = _bsdf;

    auto getFloatOrTexture = [](const CElementTexture::FloatOrTexture& src, IR::INode::SParameter<float>& dst)
    {
        if (src.value.type == SPropertyElementData::INVALID)
        {
            dst.source = IR::INode::EPS_TEXTURE;
            //TODO load texture
        }
        else
        {
            dst.source = IR::INode::EPS_CONSTANT;
            dst.value.constant = src.value.fvalue;
        }
    };
    auto getSpectrumOrTexture = [](const CElementTexture::SpectrumOrTexture& src, IR::INode::SParameter<IR::INode::color_t>& dst)
    {
        if (src.value.type == SPropertyElementData::INVALID)
        {
            dst.source = IR::INode::EPS_TEXTURE;
            //TODO load texture
        }
        else
        {
            dst.source = IR::INode::EPS_CONSTANT;
            dst.value.constant = src.value.vvalue;
        }
    };

    IR::CMicrofacetSpecularBSDFNode::E_NDF ndfMap[4]{
        IR::CMicrofacetSpecularBSDFNode::ENDF_BECKMANN,
        IR::CMicrofacetSpecularBSDFNode::ENDF_GGX,
        IR::CMicrofacetSpecularBSDFNode::ENDF_PHONG,
        IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY
    };

    core::queue<const CElementBSDF*> bsdfQ;
    bsdfQ.push(_bsdf);
    core::queue<IR::INode*> nodeQ;
    nodeQ.push(root);
    uint32_t childrenCountdown = 1u;
    IR::INode* parent = nodeQ.front();

    while (!bsdfQ.empty())
    {
        current = bsdfQ.front();
        bsdfQ.pop();

        IR::INode* nextSym;
        switch (current->type)
        {
        case CElementBSDF::TWO_SIDED:
            twosided = true;
            bsdfQ.push(current->twosided.bsdf[0]);
            break;
        case CElementBSDF::MASK:
            getSpectrumOrTexture(current->mask.opacity, opacity);
            bsdfQ.push(current->mask.bsdf[0]);
            break;
        case CElementBSDF::DIFFUSE:
        case CElementBSDF::ROUGHDIFFUSE:
            nextSym = ir->allocNode<IR::CMicrofacetDiffuseBSDFNode>();
            getSpectrumOrTexture(current->diffuse.reflectance, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym)->reflectance);
            getFloatOrTexture(current->diffuse.alpha, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym)->alpha_u);
            getFloatOrTexture(current->diffuse.alpha, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym)->alpha_v);
            break;
        case CElementBSDF::CONDUCTOR:
        case CElementBSDF::ROUGHCONDUCTOR:
        {
            nextSym = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>();
            auto* node = static_cast<IR::CMicrofacetSpecularBSDFNode*>(nextSym);
            node->ndf = ndfMap[current->conductor.distribution];
            node->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            node->scatteringMode = IR::CMicrofacetSpecularBSDFNode::ESM_REFLECT;
            getFloatOrTexture(current->conductor.alphaU, node->alpha_u);
            if (node->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                getFloatOrTexture(current->conductor.alphaV, node->alpha_v);
            else
                node->alpha_v = node->alpha_u;
            const float extEta = current->conductor.extEta;
            node->eta = current->conductor.eta.vvalue/extEta;
            node->etaK = current->conductor.k.vvalue/extEta;
        }
        break;
        case CElementBSDF::PLASTIC:
        case CElementBSDF::ROUGHPLASTIC:
        {
            nextSym = ir->allocNode<IR::CBSDFCombinerNode>(IR::CBSDFCombinerNode::ET_DIFFUSE_AND_SPECULAR);
            nextSym->children = core::make_refctd_dynamic_array<IR::INode::children_array_t>(2u);

            const float eta = current->plastic.intIOR/current->plastic.extIOR;

            auto& diffuse = nextSym->children[0];
            diffuse = ir->allocNode<IR::CMicrofacetDiffuseBSDFNode>();
            auto& specular = nextSym->children[1];
            specular = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>();

            auto* node_specular = static_cast<IR::CMicrofacetSpecularBSDFNode*>(specular);
            node_specular->ndf = ndfMap[current->plastic.distribution];
            getFloatOrTexture(current->plastic.alphaU, node_specular->alpha_u);
            if (node_specular->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                getFloatOrTexture(current->plastic.alphaV, node_specular->alpha_v);
            else
                node_specular->alpha_v = node_specular->alpha_u;
            node_specular->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            node_specular->eta = IR::INode::color_t(eta);
            node_specular->scatteringMode = IR::CMicrofacetSpecularBSDFNode::ESM_REFLECT;

            auto* node_diffuse = static_cast<IR::CMicrofacetDiffuseBSDFNode*>(diffuse);
            getSpectrumOrTexture(current->plastic.diffuseReflectance, node_diffuse->reflectance);
            node_diffuse->alpha_u = node_specular->alpha_u;
            node_diffuse->alpha_v = node_specular->alpha_v;
        }
        break;
        case CElementBSDF::DIELECTRIC:
        case CElementBSDF::THINDIELECTRIC:
        case CElementBSDF::ROUGHDIELECTRIC:
        {
            nextSym = ir->allocNode<IR::CBSDFCombinerNode>(IR::CBSDFCombinerNode::ET_FRESNEL_BLEND);
            nextSym->children = core::make_refctd_dynamic_array<IR::INode::children_array_t>(2u);

            const float eta = current->dielectric.intIOR/current->dielectric.extIOR;

            auto& refl = nextSym->children[0];
            refl = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>();
            auto& trans = nextSym->children[1];
            trans = ir->allocNode<IR::CMicrofacetSpecularBSDFNode>();

            auto* node_refl = static_cast<IR::CMicrofacetSpecularBSDFNode*>(refl);
            auto* node_trans = static_cast<IR::CMicrofacetSpecularBSDFNode*>(trans);

            node_refl->ndf = ndfMap[current->dielectric.distribution];
            node_refl->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            getFloatOrTexture(current->dielectric.alphaU, node_refl->alpha_u);
            node_refl->eta = IR::INode::color_t(eta);
            if (node_refl->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                getFloatOrTexture(current->dielectric.alphaV, node_refl->alpha_v);
            else
                node_refl->alpha_v = node_refl->alpha_u;
            node_trans->ndf = node_refl->ndf;
            node_trans->shadowing = node_refl->shadowing;
            node_trans->alpha_u = node_refl->alpha_u;
            node_trans->alpha_v = node_refl->alpha_v;
            node_trans->eta = IR::INode::color_t(eta);
        }
        break;
        case CElementBSDF::BUMPMAP:
        {
            nextSym = ir->allocNode<IR::CGeomModifierNode>(IR::CGeomModifierNode::ET_HEIGHT);
            auto* node = static_cast<IR::CGeomModifierNode*>(nextSym);
            node->source = IR::CGeomModifierNode::ESRC_TEXTURE;
            //TODO load texture
            bsdfQ.push(current->bumpmap.bsdf[0]);
        }
        break;
        case CElementBSDF::COATING:
        case CElementBSDF::ROUGHCOATING:
        {
            nextSym = ir->allocNode<IR::CCoatingBSDFNode>();
            nextSym->children = core::make_refctd_dynamic_array<IR::INode::children_array_t>(1u);

            const float eta = current->dielectric.intIOR/current->dielectric.extIOR;

            auto* node = static_cast<IR::CCoatingBSDFNode*>(nextSym);
            node->ndf = ndfMap[current->coating.distribution];
            node->shadowing = IR::CCoatingBSDFNode::EST_SMITH;
            getFloatOrTexture(current->coating.alphaU, node->alpha_u);
            if (node->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                getFloatOrTexture(current->coating.alphaV, node->alpha_v);
            else
                node->alpha_v = node->alpha_u;
            node->thickness = current->coating.thickness;
            getSpectrumOrTexture(current->coating.sigmaA, node->sigmaA);
            node->eta = IR::INode::color_t(eta);
            bsdfQ.push(current->coating.bsdf[0]);
        }
        break;
        case CElementBSDF::BLEND_BSDF:
        {
            nextSym = ir->allocNode<IR::CBSDFBlendNode>();
            nextSym->children = core::make_refctd_dynamic_array<IR::INode::children_array_t>(2u);

            auto* node = static_cast<IR::CBSDFBlendNode*>(nextSym);
            getFloatOrTexture(current->blendbsdf.weight, node->weight);

            bsdfQ.push(current->blendbsdf.bsdf[1]);
            bsdfQ.push(current->blendbsdf.bsdf[0]);
        }
        break;
        case CElementBSDF::MIXTURE_BSDF:
        {
            nextSym = ir->allocNode<IR::CBSDFMixNode>();
            auto* node = static_cast<IR::CBSDFMixNode*>(nextSym);
            const size_t cnt = current->mixturebsdf.childCount;
            nextSym->children = core::make_refctd_dynamic_array<IR::INode::children_array_t>(current->mixturebsdf.childCount);
            node->weights = core::make_refctd_dynamic_array<IR::CBSDFMixNode::weights_t>(current->mixturebsdf.childCount);
            for (int32_t i = cnt-1u; i >= 0; --i)
                (*node->weights)[i] = current->mixturebsdf.weights[i];

            for (int32_t i = cnt-1u; i >= 0; --i)
                bsdfQ.push(current->mixturebsdf.bsdf[i]);
        }
        break;
        }

        IR::INode* newParent = nullptr;
        if (nextSym->children)
            nodeQ.push(nextSym);
        if (!--childrenCountdown)
        {
            newParent = nodeQ.front();
            nodeQ.pop();
        }

        parent->children[childrenCountdown] = nextSym;//TODO consider std::move
        if (newParent)
        {
            parent = newParent;
            childrenCountdown = parent->children.count;
        }
    }

    static_cast<IR::CMaterialNode*>(root)->opacity = opacity;

    IR::INode* surfParent = root;
    if (surfParent->children[0]->symbol == IR::INode::ES_GEOM_MODIFIER)
        surfParent = surfParent->children[0];

    IR::INode::children_array_t surfaces;
    surfaces.count = twosided?2u:1u;
    auto& surf = surfaces[0];
    surf = ir->allocNode<IR::INode>(IR::INode::ES_FRONT_SURFACE);
    surf->children = surfaces;
    std::swap(surf->children,surfParent->children);
    if (surfaces.count>1u) {
        surfaces[1] = ir->allocNode<IR::INode>(IR::INode::ES_BACK_SURFACE);
        surfaces[1]->children = surf->children;
    }

    return root;
}

}}}