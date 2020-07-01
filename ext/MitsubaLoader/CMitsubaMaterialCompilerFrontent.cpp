#include "CMitsubaMaterialCompilerFrontent.h"

#include <irr/core/Types.h>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

core::smart_refctd_ptr<asset::material_compiler::IR> CMitsubaMaterialCompilerFrontent::compileToIR(const CElementBSDF* _bsdf)
{
    using namespace asset;
    using namespace material_compiler;

    auto ir = core::make_smart_refctd_ptr<IR>();
    ir->root = core::make_smart_refctd_ptr<IR::INode>(IR::INode::ES_MATERIAL);
    bool twosided = false;
    bool createdSurfaces = false;
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

    core::stack<const CElementBSDF*> bsdfStack;
    bsdfStack.push(_bsdf);
    core::stack<IR::INode*> nodeStack;
    nodeStack.push(ir->root.get());
    uint32_t childrenCountdown = 1u;

    while (!bsdfStack.empty())
    {
        current = bsdfStack.top();
        bsdfStack.pop();

        core::smart_refctd_ptr<IR::INode> nextSym;
        switch (current->type)
        {
        case CElementBSDF::TWO_SIDED:
            twosided = true;
            bsdfStack.push(_bsdf->twosided.bsdf[0]);
            break;
        case CElementBSDF::DIFFUSE:
        case CElementBSDF::ROUGHDIFFUSE:
            nextSym = core::make_smart_refctd_ptr<IR::CMicrofacetDiffuseBSDFNode>();
            getSpectrumOrTexture(current->diffuse.reflectance, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym.get())->reflectance);
            getFloatOrTexture(current->diffuse.alpha, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym.get())->alpha_u);
            getFloatOrTexture(current->diffuse.alpha, static_cast<IR::CMicrofacetDiffuseBSDFNode*>(nextSym.get())->alpha_v);
            break;
        case CElementBSDF::CONDUCTOR:
        case CElementBSDF::ROUGHCONDUCTOR:
        {
            nextSym = core::make_smart_refctd_ptr<IR::CMicrofacetSpecularBSDFNode>();
            auto* node = static_cast<IR::CMicrofacetSpecularBSDFNode*>(nextSym.get());
            node->ndf = ndfMap[current->conductor.distribution];
            getFloatOrTexture(current->conductor.alphaU, node->alpha_u);
            if (node->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                getFloatOrTexture(current->conductor.alphaV, node->alpha_v);
            else
                node->alpha_v = node->alpha_u;
            //TODO IoR
        }
        break;
        case CElementBSDF::PLASTIC:
        case CElementBSDF::ROUGHPLASTIC:
        {
            nextSym = core::make_smart_refctd_ptr<IR::CBSDFCombinerNode>(IR::CBSDFCombinerNode::ET_DIFFUSE_AND_SPECULAR);
            nextSym->children = core::make_refctd_dynamic_array<IR::INode::children_smart_array_t>(2u);

            auto& diffuse = (*nextSym->children)[0];
            diffuse = core::make_smart_refctd_ptr<IR::CMicrofacetDiffuseBSDFNode>();
            auto& specular = (*nextSym->children)[1];
            specular = core::make_smart_refctd_ptr<IR::CMicrofacetSpecularBSDFNode>();

            auto* node_specular = static_cast<IR::CMicrofacetSpecularBSDFNode*>(specular.get());
            node_specular->ndf = ndfMap[current->plastic.distribution];
            getFloatOrTexture(current->plastic.alphaU, node_specular->alpha_u);
            if (node_specular->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                getFloatOrTexture(current->plastic.alphaV, node_specular->alpha_v);
            else
                node_specular->alpha_v = node_specular->alpha_u;
            node_specular->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            node_specular->scatteringMode = IR::CMicrofacetSpecularBSDFNode::ESM_REFLECT;

            auto* node_diffuse = static_cast<IR::CMicrofacetDiffuseBSDFNode*>(diffuse.get());
            getSpectrumOrTexture(current->plastic.diffuseReflectance, node_diffuse->reflectance);
            node_diffuse->alpha_u = node_specular->alpha_u;
            node_diffuse->alpha_v = node_specular->alpha_v;
        }
        break;
        case CElementBSDF::DIELECTRIC:
        case CElementBSDF::THINDIELECTRIC:
        case CElementBSDF::ROUGHDIELECTRIC:
        {
            nextSym = core::make_smart_refctd_ptr<IR::CBSDFCombinerNode>(IR::CBSDFCombinerNode::ET_FRESNEL_BLEND);
            nextSym->children = core::make_refctd_dynamic_array<IR::INode::children_smart_array_t>(2u);

            auto& refl = (*nextSym->children)[0];
            refl = core::make_smart_refctd_ptr<IR::CMicrofacetSpecularBSDFNode>();
            auto& trans = (*nextSym->children)[1];
            trans = core::make_smart_refctd_ptr<IR::CMicrofacetSpecularBSDFNode>();

            auto* node_refl = static_cast<IR::CMicrofacetSpecularBSDFNode*>(refl.get());
            auto* node_trans = static_cast<IR::CMicrofacetSpecularBSDFNode*>(trans.get());

            node_refl->ndf = ndfMap[current->dielectric.distribution];
            node_refl->shadowing = IR::CMicrofacetSpecularBSDFNode::EST_SMITH;
            getFloatOrTexture(current->dielectric.alphaU, node_refl->alpha_u);
            if (node_refl->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                getFloatOrTexture(current->dielectric.alphaV, node_refl->alpha_v);
            else
                node_refl->alpha_v = node_refl->alpha_u;
            node_trans->ndf = node_refl->ndf;
            node_trans->shadowing = node_refl->shadowing;
            node_trans->alpha_u = node_refl->alpha_u;
            node_trans->alpha_v = node_refl->alpha_v;
        }
        break;
        case CElementBSDF::BUMPMAP:
        {
            nextSym = core::make_smart_refctd_ptr<IR::CGeomModifierNode>(IR::CGeomModifierNode::ET_HEIGHT);
            auto* node = static_cast<IR::CGeomModifierNode*>(nextSym.get());
            node->source = IR::CGeomModifierNode::ESRC_TEXTURE;
            //TODO load texture
            bsdfStack.push(current->bumpmap.bsdf[0]);
        }
        break;
        case CElementBSDF::COATING:
        case CElementBSDF::ROUGHCOATING:
        {
            nextSym = core::make_smart_refctd_ptr<IR::CCoatingBSDFNode>();
            nextSym->children = core::make_refctd_dynamic_array<IR::INode::children_smart_array_t>(1u);

            auto* node = static_cast<IR::CCoatingBSDFNode*>(nextSym.get());
            node->ndf = ndfMap[current->coating.distribution];
            node->shadowing = IR::CCoatingBSDFNode::EST_SMITH;
            getFloatOrTexture(current->coating.alphaU, node->alpha_u);
            if (node->ndf == IR::CMicrofacetSpecularBSDFNode::ENDF_ASHIKHMIN_SHIRLEY)
                getFloatOrTexture(current->coating.alphaV, node->alpha_v);
            else
                node->alpha_v = node->alpha_u;
            node->thickness = current->coating.thickness;
            getSpectrumOrTexture(current->coating.sigmaA, node->sigmaA);
            bsdfStack.push(current->coating.bsdf[0]);
        }
        break;
        case CElementBSDF::BLEND_BSDF:
        {
            nextSym = core::make_smart_refctd_ptr<IR::CBSDFBlendNode>();
            nextSym->children = core::make_refctd_dynamic_array<IR::INode::children_smart_array_t>(2u);

            auto* node = static_cast<IR::CBSDFBlendNode*>(nextSym.get());
            getFloatOrTexture(current->blendbsdf.weight, node->weight);

            bsdfStack.push(current->blendbsdf.bsdf[0]);
            bsdfStack.push(current->blendbsdf.bsdf[1]);
        }
        break;
        case CElementBSDF::MIXTURE_BSDF:
        {
            nextSym = core::make_smart_refctd_ptr<IR::CBSDFMixNode>();
            auto* node = static_cast<IR::CBSDFMixNode*>(nextSym.get());
            const size_t cnt = current->mixturebsdf.childCount;
            nextSym->children = core::make_refctd_dynamic_array<IR::INode::children_smart_array_t>(current->mixturebsdf.childCount);
            node->weights = core::make_refctd_dynamic_array<IR::CBSDFMixNode::weights_t>(current->mixturebsdf.childCount);
            for (size_t i = 0u; i < node->weights->size(); ++i)
                (*node->weights)[i] = current->mixturebsdf.weights[i];

            for (uint32_t i = 0u; i < cnt; ++i)
                bsdfStack.push(current->mixturebsdf.bsdf[i]);
        }
        break;
        }

        if (nextSym->children)
            nodeStack.push(nextSym.get());
        IR::INode* parent = nodeStack.top();
        if (!--childrenCountdown)
            nodeStack.pop();

        (*parent->children)[childrenCountdown] = nextSym;//TODO consider std::move
        if (!childrenCountdown)
            childrenCountdown = nodeStack.top()->children->size();

        /*if (!createdSurfaces)
        {
            auto createSurfaces = [&]()
            {
                auto surface = core::make_smart_refctd_ptr<IR::INode>(IR::INode::ES_FRONT_SURFACE);
                //surface->
            };

            if (nextSym->symbol == IR::INode::ES_GEOM_MODIFIER)
            {
                createdSurfaces = true;
            }
            else if (parent->symbol == IR::INode::ES_MATERIAL)
                createdSurfaces = true;
        }*/
    }
}

}}}