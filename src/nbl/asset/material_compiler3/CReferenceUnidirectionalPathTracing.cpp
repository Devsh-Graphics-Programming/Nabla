// Copyright (C) 2022-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#define _NBL_ASSET_MATERIAL_COMPILER3_C_REFERENCE_UNIDIRECTIONAL_PATH_TRACING_CPP_
#include "nbl/asset/material_compiler3/CReferenceUnidirectionalPathTracing.h"

namespace nbl::asset::material_compiler3
{

core::smart_refctd_ptr<CReferenceUnidirectionalPathTracing::CResult> CReferenceUnidirectionalPathTracing::compile(const CTrueIR* ir, const std::span<const CTrueIR::SMaterialHandle> materialHandles)
{
    auto res = core::make_smart_refctd_ptr<CResult>();

    // TODO: handle textures somehow
    // TODO: where do all the type aliases come from? e.g. sample_t, vector3_t, etc.
    std::ostringstream code;

    code << R"===(
template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>
struct gen_cache;

template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>
struct OrientedMaterial;
)===";

    // loop through layers in IR and construct materials map? maybe just node map is fine
    core::vector<CTrueIR::typed_pointer_type<const CTrueIR::INode>> nodeStack;
    //core::unordered_set<CTrueIR::typed_pointer_type<const CTrueIR::INode>> visitedNodes;
    core::unordered_map<CTrueIR::typed_pointer_type<const CTrueIR::INode>, TraversalNodeInfo> nodeInfos;
    auto compileBSDFRootNode = [&](CTrueIR::typed_pointer_type<const CTrueIR::INode> rootHandle) -> void {
        nodeStack.clear();
        nodeInfos.clear();

        nodeStack.push_back(rootHandle);
        const auto& pool = ir->getObjectPool();

        std::string rootHandleString;
        while (!nodeStack.empty())
        {
            const auto handle = nodeStack.back();
            nodeStack.pop_back();
            const auto* node = pool.deref(handle);
            if (!node)
                continue;

            getMaterialDeclarationCode(code, node, ir);
            traverseIRNode(node, ir, nodeStack, nodeInfos);
        }

        for (const auto& nodeInfo : std::views::values(nodeInfos))
        {
            getCacheDefineCode(code, nodeInfo, ir);
            getAlbedoHLSLCode(code, nodeInfo, ir);
            getNormalHLSLCode(code, nodeInfo, ir);
            getAOVThroughputHLSLCode(code, nodeInfo, ir);
            getTransparencyHLSLCode(code, nodeInfo, ir);
            getGenerateHLSLCode(code, nodeInfo, ir);
            getEvalWeightHLSLCode(code, nodeInfo, ir);
            getQuotientWeightHLSLCode(code, nodeInfo, ir);
            getEmissionHLSLCode(code, nodeInfo, ir);
            getCanGenerateHLSLCode(code, nodeInfo, ir);
            getChoiceTargetHLSLCode(code, nodeInfo, ir);
        }
        };

    const auto& materials = ir->getMaterials();
    for (uint32_t i = 0; i < materialHandles.size(); i++)
    {
        if (materialHandles[i].value == CTrueIR::SMaterialHandle::Invalid)
            continue;

        const auto& mat = materials[materialHandles[i].value];
        if (auto node = ir->getObjectPool().deref(mat.front.root); node)
            compileBSDFRootNode(mat.front.root);
        if (auto node = ir->getObjectPool().deref(mat.back.root); node)
            compileBSDFRootNode(mat.back.root);
    }

    // each layer/node writes as string its own code? or just hash
    // 8 functions each node: albedo, normal, aov_throughput, transparency, generate, quotientAndWeight, evalAndWeight, emission

    res->fragmentShaderSource_common = code.str();
    // TODO: set entry points in raytracingPipeline

    return res;
}

void CReferenceUnidirectionalPathTracing::traverseIRNode(const CTrueIR::INode* node, const CTrueIR* ir,
    core::vector<CTrueIR::typed_pointer_type<const CTrueIR::INode>>& nodeStack,
    core::unordered_map<CTrueIR::typed_pointer_type<const CTrueIR::INode>, TraversalNodeInfo>& nodeInfos)
{
    auto addChildToTraverse = [&](CTrueIR::typed_pointer_type<const CTrueIR::INode> handle, const TraversalNodeInfo& info) -> void {
        const auto [unused, inserted] = nodeInfos.insert({ handle, info });
        if (inserted)
            nodeStack.push_back(handle);
        };

    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        if (auto childBrdf = ir->getObjectPool().deref(layer->brdfTop); childBrdf)
        {
            TraversalNodeInfo info = {
                .node = childBrdf,
                .isTransmission = false
            };
            addChildToTraverse(layer->brdfTop, info);
        }
        if (auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission); childBtdf)
        {
            TraversalNodeInfo info = {
                .node = childBtdf,
                .isTransmission = true
            };
            addChildToTraverse(layer->firstTransmission, info);
        }
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        if (auto childProduct = ir->getObjectPool().deref(sum->product); childProduct)
        {
            TraversalNodeInfo info = {
                .node = childProduct,
                .isTransmission = false
            };
            addChildToTraverse(sum->product, info);
        }
        if (auto childRest = ir->getObjectPool().deref(sum->rest); childRest)
        {
            TraversalNodeInfo info = {
                .node = childRest,
                .isTransmission = false
            };
            addChildToTraverse(sum->rest, info);
        }
        break;
    }
    case CTrueIR::INode::EFinalType::CFactorCombiner:
    {
        const auto* combiner = dynamic_cast<const CTrueIR::CFactorCombiner*>(node);
        if (!combiner)
            break;

        const auto childCount = combiner->getChildCount();

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                TraversalNodeInfo info = {
                .node = child,
                .isTransmission = false
                };
                addChildToTraverse(combiner->getChildHandle(i), info);
            }
        }
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
    {
        const auto* contrib = dynamic_cast<const CTrueIR::CWeightedContributor*>(node);
        if (!contrib)
            break;

        if (auto childContrib = ir->getObjectPool().deref(contrib->contributor); childContrib)
        {
            TraversalNodeInfo info = {
                .node = childContrib,
                .isTransmission = false
            };
            addChildToTraverse(contrib->contributor, info);
        }
        if (auto childFactor = ir->getObjectPool().deref(contrib->factor); childFactor)
        {
            TraversalNodeInfo info = {
                .node = childFactor,
                .isTransmission = false
            };
            addChildToTraverse(contrib->factor, info);
        }
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CCorellatedTransmission*>(node);
        if (!transmission)
            break;

        if (auto child = ir->getObjectPool().deref(transmission->btdf); child)
        {
            TraversalNodeInfo info = {
                .node = child,
                .isTransmission = true
            };
            addChildToTraverse(transmission->btdf, info);
        }
        if (auto child = ir->getObjectPool().deref(transmission->brdfBottom); child)
        {
            TraversalNodeInfo info = {
                .node = child,
                .isTransmission = false
            };
            addChildToTraverse(transmission->brdfBottom, info);
        }
        if (auto child = ir->getObjectPool().deref(transmission->coated); child)
        {
            TraversalNodeInfo info = {
                .node = child,
                .isTransmission = true  // TODO: double check this
            };
            addChildToTraverse(transmission->coated, info);
        }
        if (auto child = ir->getObjectPool().deref(transmission->next); child)
        {
            TraversalNodeInfo info = {
                .node = child,
                .isTransmission = false
            };
            addChildToTraverse(transmission->next, info);
        }
        break;
    }
        // these ones don't have children/have children that needs to traverse
    case CTrueIR::INode::EFinalType::CSpectralVariable:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CEmitter:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::COrenNayar:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CCookTorrance:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CBeer:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CFresnel:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CThinInfiniteScatterCorrection:
        [[fallthrough]]
    default:
        break;
    }
}

std::string CReferenceUnidirectionalPathTracing::getHashAs4UintsString(const CTrueIR::INode* node, const CTrueIR* ir, const std::string& separator) const
{
    // break up hash into 8 pieces of uint32_t
    const auto hash = node->computeHash(ir->getObjectPool());
    uint32_t hashPieces[8];
    for (uint8_t i = 0; i < 8; i++)
        for (uint8_t j = 0; j < 4; j++)
            hlsl::glsl::bitfieldInsert(hashPieces[i], static_cast<uint32_t>(hash.data[4 * i + 0]), j * 8, 8);
    std::stringstream hashString;
    for (uint8_t i = 0; i < 8; i++)
    {
        hashString << hashPieces[i];
        if (i < 7)
            hashString << separator;
    }
    return hashString.str();
}

bool CReferenceUnidirectionalPathTracing::isNodeTypeContributor(CTrueIR::INode::EFinalType type) const
{
    return (type == CTrueIR::INode::EFinalType::COrenNayar) ||
        (type == CTrueIR::INode::EFinalType::CCookTorrance) ||
        (type == CTrueIR::INode::EFinalType::CDeltaTransmission);
}

void CReferenceUnidirectionalPathTracing::getMaterialDeclarationCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir)
{
    // define templates of node material structs
    const auto hashString = getHashAs4UintsString(node, ir);

    sstr << R"===(
template<>
struct gen_cache<)===" << hashString << R"===(>;

template<>
struct OrientedMaterial<)===" << hashString << R"===(>
{
    static spectral_t albedo();
    static vector3_t normal(NBL_CONST_REF_ARG(aniso_interaction_t) inter);
    static spectral_t aovThroughput();
    static scalar_t transparency();
    static sample_t generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache);
    static quotient_weight_t quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache);
    static value_weight_t evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter);
    static spectral_t emission();

    static bool canGenerate();
    static scalar_t choiceTarget(NBL_CONST_REF_ARG(aniso_interaction_t) inter, uint8_t chosenLobe, NBL_REF_ARG(scalar_t) choiceSum);
};
)===";

}

void CReferenceUnidirectionalPathTracing::getCacheDefineCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const auto* node = nodeInfo.node;
    const auto hashString = getHashAs4UintsString(node, ir);
    const auto nodeType = node->getFinalType();

    if (isNodeTypeContributor(nodeType))
    {
        switch (nodeType)
        {
        case CTrueIR::INode::EFinalType::COrenNayar:
        {
            // should be leaf here, so not going to try children
            // TODO: config type alias from where (also allow aniso)
            // TODO: also might be btdf
            sstr << R"===(
template<>
struct gen_cache<)===" << hashString << R"===(>
{
    using bxdf_t = bxdf::reflection::SOrenNayar<iso_config_t>;
    typename bxdf_t::anisocache_type bxdf_cache;
};
)===";
            break;
        }
        case CTrueIR::INode::EFinalType::CCookTorrance:
        {
            const auto cook_torrance = dynamic_cast<const CTrueIR::CCookTorrance*>(node);

            std::string bxdf_type;
            std::string fresnel_create;
            getCookTorranceBxDFHLSLCode(cook_torrance, ir, bxdf_type, fresnel_create);

            sstr << R"===(
template<>
struct gen_cache<)===" << hashString << R"===(>
{
    using bxdf_t = )===" << bxdf_type << R"===(;
    typename bxdf_t::anisocache_type bxdf_cache;
};
)===";
            break;
        }
        case CTrueIR::INode::EFinalType::CDeltaTransmission:
        {
            sstr << R"===(
template<>
struct gen_cache<)===" << hashString << R"===(>
{
    using bxdf_t = bxdf::transmission::SDeltaDistribution<iso_config_t>;
    typename bxdf_t::anisocache_type bxdf_cache;
};
)===";
        }
        default:
            break;
        }
    }
    else
    {
        sstr << R"===(
template<>
struct gen_cache<)===" << hashString << R"===(>
{
    uint8_t chosenLobe;
)===";
        const auto childCount = node->getChildCount();
        if (childCount)
        {
            for (auto childIx = 0; childIx < childCount; childIx++)
            {
                const auto childHandle = node->getChildHandle(childIx);
                if (const auto child = ir->getObjectPool().deref(childHandle); child)
                {
                    const auto childHash = getHashAs4UintsString(child, ir);
                    sstr << R"===(
    gen_cache<)===" << childHash << R"===(> child)===" << childIx << R"===(;
)===";
                }
            }
        }
        sstr << R"===(
};
)===";
    }
}

void CReferenceUnidirectionalPathTracing::getAlbedoHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const auto* node = nodeInfo.node;
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::albedo()
{
)===";

        if (auto childBrdf = ir->getObjectPool().deref(layer->brdfTop); childBrdf)
        {
            const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
            sstr << "spectral_t brdf = OrientedMaterial<" << childBrdfHash << ">::albedo();\n";
        }
        if (auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission); childBtdf)
        {
            const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
            sstr << "spectral_t btdf = OrientedMaterial<" << childBtdfHash << ">::albedo();\n";
        }

        sstr << R"===(
    return brdf + btdf;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::albedo()
{
)===";

        if (auto childProduct = ir->getObjectPool().deref(sum->product); childProduct)
        {
            const auto childProductHash = getHashAs4UintsString(childProduct, ir);
            sstr << "spectral_t product = OrientedMaterial<" << childProductHash << ">::albedo();\n";
        }
        if (auto childRest = ir->getObjectPool().deref(sum->rest); childRest)
        {
            const auto childRestHash = getHashAs4UintsString(childRest, ir);
            sstr << "spectral_t rest = OrientedMaterial<" << childRestHash << ">::albedo();\n";
        }

        sstr << R"===(
    return product + rest;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CFactorCombiner:
    {
        const auto* combiner = dynamic_cast<const CTrueIR::CFactorCombiner*>(node);
        if (!combiner)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::albedo()
{
)===";

        const auto childCount = combiner->getChildCount();

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                const auto childHash = getHashAs4UintsString(child, ir);
                sstr << "spectral_t child" << static_cast<uint32_t>(i) << " = OrientedMaterial<" << childHash << ">::albedo();\n";
            }
        }

        sstr << "spectral_t retval = ";
        const std::string op = (combiner->getState().type == CTrueIR::CFactorCombiner::Type::Mul) ? " * " : " + ";
        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
            sstr << "child" << static_cast<uint32_t>(i) << (i < childCount - 1 ? op : "");
        sstr << R"===(;
    return retval;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
    {
        const auto* contrib = dynamic_cast<const CTrueIR::CWeightedContributor*>(node);
        if (!contrib)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::albedo()
{
)===";

        if (auto childContrib = ir->getObjectPool().deref(contrib->contributor); childContrib)
        {
            const auto childContribHash = getHashAs4UintsString(childContrib, ir);
            sstr << "spectral_t contributor = OrientedMaterial<" << childContribHash << ">::albedo();\n";
        }
        if (auto childFactor = ir->getObjectPool().deref(contrib->factor); childFactor)
        {
            const auto childFactorHash = getHashAs4UintsString(childFactor, ir);
            sstr << "spectral_t factor = OrientedMaterial<" << childFactorHash << ">::albedo();\n";
        }

        sstr << R"===(
    return contributor * factor;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CCorellatedTransmission*>(node);
        if (!transmission)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::albedo()
{
)===";

        if (auto child = ir->getObjectPool().deref(transmission->btdf); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t btdf = OrientedMaterial<" << childHash << ">::albedo();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->brdfBottom); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t brdf = OrientedMaterial<" << childHash << ">::albedo();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->coated); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t coated = OrientedMaterial<" << childHash << ">::albedo();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->next); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t next = OrientedMaterial<" << childHash << ">::albedo();\n";
        }

        sstr << R"===(
    return btdf + brdf + coated + next;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CSpectralVariable:
    {
        const auto* spectral = dynamic_cast<const CTrueIR::CSpectralVariableFactor*>(node);
        if (!spectral)
            break;

        auto bins = spectral->getSpectralBins();
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::albedo()
{
)===";
        if (bins > 1)
        {
            sstr << "return spectral_t(";
            for (uint8_t i = 0; i < bins; i++)
                sstr << spectral->getParameter(i).scale << (i < bins - 1 ? "," : "");
            sstr << ");\n}\n";
        }
        else
            sstr << "return hlsl::promote<spectral_t>(" << spectral->getParameter(0).scale << ");\n}\n";
        break;
    }
    case CTrueIR::INode::EFinalType::CEmitter:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::COrenNayar:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CCookTorrance:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CBeer:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CFresnel:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CThinInfiniteScatterCorrection:
        [[fallthrough]]
    default:
    {
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::albedo()
{
    return hlsl::promote<spectral_t>(1.0);
}
)===";
    }
    }
}

void CReferenceUnidirectionalPathTracing::getNormalHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const auto* node = nodeInfo.node;
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CContributorSum:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CFactorCombiner:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CWeightedContributor:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CSpectralVariable:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CEmitter:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::COrenNayar:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CCookTorrance:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CBeer:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CFresnel:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CThinInfiniteScatterCorrection:
        [[fallthrough]]
    default:
    {
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::normal(NBL_CONST_REF_ARG(aniso_interaction_t) inter)
{
    return inter.getN();
}
)===";    // return shading normal by default
    }
    }
}

void CReferenceUnidirectionalPathTracing::getAOVThroughputHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const auto* node = nodeInfo.node;
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::aovThroughput()
{
)===";

        if (auto childBrdf = ir->getObjectPool().deref(layer->brdfTop); childBrdf)
        {
            const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
            sstr << "spectral_t brdf = OrientedMaterial<" << childBrdfHash << ">::aovThroughput();\n";
        }
        if (auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission); childBtdf)
        {
            const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
            sstr << "spectral_t btdf = OrientedMaterial<" << childBtdfHash << ">::aovThroughput();\n";
        }

        sstr << R"===(
    return brdf + btdf;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::aovThroughput()
{
)===";

        if (auto childProduct = ir->getObjectPool().deref(sum->product); childProduct)
        {
            const auto childProductHash = getHashAs4UintsString(childProduct, ir);
            sstr << "spectral_t product = OrientedMaterial<" << childProductHash << ">::aovThroughput();\n";
        }
        if (auto childRest = ir->getObjectPool().deref(sum->rest); childRest)
        {
            const auto childRestHash = getHashAs4UintsString(childRest, ir);
            sstr << "spectral_t rest = OrientedMaterial<" << childRestHash << ">::aovThroughput();\n";
        }

        sstr << R"===(
    return product + rest;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CFactorCombiner:
    {
        const auto* combiner = dynamic_cast<const CTrueIR::CFactorCombiner*>(node);
        if (!combiner)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::aovThroughput()
{
)===";

        const auto childCount = combiner->getChildCount();

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                const auto childHash = getHashAs4UintsString(child, ir);
                sstr << "spectral_t child" << static_cast<uint32_t>(i) << " = OrientedMaterial<" << childHash << ">::aovThroughput();\n";
            }
        }

        sstr << "spectral_t retval = ";
        const std::string op = (combiner->getState().type == CTrueIR::CFactorCombiner::Type::Mul) ? " * " : " + ";
        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
            sstr << "child" << static_cast<uint32_t>(i) << (i < childCount - 1 ? op : "");
        sstr << R"===(;
    return retval;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
    {
        const auto* contrib = dynamic_cast<const CTrueIR::CWeightedContributor*>(node);
        if (!contrib)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::aovThroughput()
{
)===";

        if (auto childContrib = ir->getObjectPool().deref(contrib->contributor); childContrib)
        {
            const auto childContribHash = getHashAs4UintsString(childContrib, ir);
            sstr << "spectral_t contributor = OrientedMaterial<" << childContribHash << ">::aovThroughput();\n";
        }
        if (auto childFactor = ir->getObjectPool().deref(contrib->factor); childFactor)
        {
            const auto childFactorHash = getHashAs4UintsString(childFactor, ir);
            sstr << "spectral_t factor = OrientedMaterial<" << childFactorHash << ">::aovThroughput();\n";
        }

        sstr << R"===(
    return contributor * factor;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CCorellatedTransmission*>(node);
        if (!transmission)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::aovThroughput()
{
)===";

        if (auto child = ir->getObjectPool().deref(transmission->btdf); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t btdf = OrientedMaterial<" << childHash << ">::aovThroughput();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->brdfBottom); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t brdf = OrientedMaterial<" << childHash << ">::aovThroughput();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->coated); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t coated = OrientedMaterial<" << childHash << ">::aovThroughput();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->next); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t next = OrientedMaterial<" << childHash << ">::aovThroughput();\n";
        }

        sstr << R"===(
    return btdf + brdf + coated + next;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCookTorrance: // only cook torrance btdf
    {
        const auto* btdf = dynamic_cast<const CTrueIR::CCookTorrance*>(node);
        if (!btdf)
            break;

        const auto transparency = hlsl::pow(btdf->ndfParams.getRougness()[0].scale, 0.001f);
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::aovThroughput()
{
    return )===" << transparency << R"===(;
}
)===";
    }
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CDeltaTransmission*>(node);
        if (!transmission)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::aovThroughput()
{
    return hlsl::promote<spectral_t>(1.0);
}
)===";
    }
    case CTrueIR::INode::EFinalType::CSpectralVariable:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CEmitter:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::COrenNayar:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CBeer:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CFresnel:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CThinInfiniteScatterCorrection:
        [[fallthrough]]
    default:
    {
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::aovThroughput()
{
    return hlsl::promote<spectral_t>(0.0);
}
)===";
    }
    }
}

void CReferenceUnidirectionalPathTracing::getTransparencyHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const auto* node = nodeInfo.node;
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::transparency()
{
)===";

        if (auto childBrdf = ir->getObjectPool().deref(layer->brdfTop); childBrdf)
        {
            const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
            sstr << "scalar_t brdf = OrientedMaterial<" << childBrdfHash << ">::transparency();\n";
        }
        if (auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission); childBtdf)
        {
            const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
            sstr << "scalar_t btdf = OrientedMaterial<" << childBtdfHash << ">::transparency();\n";
        }

        sstr << R"===(
    return brdf + btdf;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::transparency()
{
)===";

        if (auto childProduct = ir->getObjectPool().deref(sum->product); childProduct)
        {
            const auto childProductHash = getHashAs4UintsString(childProduct, ir);
            sstr << "scalar_t product = OrientedMaterial<" << childProductHash << ">::transparency();\n";
        }
        if (auto childRest = ir->getObjectPool().deref(sum->rest); childRest)
        {
            const auto childRestHash = getHashAs4UintsString(childRest, ir);
            sstr << "scalar_t rest = OrientedMaterial<" << childRestHash << ">::transparency();\n";
        }

        sstr << R"===(
    return product + rest;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CFactorCombiner:
    {
        const auto* combiner = dynamic_cast<const CTrueIR::CFactorCombiner*>(node);
        if (!combiner)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::transparency()
{
)===";

        const auto childCount = combiner->getChildCount();

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                const auto childHash = getHashAs4UintsString(child, ir);
                sstr << "scalar_t child" << static_cast<uint32_t>(i) << " = OrientedMaterial<" << childHash << ">::transparency();\n";
            }
        }

        sstr << "scalar_t retval = ";
        const std::string op = (combiner->getState().type == CTrueIR::CFactorCombiner::Type::Mul) ? " * " : " + ";
        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
            sstr << "child" << static_cast<uint32_t>(i) << (i < childCount - 1 ? op : "");
        sstr << R"===(;
    return retval;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
    {
        const auto* contrib = dynamic_cast<const CTrueIR::CWeightedContributor*>(node);
        if (!contrib)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::transparency()
{
)===";

        if (auto childContrib = ir->getObjectPool().deref(contrib->contributor); childContrib)
        {
            const auto childContribHash = getHashAs4UintsString(childContrib, ir);
            sstr << "scalar_t contributor = OrientedMaterial<" << childContribHash << ">::transparency();\n";
        }
        if (auto childFactor = ir->getObjectPool().deref(contrib->factor); childFactor)
        {
            const auto childFactorHash = getHashAs4UintsString(childFactor, ir);
            sstr << "scalar_t factor = OrientedMaterial<" << childFactorHash << ">::transparency();\n";
        }

        sstr << R"===(
    return contributor * factor;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CCorellatedTransmission*>(node);
        if (!transmission)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::transparency()
{
)===";

        if (auto child = ir->getObjectPool().deref(transmission->btdf); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "scalar_t btdf = OrientedMaterial<" << childHash << ">::transparency();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->brdfBottom); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "scalar_t brdf = OrientedMaterial<" << childHash << ">::transparency();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->coated); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "scalar_t coated = OrientedMaterial<" << childHash << ">::transparency();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->next); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "scalar_t next = OrientedMaterial<" << childHash << ">::transparency();\n";
        }

        sstr << R"===(
    return btdf + brdf + coated + next;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCookTorrance: // only cook torrance btdf
    {
        const auto* btdf = dynamic_cast<const CTrueIR::CCookTorrance*>(node);
        if (!btdf)
            break;
        
        const auto transparency = hlsl::pow(btdf->ndfParams.getRougness()[0].scale, 0.001f);
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::transparency()
{
    return )===" << transparency << R"===(;
}
)===";
    }
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CDeltaTransmission*>(node);
        if (!transmission)
            break;
        
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::transparency()
{
    return 1.0;
}
)===";
    }
    case CTrueIR::INode::EFinalType::CSpectralVariable:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CEmitter:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::COrenNayar:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CBeer:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CFresnel:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CThinInfiniteScatterCorrection:
        [[fallthrough]]
    default:
    {
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::transparency()
{
    return 0.0;
}
)===";
    }
    }
}

void CReferenceUnidirectionalPathTracing::getGenerateHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const auto* node = nodeInfo.node;
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    spectral_t lumaContrib = inter.getLuminosityContributionHint();
)===";

        const auto childBrdf = ir->getObjectPool().deref(layer->brdfTop);
        const auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission);
        // TODO: what if either node (or both) not available?

        const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
        const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);

        sstr << R"===(
    scalar_t dummy;
    scalar_t prob = OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 0, dummy);
    if (u.x < prob)
    {
        cache.chosenLobe = 0;
        return OrientedMaterial<)===" << childBrdfHash << R"===(>::generate(inter, xi, xi_extra, cache.child0);
    }
    else
    {
        cache.chosenLobe = 1;
        return OrientedMaterial<)===" << childBtdfHash << R"===(>::generate(inter, xi, xi_extra, cache.child1);
    }
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    spectral_t lumaContrib = inter.getLuminosityContributionHint();
)===";

        const auto childProduct = ir->getObjectPool().deref(sum->product);
        const auto childRest = ir->getObjectPool().deref(sum->rest);
        // TODO: what if either node (or both) not available?

        const auto childProductHash = getHashAs4UintsString(childProduct, ir);
        const auto childRestHash = getHashAs4UintsString(childRest, ir);

        sstr << R"===(
    scalar_t dummy;
    scalar_t prob = OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 0, dummy);
    if (u.x < prob)
    {
        cache.chosenLobe = 0;
        return OrientedMaterial<)===" << childProductHash << R"===(>::generate(inter, xi, xi_extra, cache.child0);
    }
    else
    {
        cache.chosenLobe = 1;
        return OrientedMaterial<)===" << childRestHash << R"===(>::generate(inter, xi, xi_extra, cache.child1);
    }
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CFactorCombiner:
    {
        const auto* combiner = dynamic_cast<const CTrueIR::CFactorCombiner*>(node);
        if (!combiner)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
)===";

        const auto childCount = combiner->getChildCount();

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                const auto childHash = getHashAs4UintsString(child, ir);
                if (i < childCount - 1)
                    sstr << R"===(
    {
        scalar_t dummy;
        scalar_t prob = OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, )===" << static_cast<uint32_t>(i) << R"===(, dummy);
        if (u.x < prob)
        {
            cache.chosenLobe = )===" << static_cast<uint32_t>(i) << R"===(;
            return OrientedMaterial<)===" << childHash << R"===(>::generate(inter, xi, xi_extra, cache.child)===" << static_cast<uint32_t>(i) << R"===();
        }
    }
)===";
                else
                    sstr << R"===(
    {
        cache.chosenLobe = )===" << static_cast<uint32_t>(i) << R"===(;
        return OrientedMaterial<)===" << childHash << R"===(>::generate(inter, xi, xi_extra, cache.child)===" << static_cast<uint32_t>(i) << R"===();
    }
)===";
            }
        }
        sstr << R"===(
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
    {
        const auto* contrib = dynamic_cast<const CTrueIR::CWeightedContributor*>(node);
        if (!contrib)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
)===";

        if (auto child = ir->getObjectPool().deref(contrib->contributor); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << R"===(
    sample_t contrib = OrientedMaterial<)===" << childHash << R"===(>::generate(inter, xi, xi_extra, cache.child0);
)===";
        }

        sstr << R"===(
    return contrib;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CCorellatedTransmission*>(node);
        if (!transmission)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    spectral_t lumaContrib = inter.getLuminosityContributionHint();
)===";

        const auto childBtdf= ir->getObjectPool().deref(transmission->btdf);
        const auto childBottom = ir->getObjectPool().deref(transmission->brdfBottom);
        const auto childCoated= ir->getObjectPool().deref(transmission->coated);
        // TODO: what if some nodes not available?

        const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
        const auto childBottomHash = getHashAs4UintsString(childBottom, ir);
        const auto childCoatedHash = getHashAs4UintsString(childCoated, ir);

        sstr << R"===(
    scalar_t weightBtdf = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childBtdfHash << R"===(>::albedo());
    scalar_t weightBottom = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childBottomHash << R"===(>::albedo());
    scalar_t weightCoated = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childCoatedHash << R"===(>::albedo());
    scalar_t probA = 1.0 / (1.0 + weightBottom / weightBtdf);
    scalar_t probB = 1.0 / (1.0 + (weightCoated + weightBtdf) / weightBottom);
    if (u.x < probA)
    {
        cache.chosenLobe = 0;
        return OrientedMaterial<)===" << childBtdfHash << R"===(>::generate(inter, xi, xi_extra, cache.child0);
    }
    else if (u.y < probB)
    {
        cache.chosenLobe = 1;
        return OrientedMaterial<)===" << childBottomHash << R"===(>::generate(inter, xi, xi_extra, cache.child1);
    }
    else
    {
        cache.chosenLobe = 2;
        return OrientedMaterial<)===" << childCoatedHash << R"===(>::generate(inter, xi, xi_extra, cache.child2);
    }
)===";
        // TODO: next node?
        break;
    }
    case CTrueIR::INode::EFinalType::COrenNayar:
    {
        const auto* oren_nayar = dynamic_cast<const CTrueIR::COrenNayar*>(node);
        if (!oren_nayar)
            break;

        // TODO: config type alias from where
        std::string bxdf_type;
        getOrenNayarBxDFHLSLCode(nodeInfo, ir, bxdf_type);

        const auto hashString = getHashAs4UintsString(node, ir);
        auto roughness = oren_nayar->ndfParams.getRougness();
        sstr << R"===(
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    using oren_nayar_t = )===" << bxdf_type << R"===(;
    using creation_t = typename oren_nayar_t::creation_type;
    creation_t params;
    params.A = )===" << roughness.data()[0].scale << R"===(;
    oren_nayar_t bxdf = oren_nayar_t::create(params);
    sample_t _sample = bxdf.generate(inter, xi, cache.bxdf_cache);
    return _sample;
}
)===";
        // TODO: what to do with child caches?
        break;
    }
    case CTrueIR::INode::EFinalType::CCookTorrance:
    {
        const auto* cook_torrance = dynamic_cast<const CTrueIR::CCookTorrance*>(node);
        if (!cook_torrance)
            break;

        // TODO: config type alias from where

        const auto hashString = getHashAs4UintsString(node, ir);
        const auto roughness = cook_torrance->ndfParams.getRougness();

        std::string bxdf_type;
        std::string fresnel_create;
        getCookTorranceBxDFHLSLCode(nodeInfo, ir, bxdf_type, fresnel_create);

        sstr << R"===(
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    using cook_torrance_t = )===" << bxdf_type << R"===(;
    cook_torrance_t bxdf;
    bxdf.ndf = cook_torrance_t::ndf_type::create()===" << roughness.data()[0].scale << R"===();
)===" << fresnel_create << R"===(
    sample_t _sample = bxdf.generate(inter, xi, cache.bxdf_cache);
    return _sample;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CDeltaTransmission*>(node);
        if (!transmission)
            break;

        // TODO: config type alias from where (also allow aniso)

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    using transmission_t = bxdf::transmission::SDeltaDistribution<iso_config_t>;
    transmission_t bxdf;
    sample_t _sample = bxdf.generate(inter, xi, cache.bxdf_cache);
    return _sample;
}
)===";
        break;
    }
    default:
        break;
    }
}

void CReferenceUnidirectionalPathTracing::getQuotientWeightHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const std::string combineRestWithRetValCode = R"===(
        retval.weight += rest.weight();

        constexpr bool UseMoreDiv = false;
        if constexpr (UseMoreDiv)
        {
            retval.quotient /= scalar_t(1.0) + rest.weight() / chosenPdf;
            retval.quotient += rest.value() / retval.weight();
        }
        else // this branch uses one less div
        {
            const scalar_t rcpChosenPdf = scalar_t(1.0) / chosenPdf;
            retval.quotient += rest.value() * rcpChosenPdf;
            retval.quotient /= scalar_t(1.0) + rest.weight() * rcpChosenPdf;
        }

        // correct for the fact that the `choiceProb` are not normalized
        retval.quotient *= choiceSum;
        retval.pdf /= choiceSum;
    }

    return retval;
)===";

    const auto* node = nodeInfo.node;
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        // TODO
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static quotient_weight_t OrientedMaterial<)===" << hashString << R"===(>::quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    uint8_t chosenLobe = child.chosenLobe;
    scalar_t choiceSum;
    scalar_t choiceProb = OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, chosenLobe, choiceSum);
)===";

        const auto childBrdf = ir->getObjectPool().deref(layer->brdfTop);
        const auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission);
        // TODO: what if either node (or both) not available?

        const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
        const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);

        sstr << R"===(
    quotient_weight_t retval;
    if (chosenLobe == 0)
        retval = OrientedMaterial<)===" << childBrdfHash << R"===(>::quotientAndWeight(_sample, inter, cache.child0);
    else
        retval = OrientedMaterial<)===" << childBtdfHash << R"===(>::quotientAndWeight(_sample, inter, cache.child1);

    if (retval.weight() < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity))
    {
        const scalar_t chosenPdf = retval.weight() * choiceProb;
        value_weight_t rest = value_weight_t::create(0.0, 0.0);
        value_weight_t other;

        if (chosenLobe == 0)
        {
            other = OrientedMaterial<)===" << childBtdfHash << R"===(>::evalAndWeight(_sample, inter);
            rest.value = other.value();
            if (OrientedMaterial<)===" << childBtdfHash << R"===(>::canGenerate())
                rest.weight = other.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 1, choiceSum);
        }
        else
        {
            other = OrientedMaterial<)===" << childBrdfHash << R"===(>::evalAndWeight(_sample, inter);
            rest.value = other.value();
            if (OrientedMaterial<)===" << childBrdfHash << R"===(>::canGenerate())
                rest.weight = other.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 0, choiceSum);
        }
)===" << combineRestWithRetValCode;
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static quotient_weight_t OrientedMaterial<)===" << hashString << R"===(>::quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    uint8_t chosenLobe = cache.chosenLobe;
    scalar_t choiceSum;
    scalar_t choiceProb = OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, chosenLobe, choiceSum);
)===";

        const auto childProduct = ir->getObjectPool().deref(sum->product);
        const auto childRest = ir->getObjectPool().deref(sum->rest);
        // TODO: what if either node (or both) not available?

        const auto childProductHash = getHashAs4UintsString(childProduct, ir);
        const auto childRestHash = getHashAs4UintsString(childRest, ir);

        sstr << R"===(
    quotient_weight_t retval;
    if (chosenLobe == 0)
        retval = OrientedMaterial<)===" << childProductHash << R"===(>::quotientAndWeight(_sample, inter, cache.child0);
    else
        retval = OrientedMaterial<)===" << childRestHash << R"===(>::quotientAndWeight(_sample, inter, cache.child1);

    if (retval.weight() < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity))
    {
        const scalar_t chosenPdf = retval.weight() * choiceProb;
        value_weight_t rest = value_weight_t::create(0.0, 0.0);
        value_weight_t other;

        if (chosenLobe == 0)
        {
            other = OrientedMaterial<)===" << childRestHash << R"===(>::evalAndWeight(_sample, inter);
            rest.value = other.value();
            if (OrientedMaterial<)===" << childRestHash << R"===(>::canGenerate())
                rest.weight = other.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 1, choiceSum);
        }
        else
        {
            other = OrientedMaterial<)===" << childProductHash << R"===(>::evalAndWeight(_sample, inter);
            rest.value = other.value();
            if (OrientedMaterial<)===" << childProductHash << R"===(>::canGenerate())
                rest.weight = other.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 0, choiceSum);
        }
)===" << combineRestWithRetValCode;
        break;
    }
    case CTrueIR::INode::EFinalType::CFactorCombiner:
    {
        const auto* combiner = dynamic_cast<const CTrueIR::CFactorCombiner*>(node);
        if (!combiner)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static quotient_weight_t OrientedMaterial<)===" << hashString << R"===(>::quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    uint8_t chosenLobe = cache.chosenLobe;
    scalar_t choiceSum;
    scalar_t choiceProb = OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, chosenLobe, choiceSum);

    quotient_weight_t retval;
)===";

        const auto childCount = combiner->getChildCount();

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                const auto childHash = getHashAs4UintsString(child, ir);
                if (i < childCount - 1)
                    sstr << R"===(
    if (chosenLobe ==  )===" << static_cast<uint32_t>(i) << R"===(
        retval = OrientedMaterial<)===" << childHash << R"===(>::quotientAndWeight(_sample, inter, cache.child)===" << static_cast<uint32_t>(i) << R"===();
)===";
                else
                    sstr << R"===(
    else
        retval = OrientedMaterial<)===" << childHash << R"===(>::quotientAndWeight(_sample, inter, cache.child)===" << static_cast<uint32_t>(i) << R"===();
)===";
            }
        }

        sstr << R"===(
    if (retval.weight() < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity))
    {
        const scalar_t chosenPdf = retval.weight() * choiceProb;
        value_weight_t rest = value_weight_t::create(0.0,0.0);
)===";

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                const auto childHash = getHashAs4UintsString(child, ir);
                sstr << R"===(
        if (chosenLobe !=  )===" << static_cast<uint32_t>(i) << R"===(
        {
            value_weight_t child = OrientedMaterial<)===" << childHash << R"===(>::quotientAndWeight(_sample, inter, cache.child)===" << static_cast<uint32_t>(i) << R"===();
            if (OrientedMaterial<)===" << childHash << R"===(>::canGenerate())
                rest.weight += child.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, )===" << static_cast<uint32_t>(i) << R"===(, choiceSum);
        }
)===";
            }
        }

        sstr << combineRestWithRetValCode;
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
    {
        const auto* contrib = dynamic_cast<const CTrueIR::CWeightedContributor*>(node);
        if (!contrib)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static quotient_weight_t OrientedMaterial<)===" << hashString << R"===(>::quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
)===";

        if (auto child = ir->getObjectPool().deref(contrib->contributor); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << R"===(
    quotient_weight_t contrib = OrientedMaterial<)===" << childHash << R"===(>::quotientAndWeight(_sample, inter, cache.child0);
)===";
        }

        if (auto child = ir->getObjectPool().deref(contrib->factor); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << R"===(
    gen_cache<)===" << childHash << R"===(> factor_cache;
    quotient_weight_t factor = OrientedMaterial<)===" << childHash << R"===(>::quotientAndWeight(_sample, inter, factor_cache);
    bool factorHasWeight = OrientedMaterial<)===" << childHash << R"===(>::canGenerate();
)===";
        }

        sstr << R"===(
    contrib.quotient *= factor.quotient();
    if (factorHasWeight)
        contrib.weight *= factor.weight();
    return contrib;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CCorellatedTransmission*>(node);
        if (!transmission)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static quotient_weight_t OrientedMaterial<)===" << hashString << R"===(>::quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    uint8_t chosenLobe = cache.chosenLobe;
    scalar_t choiceSum;
    scalar_t choiceProb = OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, chosenLobe, choiceSum);
)===";

        const auto childBtdf = ir->getObjectPool().deref(transmission->btdf);
        const auto childBottom = ir->getObjectPool().deref(transmission->brdfBottom);
        const auto childCoated = ir->getObjectPool().deref(transmission->coated);
        // TODO: what if some nodes not available?

        const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
        const auto childBottomHash = getHashAs4UintsString(childBottom, ir);
        const auto childCoatedHash = getHashAs4UintsString(childCoated, ir);
        
        sstr << R"===(
    quotient_weight_t retval;
    if (chosenLobe == 0)
        retval = OrientedMaterial<)===" << childBtdfHash << R"===(>::quotientAndWeight(_sample, inter, cache.child0);
    else if (chosenLobe == 1)
        retval = OrientedMaterial<)===" << childBottomHash << R"===(>::quotientAndWeight(_sample, inter, cache.child1);
    else
        retval = OrientedMaterial<)===" << childCoatedHash << R"===(>::quotientAndWeight(_sample, inter, cache.child2);

    if (retval.weight() < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity))
    {
        const scalar_t chosenPdf = retval.weight() * choiceProb;
        value_weight_t rest = value_weight_t::create(0.0,0.0);

        if (chosenLobe != 0)
        {
            value_weight_t child = OrientedMaterial<)===" << childBtdfHash << R"===(>::evalAndWeight(_sample, inter);
            rest.value += child.value();
            if (OrientedMaterial<)===" << childBtdfHash << R"===(>::canGenerate())
                rest.weight += child.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 0, choiceSum);
        }
        if (chosenLobe != 1)
        {
            value_weight_t child = OrientedMaterial<)===" << childBottomHash << R"===(>::evalAndWeight(_sample, inter);
            rest.value += child.value();
            if (OrientedMaterial<)===" << childBottomHash << R"===(>::canGenerate())
                rest.weight += child.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 1, choiceSum);
        }
        if (chosenLobe != 2)
        {
            value_weight_t child = OrientedMaterial<)===" << childCoatedHash << R"===(>::evalAndWeight(_sample, inter);
            rest.value += child.value();
            if (OrientedMaterial<)===" << childCoatedHash << R"===(>::canGenerate())
                rest.weight += child.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 2, choiceSum);
        }
)===" << combineRestWithRetValCode;
        // TODO: next node?
        break;
    }
    case CTrueIR::INode::EFinalType::COrenNayar:
    {
        const auto* oren_nayar = dynamic_cast<const CTrueIR::COrenNayar*>(node);
        if (!oren_nayar)
            break;

        // TODO: config type alias from where
        std::string bxdf_type;
        getOrenNayarBxDFHLSLCode(nodeInfo, ir, bxdf_type);

        const auto hashString = getHashAs4UintsString(node, ir);
        auto roughness = oren_nayar->ndfParams.getRougness();
        sstr << R"===(
static quotient_weight_t OrientedMaterial<)===" << hashString << R"===(>::quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    using oren_nayar_t = )===" << bxdf_type << R"===(;
    using creation_t = typename oren_nayar_t::creation_type;
    creation_t params;
    params.A = )===" << roughness.data()[0].scale << R"===(;
    oren_nayar_t bxdf = oren_nayar_t::create(params);
    quotient_weight_t quo = bxdf.quotientAndWeight(_sample, inter, cache.bxdf_cache);
    return quo;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCookTorrance:
    {
        const auto* cook_torrance = dynamic_cast<const CTrueIR::CCookTorrance*>(node);
        if (!cook_torrance)
            break;

        // TODO: config type alias from where

        const auto hashString = getHashAs4UintsString(node, ir);
        const auto roughness = cook_torrance->ndfParams.getRougness();

        std::string bxdf_type;
        std::string fresnel_create;
        getCookTorranceBxDFHLSLCode(nodeInfo, ir, bxdf_type, fresnel_create);

        sstr << R"===(
static quotient_weight_t OrientedMaterial<)===" << hashString << R"===(>::quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    using cook_torrance_t = )===" << bxdf_type << R"===(;
    cook_torrance_t bxdf;
    bxdf.ndf = cook_torrance_t::ndf_type::create()===" << roughness.data()[0].scale << R"===();
)===" << fresnel_create << R"===(
    quotient_weight_t quo = bxdf.quotientAndWeight(_sample, inter, cache.bxdf_cache);
    return quo;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CDeltaTransmission*>(node);
        if (!transmission)
            break;

        // TODO: config type alias from where (also allow aniso)

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static quotient_weight_t OrientedMaterial<)===" << hashString << R"===(>::quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache)
{
    using transmission_t = bxdf::transmission::SDeltaDistribution<iso_config_t>;
    transmission_t bxdf;
    quotient_weight_t quo = bxdf.quotientAndWeight(_sample, inter, cache.bxdf_cache);
    return quo;
}
)===";
        break;
    }
    default:
        break;
    }
}

void CReferenceUnidirectionalPathTracing::getEvalWeightHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const auto* node = nodeInfo.node;
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static value_weight_t OrientedMaterial<)===" << hashString << R"===(>::evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter)
{ 
)===";

        const auto childBrdf = ir->getObjectPool().deref(layer->brdfTop);
        const auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission);
        // TODO: what if either node (or both) not available?

        const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
        const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);

        // TODO defensive sampler

        sstr << R"===(
    value_weight_t retval = value_weight_t::create(0.0,0.0);
    scalar_t targetSum;
    {
        value_weight_t lobe = OrientedMaterial<)===" << childBrdfHash << R"===(>::evalAndWeight(_sample, inter);
        retval.value += lobe.value();
        if (OrientedMaterial<)===" << childBrdfHash << R"===(>::canGenerate())
            retval.weight += lobe.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 0, targetSum);
    }
    {
        retval = OrientedMaterial<)===" << childBtdfHash << R"===(>::evalAndWeight(_sample, inter);
        retval.value += lobe.value();
        if (OrientedMaterial<)===" << childBtdfHash << R"===(>::canGenerate())
            retval.weight += lobe.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 1, targetSum);
    }

    retval.weight /= targetSum;
    return retval;
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static value_weight_t OrientedMaterial<)===" << hashString << R"===(>::evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter)
{
)===";

        const auto childProduct = ir->getObjectPool().deref(sum->product);
        const auto childRest = ir->getObjectPool().deref(sum->rest);
        // TODO: what if either node (or both) not available?

        const auto childProductHash = getHashAs4UintsString(childProduct, ir);
        const auto childRestHash = getHashAs4UintsString(childRest, ir);

        sstr << R"===(
    value_weight_t retval = value_weight_t::create(0.0,0.0);
    scalar_t targetSum;
    {
        value_weight_t lobe = OrientedMaterial<)===" << childProductHash << R"===(>::evalAndWeight(_sample, inter);
        retval.value += lobe.value();
        if (OrientedMaterial<)===" << childProductHash << R"===(>::canGenerate())
            retval.weight += lobe.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 0, targetSum);
    }
    {
        value_weight_t lobe = OrientedMaterial<)===" << childRestHash << R"===(>::evalAndWeight(_sample, inter);
        retval.value += lobe.value();
        if (OrientedMaterial<)===" << childRestHash << R"===(>::canGenerate())
            retval.weight += lobe.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 1, targetSum);
    }

    retval.weight /= targetSum;
    return retval;
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CFactorCombiner:
    {
        const auto* combiner = dynamic_cast<const CTrueIR::CFactorCombiner*>(node);
        if (!combiner)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static value_weight_t OrientedMaterial<)===" << hashString << R"===(>::evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter)
{
    value_weight_t retval = value_weight_t::create(0.0,0.0);
    scalar_t targetSum;
)===";

        const auto childCount = combiner->getChildCount();

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                const auto childHash = getHashAs4UintsString(child, ir);
                    sstr << R"===(
    {
        value_weight_t lobe = OrientedMaterial<)===" << childHash << R"===(>::evalAndWeight(_sample, inter);
        retval.value += lobe.value();
        if (OrientedMaterial<)===" << childHash << R"===(>::canGenerate())
            retval.weight += lobe.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, )===" << static_cast<uint32_t>(i) << R"===(, targetSum);
    }
)===";
            }
        }

        sstr << R"===(
    retval.weight /= targetSum;
    return retval;
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
    {
        const auto* contrib = dynamic_cast<const CTrueIR::CWeightedContributor*>(node);
        if (!contrib)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static value_weight_t OrientedMaterial<)===" << hashString << R"===(>::evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter)
{
)===";

        if (auto child = ir->getObjectPool().deref(contrib->contributor); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << R"===(
    value_weight_t contrib = OrientedMaterial<)===" << childHash << R"===(>::evalAndWeight(_sample, inter);
)===";
        }

        if (auto child = ir->getObjectPool().deref(contrib->factor); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << R"===(
    value_weight_t factor = OrientedMaterial<)===" << childHash << R"===(>::evalAndWeight(_sample, inter);
    bool factorHasWeight = OrientedMaterial<)===" << childHash << R"===(>::canGenerate();
)===";
        }

        sstr << R"===(
    contrib.quotient *= factor.quotient();
    if (factorHasWeight)
        contrib.weight *= factor.weight();
    return contrib;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CCorellatedTransmission*>(node);
        if (!transmission)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static value_weight_t OrientedMaterial<)===" << hashString << R"===(>::evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter)
{
)===";

        const auto childBtdf = ir->getObjectPool().deref(transmission->btdf);
        const auto childBottom = ir->getObjectPool().deref(transmission->brdfBottom);
        const auto childCoated = ir->getObjectPool().deref(transmission->coated);
        // TODO: what if some nodes not available?

        const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
        const auto childBottomHash = getHashAs4UintsString(childBottom, ir);
        const auto childCoatedHash = getHashAs4UintsString(childCoated, ir);

        sstr << R"===(
    value_weight_t retval = value_weight_t::create(0.0,0.0);
    scalar_t targetSum;
    {
        value_weight_t lobe = OrientedMaterial<)===" << childBtdfHash << R"===(>::evalAndWeight(_sample, inter);
        retval.value += lobe.value();
        if (OrientedMaterial<)===" << childBtdfHash << R"===(>::canGenerate())
            retval.weight += lobe.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 0, targetSum);
    }
    {
        value_weight_t lobe = OrientedMaterial<)===" << childBottomHash << R"===(>::evalAndWeight(_sample, inter);
        retval.value += lobe.value();
        if (OrientedMaterial<)===" << childBottomHash << R"===(>::canGenerate())
            retval.weight += lobe.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 1, targetSum);
    }
    {
        value_weight_t lobe = OrientedMaterial<)===" << childCoatedHash << R"===(>::evalAndWeight(_sample, inter);
        retval.value += lobe.value();
        if (OrientedMaterial<)===" << childCoatedHash << R"===(>::canGenerate())
            retval.weight += lobe.weight() * OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(inter, 2, targetSum);
    }

    retval.weight /= targetSum;
    return retval;
)===";
        // TODO: next node?
        break;
    }
    case CTrueIR::INode::EFinalType::COrenNayar:
    {
        const auto* oren_nayar = dynamic_cast<const CTrueIR::COrenNayar*>(node);
        if (!oren_nayar)
            break;

        // TODO: config type alias from where

        std::string bxdf_type;
        getOrenNayarBxDFHLSLCode(nodeInfo, ir, bxdf_type);

        const auto hashString = getHashAs4UintsString(node, ir);
        auto roughness = oren_nayar->ndfParams.getRougness();
        sstr << R"===(
static value_weight_t OrientedMaterial<)===" << hashString << R"===(>::evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter)
{
    using oren_nayar_t = )===" << bxdf_type << R"===(;
    using creation_t = typename oren_nayar_t::creation_type;
    creation_t params;
    params.A = )===" << roughness.data()[0].scale << R"===(;
    oren_nayar_t bxdf = oren_nayar_t::create(params);
    quotient_weight_t quo = bxdf.evalAndWeight(_sample, inter);
    return quo;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCookTorrance:
    {
        const auto* cook_torrance = dynamic_cast<const CTrueIR::CCookTorrance*>(node);
        if (!cook_torrance)
            break;

        // TODO: config type alias from where

        const auto hashString = getHashAs4UintsString(node, ir);
        const auto roughness = cook_torrance->ndfParams.getRougness();

        std::string bxdf_type;
        std::string fresnel_create;
        getCookTorranceBxDFHLSLCode(nodeInfo, ir, bxdf_type, fresnel_create);

        sstr << R"===(
static value_weight_t OrientedMaterial<)===" << hashString << R"===(>::evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter)
{
    using cook_torrance_t = )===" << bxdf_type << R"===(;
    cook_torrance_t bxdf;
    bxdf.ndf = cook_torrance_t::ndf_type::create()===" << roughness.data()[0].scale << R"===();
)===" << fresnel_create << R"===(
    value_weight_t quo = bxdf.evalAndWeight(_sample, inter);
    return quo;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CDeltaTransmission*>(node);
        if (!transmission)
            break;

        // TODO: config type alias from where (also allow aniso)

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static value_weight_t OrientedMaterial<)===" << hashString << R"===(>::evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter)
{
    using transmission_t = bxdf::transmission::SDeltaDistribution<iso_config_t>;
    transmission_t bxdf;
    value_weight_t quo = bxdf.evalAndWeight(_sample, inter);
    return quo;
}
)===";
        break;
    }
    default:
        break;
    }
}

void CReferenceUnidirectionalPathTracing::getEmissionHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const auto* node = nodeInfo.node;
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::emission()
{
)===";

        if (auto childBrdf = ir->getObjectPool().deref(layer->brdfTop); childBrdf)
        {
            const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
            sstr << "spectral_t brdf = OrientedMaterial<" << childBrdfHash << ">::emission();\n";
        }
        if (auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission); childBtdf)
        {
            const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
            sstr << "spectral_t btdf = OrientedMaterial<" << childBtdfHash << ">::emission();\n";
        }

        sstr << R"===(
    return brdf + btdf;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::emission()
{
)===";

        if (auto childProduct = ir->getObjectPool().deref(sum->product); childProduct)
        {
            const auto childProductHash = getHashAs4UintsString(childProduct, ir);
            sstr << "spectral_t product = OrientedMaterial<" << childProductHash << ">::emission();\n";
        }
        if (auto childRest = ir->getObjectPool().deref(sum->rest); childRest)
        {
            const auto childRestHash = getHashAs4UintsString(childRest, ir);
            sstr << "spectral_t rest = OrientedMaterial<" << childRestHash << ">::emission();\n";
        }

        sstr << R"===(
    return product + rest;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CFactorCombiner:
    {
        const auto* combiner = dynamic_cast<const CTrueIR::CFactorCombiner*>(node);
        if (!combiner)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::emission()
{
)===";

        const auto childCount = combiner->getChildCount();

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                const auto childHash = getHashAs4UintsString(child, ir);
                sstr << "spectral_t child" << static_cast<uint32_t>(i) << " = OrientedMaterial<" << childHash << ">::emission();\n";
            }
        }

        sstr << "spectral_t retval = ";
        const std::string op = (combiner->getState().type == CTrueIR::CFactorCombiner::Type::Mul) ? " * " : " + ";
        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
            sstr << "child" << static_cast<uint32_t>(i) << (i < childCount - 1 ? op : "");
        sstr << R"===(;
    return retval;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
    {
        const auto* contrib = dynamic_cast<const CTrueIR::CWeightedContributor*>(node);
        if (!contrib)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::emission()
{
)===";

        if (auto childContrib = ir->getObjectPool().deref(contrib->contributor); childContrib)
        {
            const auto childContribHash = getHashAs4UintsString(childContrib, ir);
            sstr << "spectral_t contributor = OrientedMaterial<" << childContribHash << ">::emission();\n";
        }
        if (auto childFactor = ir->getObjectPool().deref(contrib->factor); childFactor)
        {
            const auto childFactorHash = getHashAs4UintsString(childFactor, ir);
            sstr << "spectral_t factor = OrientedMaterial<" << childFactorHash << ">::emission();\n";
        }

        sstr << R"===(
    return contributor * factor;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CCorellatedTransmission*>(node);
        if (!transmission)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static spectral_t OrientedMaterial<)===" << hashString << R"===(>::emission()
{
)===";

        if (auto child = ir->getObjectPool().deref(transmission->btdf); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t btdf = OrientedMaterial<" << childHash << ">::emission();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->brdfBottom); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t brdf = OrientedMaterial<" << childHash << ">::emission();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->coated); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t coated = OrientedMaterial<" << childHash << ">::emission();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->next); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "spectral_t next = OrientedMaterial<" << childHash << ">::emission();\n";
        }

        sstr << R"===(
    return btdf + brdf + coated + next;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CEmitter:
    {
        const auto* emitter = dynamic_cast<const CTrueIR::CEmitter*>(node);
        if (!emitter)
            break;

        const auto strength = emitter->profile.scale;   // TODO: how to get emission color like from IES?
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::emission()
{
    return hlsl::promote<spectral_t>()===" << strength << R"===();
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCookTorrance:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CSpectralVariable:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::COrenNayar:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CBeer:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CFresnel:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CThinInfiniteScatterCorrection:
        [[fallthrough]]
    default:
    {
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::emission()
{
    return hlsl::promote<spectral_t>(0.0);
}
)===";
    }
    }
}

void CReferenceUnidirectionalPathTracing::getCanGenerateHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const auto* node = nodeInfo.node;
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static bool OrientedMaterial<)===" << hashString << R"===(>::canGenerate()
{
)===";

        if (auto childBrdf = ir->getObjectPool().deref(layer->brdfTop); childBrdf)
        {
            const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
            sstr << "bool brdf = OrientedMaterial<" << childBrdfHash << ">::canGenerate();\n";
        }
        if (auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission); childBtdf)
        {
            const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
            sstr << "bool btdf = OrientedMaterial<" << childBtdfHash << ">::canGenerate();\n";
        }

        sstr << R"===(
    return brdf || btdf;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static bool OrientedMaterial<)===" << hashString << R"===(>::canGenerate()
{
)===";

        if (auto childProduct = ir->getObjectPool().deref(sum->product); childProduct)
        {
            const auto childProductHash = getHashAs4UintsString(childProduct, ir);
            sstr << "bool product = OrientedMaterial<" << childProductHash << ">::canGenerate();\n";
        }
        if (auto childRest = ir->getObjectPool().deref(sum->rest); childRest)
        {
            const auto childRestHash = getHashAs4UintsString(childRest, ir);
            sstr << "bool rest = OrientedMaterial<" << childRestHash << ">::canGenerate();\n";
        }

        sstr << R"===(
    return product || rest;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CFactorCombiner:
    {
        const auto* combiner = dynamic_cast<const CTrueIR::CFactorCombiner*>(node);
        if (!combiner)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static bool OrientedMaterial<)===" << hashString << R"===(>::canGenerate()
{
)===";

        const auto childCount = combiner->getChildCount();

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                const auto childHash = getHashAs4UintsString(child, ir);
                sstr << "bool child" << static_cast<uint32_t>(i) << " = OrientedMaterial<" << childHash << ">::canGenerate();\n";
            }
        }

        sstr << "bool retval = ";
        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
            sstr << "child" << static_cast<uint32_t>(i) << (i < childCount - 1 ? " || " : "");
        sstr << R"===(;
    return retval;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
    {
        const auto* contrib = dynamic_cast<const CTrueIR::CWeightedContributor*>(node);
        if (!contrib)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static bool OrientedMaterial<)===" << hashString << R"===(>::canGenerate()
{
)===";

        if (auto childContrib = ir->getObjectPool().deref(contrib->contributor); childContrib)
        {
            const auto childContribHash = getHashAs4UintsString(childContrib, ir);
            sstr << "bool contributor = OrientedMaterial<" << childContribHash << ">::canGenerate();\n";
        }
        if (auto childFactor = ir->getObjectPool().deref(contrib->factor); childFactor)
        {
            const auto childFactorHash = getHashAs4UintsString(childFactor, ir);
            sstr << "bool factor = OrientedMaterial<" << childFactorHash << ">::canGenerate();\n";
        }

        sstr << R"===(
    return contributor || factor;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CCorellatedTransmission*>(node);
        if (!transmission)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static bool OrientedMaterial<)===" << hashString << R"===(>::canGenerate()
{
)===";

        if (auto child = ir->getObjectPool().deref(transmission->btdf); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "bool btdf = OrientedMaterial<" << childHash << ">::canGenerate();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->brdfBottom); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "bool brdf = OrientedMaterial<" << childHash << ">::canGenerate();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->coated); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "bool coated = OrientedMaterial<" << childHash << ">::canGenerate();\n";
        }
        if (auto child = ir->getObjectPool().deref(transmission->next); child)
        {
            const auto childHash = getHashAs4UintsString(child, ir);
            sstr << "bool next = OrientedMaterial<" << childHash << ">::canGenerate();\n";
        }

        sstr << R"===(
    return btdf || brdf || coated || next;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::COrenNayar:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CCookTorrance:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
    {
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static bool OrientedMaterial<)===" << hashString << R"===(>::canGenerate()
{
    return true;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CSpectralVariable:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CEmitter:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CBeer:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CFresnel:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CThinInfiniteScatterCorrection:
        [[fallthrough]]
    default:
    {
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static bool OrientedMaterial<)===" << hashString << R"===(>::canGenerate()
{
    return false;
}
)===";
    }
    }
}

void CReferenceUnidirectionalPathTracing::getChoiceTargetHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir)
{
    const auto* node = nodeInfo.node;
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(NBL_CONST_REF_ARG(aniso_interaction_t) inter, uint8_t chosenLobe, NBL_REF_ARG(scalar_t) choiceSum)
{
    spectral_t lumaContrib = inter.getLuminosityContributionHint();
)===";

        const auto childBrdf = ir->getObjectPool().deref(layer->brdfTop);
        const auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission);
        // TODO: what if either node (or both) not available?

        const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
        const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);

        sstr << R"===(
    scalar_t weightBrdf = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childBrdfHash << R"===(>::albedo());
    scalar_t weightBtdf = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childBtdfHash << R"===(>::albedo());
    choiceSum = weightBrdf + weightBtdf;
    if (chosenLobe == 0)
        return 1.0 / (1.0 + weightBtdf / weightBrdf);
    else
        return 1.0 / (1.0 + weightBrdf / weightBtdf);
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(NBL_CONST_REF_ARG(aniso_interaction_t) inter, uint8_t chosenLobe, NBL_REF_ARG(scalar_t) choiceSum)
{
    spectral_t lumaContrib = inter.getLuminosityContributionHint();
)===";

        const auto childProduct = ir->getObjectPool().deref(sum->product);
        const auto childRest = ir->getObjectPool().deref(sum->rest);
        // TODO: what if either node (or both) not available?

        const auto childProductHash = getHashAs4UintsString(childProduct, ir);
        const auto childRestHash = getHashAs4UintsString(childRest, ir);

        sstr << R"===(
    scalar_t weightProduct = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childProductHash << R"===(>::albedo());
    scalar_t weightRest = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childRestHash << R"===(>::albedo());
    choiceSum = weightProduct + weightRest;
    if (chosenLobe == 0)
        return 1.0 / (1.0 + weightRest / weightProduct);
    else
        return 1.0 / (1.0 + weightProduct / weightRest);
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CFactorCombiner:
    {
        const auto* combiner = dynamic_cast<const CTrueIR::CFactorCombiner*>(node);
        if (!combiner)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(NBL_CONST_REF_ARG(aniso_interaction_t) inter, uint8_t chosenLobe, NBL_REF_ARG(scalar_t) choiceSum)
{
    spectral_t lumaContrib = inter.getLuminosityContributionHint();
)===";

        const auto childCount = combiner->getChildCount();

        for (uint8_t i = 0; i < childCount; i++)
        {
            if (auto child = ir->getObjectPool().deref(combiner->getChildHandle(i)); child)
            {
                const auto childHash = getHashAs4UintsString(child, ir);
                sstr << R"===(
    scalar_t weightChild)===" << static_cast<uint32_t>(i) << R"===( = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childHash << R"===(>::albedo());
)===";
            }
        }

        sstr << "choiceSum = ";
        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
            sstr << "weightChild" << static_cast<uint32_t>(i) << (i < childCount - 1 ? " + " : "");

        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
        {
            if (i < childCount - 1)
                sstr << R"===(
    if (chosenLobe == )===" << static_cast<uint32_t>(i) << R"===()
        return weightChild)===" << static_cast<uint32_t>(i) << R"===( / choiceSum;
)===";
            else
                sstr << R"===(
    else
        return weightChild)===" << static_cast<uint32_t>(i) << R"===( / choiceSum;
)===";
        }
      
        sstr << R"===(
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        const auto* transmission = dynamic_cast<const CTrueIR::CCorellatedTransmission*>(node);
        if (!transmission)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static scalar_t OrientedMaterial<)===" << hashString << R"===(>::choiceTarget(NBL_CONST_REF_ARG(aniso_interaction_t) inter, uint8_t chosenLobe, NBL_REF_ARG(scalar_t) choiceSum)
{
    spectral_t lumaContrib = inter.getLuminosityContributionHint();
)===";

        const auto childBtdf = ir->getObjectPool().deref(transmission->btdf);
        const auto childBottom = ir->getObjectPool().deref(transmission->brdfBottom);
        const auto childCoated = ir->getObjectPool().deref(transmission->coated);
        // TODO: what if some nodes not available?

        const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
        const auto childBottomHash = getHashAs4UintsString(childBottom, ir);
        const auto childCoatedHash = getHashAs4UintsString(childCoated, ir);

        sstr << R"===(
    scalar_t weightBtdf = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childBtdfHash << R"===(>::albedo());
    scalar_t weightBottom = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childBottomHash << R"===(>::albedo());
    scalar_t weightCoated = hlsl::dot(lumaContrib, OrientedMaterial<)===" << childCoatedHash << R"===(>::albedo());
    choiceSum = weightBtdf + weightBottom + weightCoated;
    if (chosenLobe == 0)
        return 1.0 / (1.0 + (weightCoated + weightBottom) / weightBtdf);
    else if (chosenLobe == 1)
        return 1.0 / (1.0 + (weightCoated + weightBtdf) / weightBottom);
    else
        return  1.0 / (1.0 + (weightBtdf + weightBottom) / weightCoated);
}
)===";
        // TODO: next node?
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::COrenNayar:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CCookTorrance:
        [[fallthrough]]
    case CTrueIR::INode::EFinalType::CDeltaTransmission:
        [[fallthrough]]
    default:
        break;
    }
}

void CReferenceUnidirectionalPathTracing::getOrenNayarBxDFHLSLCode(const TraversalNodeInfo& nodeInfo, const CTrueIR* ir, std::string& bxdf_type)
{
    const auto* oren_nayar = dynamic_cast<const CTrueIR::COrenNayar*>(nodeInfo.node);

    // config can be iso or aniso (doesn't really matter for oren nayar)
    std::string config = "aniso_config_t";
    if (oren_nayar->ndfParams.definitelyIsotropic())
        config = "iso_config_t";

    if (nodeInfo.isTransmission)
        bxdf_type = "bxdf::transmission::SOrenNayar<" + config + ">";
    else
        bxdf_type = "bxdf::reflection::SOrenNayar<" + config + ">";
}

void CReferenceUnidirectionalPathTracing::getCookTorranceBxDFHLSLCode(const TraversalNodeInfo& nodeInfo, const CTrueIR* ir,
                                                                      std::string& bxdf_type, std::string& fresnel_create)
{
    const auto* cook_torrance = dynamic_cast<const CTrueIR::CCookTorrance*>(nodeInfo.node);
    if (nodeInfo.isTransmission)
    {
        assert(cook_torrance->orientedRealEta); // should exist/not be null
        // btdf
        if (cook_torrance->ndfParams.definitelyIsotropic())
        {
            if (cook_torrance->ndfParams.getDistribution() == CTrueIR::SBasicNDFParams::EDistribution::GGX)
                bxdf_type = "bxdf::transmission::SGGXDielectricIsotropic<aniso_microfacet_config_t>";
            else
                bxdf_type = "bxdf::transmission::SBeckmannDielectricIsotropic<aniso_microfacet_config_t>";
        }
        else
        {
            if (cook_torrance->ndfParams.getDistribution() == CTrueIR::SBasicNDFParams::EDistribution::GGX)
                bxdf_type = "bxdf::transmission::SGGXDielectricAnisotropic<iso_microfacet_config_t>";
            else
                bxdf_type = "bxdf::transmission::SBeckmannDielectricAnisotropic<iso_microfacet_config_t>";
        }

        const auto eta = ir->getObjectPool().deref(cook_torrance->orientedRealEta)->getParameter(0).scale;
        fresnel_create = R"===(
    bxdf::fresnel::OrientedEtas<typename cook_torrance_t::monochrome_type> orientedEta = bxdf::fresnel::OrientedEtas<typename cook_torrance_t::monochrome_type>::create(1.0, hlsl::promote<typename cook_torrance_t::monochrome_type>()===" + std::to_string(eta) + R"===());
    bxdf.fresnel = cook_torrance_t::fresnel_type::create(orientedEta);
)===";
    }
    else
    {
        // brdf
        if (cook_torrance->ndfParams.definitelyIsotropic())
        {
            if (cook_torrance->ndfParams.getDistribution() == CTrueIR::SBasicNDFParams::EDistribution::GGX)
                bxdf_type = "bxdf::reflection::SGGXIsotropic<iso_microfacet_config_t>";
            else
                bxdf_type = "bxdf::reflection::SBeckmannIsotropic<iso_microfacet_config_t>";
        }
        else
        {
            if (cook_torrance->ndfParams.getDistribution() == CTrueIR::SBasicNDFParams::EDistribution::GGX)
                bxdf_type = "bxdf::reflection::SGGXAnisotropic<aniso_microfacet_config_t>";
            else
                bxdf_type = "bxdf::reflection::SBeckmannAnisotropic<aniso_microfacet_config_t>";
        }

        const auto etaNode = ir->getObjectPool().deref(cook_torrance->orientedRealEta);
        const auto eta = etaNode->getParameter(0).scale;
        const auto etak = etaNode->getParameter(1).scale;   // TODO: double check how eta is stored
        fresnel_create = R"===(
    bxdf.fresnel = cook_torrance_t::fresnel_type::create()===" + std::to_string(eta) + ", " + std::to_string(etak) + R"===();
)===";
    }
}

}
