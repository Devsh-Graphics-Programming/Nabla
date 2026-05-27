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
    // TODO: templated structs for types in certain functions, e.g. cache (generate + quotient)
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
    core::unordered_set<CTrueIR::typed_pointer_type<const CTrueIR::INode>> visitedNodes;
    auto compileBSDFRootNode = [&](CTrueIR::typed_pointer_type<const CTrueIR::INode> rootHandle) -> void {
        nodeStack.clear();
        visitedNodes.clear();

        nodeStack.push_back(rootHandle);
        const auto& pool = ir->getObjectPool();

        while (!nodeStack.empty())
        {
            const auto handle = nodeStack.back();
            nodeStack.pop_back();
            const auto* node = pool.deref(handle);
            if (!node)
                continue;

            getMaterialDeclarationCode(code, node, ir);

            const auto childCount = node->getChildCount();
            if (childCount)
            {
                for (auto childIx = 0; childIx < childCount; childIx++)
                {
                    const auto childHandle = node->getChildHandle(childIx);
                    if (const auto child = pool.deref(childHandle); child)
                    {
                        const auto [unused, inserted] = visitedNodes.insert(childHandle);
                        if (inserted)
                            nodeStack.push_back(childHandle);
                    }
                }
            }
        }

        for (const auto& nodeHandle : visitedNodes)
        {
            const auto* node = pool.deref(nodeHandle);
            getAlbedoHLSLCode(code, node, ir);
            getNormalHLSLCode(code, node, ir);
            getAOVThroughputHLSLCode(code, node, ir);
            getTransparencyHLSLCode(code, node, ir);
            getGenerateHLSLCode(code, node, ir);
            getEvalWeightHLSLCode(code, node, ir);
            getQuotientWeightHLSLCode(code, node, ir);
            getEmissionHLSLCode(code, node, ir);
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

void CReferenceUnidirectionalPathTracing::getMaterialDeclarationCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir)
{
    // define templates of node material structs
    const auto hashString = getHashAs4UintsString(node, ir);
    
    // TODO: specialize gen_cache struct

    sstr << R"===(
template<>
struct OrientedMaterial<)===" << hashString << R"===(>
{
    static spectral_t albedo();
    static vector3_t normal(NBL_CONST_REF_ARG(aniso_interaction_t) inter);
    static spectral_t aovThroughput();
    static scalar_t transparency();
    static sample_t generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache);
    static quotient_weight_t quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache);
    static eval_weight_t evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache);
    static spectral_t emission();
}
)===";

}

void CReferenceUnidirectionalPathTracing::getAlbedoHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir)
{
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
        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
            sstr << "child" << static_cast<uint32_t>(i) << (i < childCount - 1 ? " + " : "");
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

void CReferenceUnidirectionalPathTracing::getNormalHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir)
{
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

void CReferenceUnidirectionalPathTracing::getAOVThroughputHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir)
{
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
        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
            sstr << "child" << static_cast<uint32_t>(i) << (i < childCount - 1 ? " + " : "");
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

void CReferenceUnidirectionalPathTracing::getTransparencyHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir)
{
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
        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
            sstr << "child" << static_cast<uint32_t>(i) << (i < childCount - 1 ? " + " : "");
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

void CReferenceUnidirectionalPathTracing::getGenerateHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir)
{
    switch (node->getFinalType())
    {
    case CTrueIR::INode::EFinalType::COrientedLayer:
    {
        const auto* layer = dynamic_cast<const CTrueIR::COrientedLayer*>(node);
        if (!layer)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << R"===(
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache"
{
)===";

        if (auto childBrdf = ir->getObjectPool().deref(layer->brdfTop); childBrdf)
        {
            const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
            sstr << R"===(
    gen_cache<)===" << childBrdfHash << R"===(> brdf_cache;
    sample_t brdf = OrientedMaterial<)===" << childBrdfHash << R"===(>::generate(inter, xi, xi_extra, brdf_cache);
)===";
            // TODO: what to do with child caches?
        }
        if (auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission); childBtdf)
        {
            const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
            sstr << R"===(
    gen_cache<)===" << childBtdfHash << R"===(> btdf_cache;
    sample_t btdf = OrientedMaterial<)===" << childBtdfHash << R"===(>::generate(inter, xi, xi_extra, btdf_cache);
)===";
            // TODO: what to do with child caches?
        }

        // TODO: do resampled importance sampling HLSL code

        sstr << R"===(
    return retval;
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
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache"
{
    uint16_t chosenLobe = 0;
    pdf_t choiceRcpPdf = 1.f;
)===";

        if (auto childProduct = ir->getObjectPool().deref(sum->product); childProduct)
        {
            const auto childProductHash = getHashAs4UintsString(childProduct, ir);
            sstr << R"===(
    gen_cache<)===" << childProductHash << R"===(> product_cache;
    sample_t product = OrientedMaterial<)===" << childProductHash << R"===(>::generate(inter, xi, xi_extra, product_cache);
)===";
            // TODO: what to do with child caches?
        }
        if (auto childRest = ir->getObjectPool().deref(sum->rest); childRest)
        {
            const auto childRestHash = getHashAs4UintsString(childRest, ir);
            sstr << R"===(
    gen_cache<)===" << childRestHash << R"===(> rest_cache;
    sample_t rest = OrientedMaterial<)===" << childRestHash << R"===(>::generate(inter, xi_extra, xi, rest_cache);
)===";
            // TODO: what to do with child caches?
        }

        // TODO: weight sample

        sstr << R"===(
    return retval;
}
)===";
        break;
    }
    case CTrueIR::INode::EFinalType::CWeightedContributor:
    {
        // TODO
        break;
    }
    case CTrueIR::INode::EFinalType::CCorellatedTransmission:
    {
        // TODO
        break;
    }
    case CTrueIR::INode::EFinalType::COrenNayar:
    {
        const auto* oren_nayar = dynamic_cast<const CTrueIR::COrenNayar*>(node);
        if (!oren_nayar)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        auto roughness = oren_nayar->ndfParams.getRougness();
        sstr << R"===(
static sample_t OrientedMaterial<)===" << hashString << R"===(>::generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<)===" << hashString << R"===(>) cache"
{
    using oren_nayar_t = bxdf::reflection::SOrenNayar<iso_config_t>;
    using creation_t = typename oren_nayar_t::creation_type;
    creation_t params;
    params.A = )===" << roughness.data()[0].scale << R"===(;
    oren_nayar_t bxdf = diffuse_op_type::create(params);
    typename oren_nayar_t::anisocache_type bxdf_cache;
    sample_t _sample = bxdf.generate(inter, xi, bxdf_cache);
    return _sample;
}
)===";
        // TODO: what to do with child caches?
        break;
    }
    case CTrueIR::INode::EFinalType::CCookTorrance:
    {
        // TODO
        break;
    }
    default:
        break;
    }
}

void CReferenceUnidirectionalPathTracing::getEmissionHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir)
{
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
        for (uint8_t i = 0; i < childCount; i++)    // TODO: check for invalid children?
            sstr << "child" << static_cast<uint32_t>(i) << (i < childCount - 1 ? " + " : "");
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

}
