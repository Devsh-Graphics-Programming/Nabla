// Copyright (C) 2022-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#define _NBL_ASSET_MATERIAL_COMPILER3_C_REFERENCE_UNIDIRECTIONAL_PATH_TRACING_CPP_
#include "nbl/asset/material_compiler3/CReferenceUnidirectionalPathTracing.h"

namespace nbl::asset::material_compiler3
{

CReferenceUnidirectionalPathTracing::CResult CReferenceUnidirectionalPathTracing::compile(const CTrueIR* ir, const std::span<const CTrueIR::SMaterialHandle> materialHandles)
{
    CResult res;

    // TODO: handle textures somehow
    // TODO: templated structs for types in certain functions, e.g. cache (generate + quotient)
    // TODO: where do all the type aliases come from? e.g. sample_t, vector3_t, etc.
    std::ostringstream code;

    // define templates of node functions + util structs
    code << "template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>\nstruct gen_cache;\n"; // cache struct

    code << "template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>\nspectral_t albedo();\n"; // TODO: what args needed? uv?
    code << "template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>\nvector3_t normal();\n";  // TODO: what args needed? uv?
    code << "template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>\nvector3_t aov_throughput();\n";  // TODO: confirm return value is vec3, what args needed? uv?
    code << "template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>\nvector3_t transparency();\n";  // TODO: return value is vec3? what args needed?
    
    // TODO: check how many rand numbers, might need to declare template functions by node type as well (because rand numbers vary by generate, etc.)
    // if the above is the case, will have to check whether the node-specific template function declaration is included yet, then add before function definition
    code << "template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>\n"
        << "sample_t generate(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, "
        << "NBL_REF_ARG(gen_cache<hash0, hash1, hash2, hash3, hash4, hash5, hash6, hash7>) cache);\n";

    code << "template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>\n"
        << "quotient_weight_t quotientAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter, "
        << "NBL_REF_ARG(gen_cache<hash0, hash1, hash2, hash3, hash4, hash5, hash6, hash7>) cache);\n";

    code << "template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>\n"
        << "eval_weight_t evalAndWeight(NBL_CONST_REF_ARG(sample_t) _sample, NBL_CONST_REF_ARG(aniso_interaction_t) inter);\n";

    code << "template<uint32_t hash0, uint32_t hash1, uint32_t hash2, uint32_t hash3, uint32_t hash4, uint32_t hash5, uint32_t hash6, uint32_t hash7>\nspectral_t emission();\n";  // TODO: what args needed?

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

            getAlbedoHLSLCode(code, node, ir);
            getNormalHLSLCode(code, node, ir);
            getAOVThroughputHLSLCode(code, node, ir);
            getTransparencyHLSLCode(code, node, ir);
            getGenerateHLSLCode(code, node, ir);
            getEvalWeightHLSLCode(code, node, ir);
            getQuotientWeightHLSLCode(code, node, ir);
            getEmissionHLSLCode(code, node, ir);

            // TODO: might need to do children first or forward declare function signatures
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

    res.fragmentShaderSource = code.str();

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
        sstr << "template<>\nspectral_t albedo<" << hashString << ">()\n{\n";   // TODO: what args needed? uv?

        if (auto childBrdf = ir->getObjectPool().deref(layer->brdfTop); childBrdf)
        {
            const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
            sstr << "spectral_t brdf = albedo<" << childBrdfHash << ">();\n";
        }
        if (auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission); childBtdf)
        {
            const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
            sstr << "spectral_t btdf = albedo<" << childBtdfHash << ">();\n";
        }

        // TODO: mix result
        sstr << "spectral_t retval = (brdf + btdf) * 0.5;\n";
        sstr << "return retval;\n}\n";
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << "template<>\nspectral_t albedo<" << hashString << ">()\n{\n";   // TODO: what args needed? uv?

        if (auto childProduct = ir->getObjectPool().deref(sum->product); childProduct)
        {
            const auto childProductHash = getHashAs4UintsString(childProduct, ir);
            sstr << "spectral_t product = albedo<" << childProductHash << ">();\n";
            // TODO: what to do with child caches?
        }
        if (auto childRest = ir->getObjectPool().deref(sum->rest); childRest)
        {
            const auto childRestHash = getHashAs4UintsString(childRest, ir);
            sstr << "spectral_t rest = albedo<" << childRestHash << ">();\n";
            // TODO: what to do with child caches?
        }

        // TODO: weight sample
        sstr << "spectral_t retval = (product + rest) * 0.5;\n";
        sstr << "return retval;\n}\n";
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
        sstr << "template<>\nspectral_t albedo<" << hashString << ">()\n{\n";   // TODO: what args needed? uv?

        auto roughness = oren_nayar->ndfParams.getRougness();
        sstr << "using oren_nayar_t = bxdf::reflection::SOrenNayar<iso_config_t>;\n";
        sstr << "using creation_t = typename oren_nayar_t::creation_type;\n";
        sstr << "creation_t params;\nparams.A = " << roughness.data()[0].scale << ";\n";
        sstr << "oren_nayar_t bxdf = diffuse_op_type::create(params);\n";

        sstr << "typename oren_nayar_t::anisocache_type bxdf_cache;\n";
        sstr << "sample_t _sample = bxdf.generate(inter, xi, bxdf_cache);\n";
        // TODO: what to do with child caches?

        sstr << "return hlsl::promote<spectral_t>();\n}\n";
        break;
        break;
    }
    case CTrueIR::INode::EFinalType::CCookTorrance:
    {
        // TODO
        break;
    }
    default:
    {
        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << "template<>\nspectral_t albedo<" << hashString << ">()\n{\nreturn hlsl::promote<spectral_t>(0);\n}\n";   // TODO: what args needed? uv?
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
        sstr << "template<>\nsample_t generate<" << hashString;
        sstr << ">(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(rand_t) xi_extra, NBL_REF_ARG(gen_cache<"	// TODO: class type specific cache
            << hashString << ">) cache)\n{\n";

        if (auto childBrdf = ir->getObjectPool().deref(layer->brdfTop); childBrdf)
        {
            const auto childBrdfHash = getHashAs4UintsString(childBrdf, ir);
            sstr << "gen_cache<" << childBrdfHash << "> brdf_cache;\n";
            sstr << "sample_t brdf = generate<" << childBrdfHash << ">(inter, xi, brdf_cache);\n";
            // TODO: what to do with child caches?
        }
        if (auto childBtdf = ir->getObjectPool().deref(layer->firstTransmission); childBtdf)
        {
            const auto childBtdfHash = getHashAs4UintsString(childBtdf, ir);
            sstr << "gen_cache<" << childBtdfHash << "> btdf_cache;\n";
            sstr << "sample_t btdf = generate<" << childBtdfHash << ">(inter, xi_extra, btdf_cache);\n";
            // TODO: what to do with child caches?
        }

        // TODO: do resampled importance sampling HLSL code

        sstr << "return retval;\n}\n";
        break;
    }
    case CTrueIR::INode::EFinalType::CContributorSum:
    {
        const auto* sum = dynamic_cast<const CTrueIR::CContributorSum*>(node);
        if (!sum)
            break;

        const auto hashString = getHashAs4UintsString(node, ir);
        sstr << "template<>\nsample_t generate<" << hashString;
        sstr << ">(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(gen_cache<"
            << hashString << ">) cache)\n{\n";

        sstr << "uint16_t chosenLobe = 0;\npdf_t choiceRcpPdf = 1.f;\n";

        if (auto childProduct = ir->getObjectPool().deref(sum->product); childProduct)
        {
            const auto childProductHash = getHashAs4UintsString(childProduct, ir);
            sstr << "gen_cache<" << childProductHash << "> product_cache;\n";
            sstr << "sample_t product = generate<" << childProductHash << ">(inter, xi, product_cache);\n";
            // TODO: what to do with child caches?
        }
        if (auto childRest = ir->getObjectPool().deref(sum->rest); childRest)
        {
            const auto childRestHash = getHashAs4UintsString(childRest, ir);
            sstr << "gen_cache<" << childRestHash << "> rest_cache;\n";
            sstr << "sample_t rest = generate<" << childRestHash << ">(inter, xi, rest_cache);\n";
            // TODO: what to do with child caches?
        }

        // TODO: weight sample

        sstr << "return retval;\n}\n";
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
        sstr << "template<>\nsample_t generate<" << hashString;
        sstr << ">(NBL_CONST_REF_ARG(aniso_interaction_t) inter, NBL_REF_ARG(rand_t) xi, NBL_REF_ARG(gen_cache<"
            << hashString << ">) cache)\n{\n";

        auto roughness = oren_nayar->ndfParams.getRougness();
        sstr << "using oren_nayar_t = bxdf::reflection::SOrenNayar<iso_config_t>;\n";
        sstr << "using creation_t = typename oren_nayar_t::creation_type;\n";
        sstr << "creation_t params;\nparams.A = " << roughness.data()[0].scale << ";\n";
        sstr << "oren_nayar_t bxdf = diffuse_op_type::create(params);\n";

        sstr << "typename oren_nayar_t::anisocache_type bxdf_cache;\n";
        sstr << "sample_t _sample = bxdf.generate(inter, xi, bxdf_cache);\n";
        // TODO: what to do with child caches?

        sstr << "return _sample;\n}\n";
        break;
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

}
