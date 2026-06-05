// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_REFERENCE_UNIDIRECTIONAL_PATH_TRACING_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_REFERENCE_UNIDIRECTIONAL_PATH_TRACING_H_INCLUDED_

#include "nbl/asset/material_compiler3/IBackend.h"

namespace nbl::asset::material_compiler3
{

class CReferenceUnidirectionalPathTracing final : public IBackend
{
public:
    class CResult final : public IBackend::IResult
    {
    public:
        std::string fragmentShaderSource_common;
        std::string fragmentShaderSource_raytracingPipeline;    // only maps entry point to templated funcs
    };

    core::smart_refctd_ptr<CResult> compile(const CTrueIR* ir, const std::span<const CTrueIR::SMaterialHandle> materials);

private:
    struct TraversalNodeInfo
    {
        const CTrueIR::INode* node;
        bool isTransmission;
    };

    void traverseIRNode(const CTrueIR::INode* node, const CTrueIR* ir, core::vector<CTrueIR::typed_pointer_type<const CTrueIR::INode>>& nodeStack, core::unordered_map<CTrueIR::typed_pointer_type<const CTrueIR::INode>, TraversalNodeInfo>& nodeInfos);

    std::string getHashAs4UintsString(const CTrueIR::INode* node, const CTrueIR* ir, const std::string& separator = ",") const;

    bool isNodeTypeContributor(CTrueIR::INode::EFinalType type) const;
    void getMaterialDeclarationCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);
    void getCacheDefineCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);

    void getAlbedoHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);
    void getNormalHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);
    void getAOVThroughputHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);
    void getTransparencyHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);
    void getGenerateHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);
    void getQuotientWeightHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);
    void getEvalWeightHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);
    void getEmissionHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);

    void getCanGenerateHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);
    void getChoiceTargetHLSLCode(std::ostringstream& sstr, const TraversalNodeInfo& nodeInfo, const CTrueIR* ir);

    void getCookTorranceBxDFHLSLCode(const CTrueIR::CCookTorrance* cook_torrance, const CTrueIR* ir, std::string& bxdf_type, std::string& fresnel_create);
};

}

#endif
