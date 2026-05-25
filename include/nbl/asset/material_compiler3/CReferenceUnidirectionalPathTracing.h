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
        std::string fragmentShaderSource_declarations;
        std::string fragmentShaderSource;
    };

    core::smart_refctd_ptr<CResult> compile(const CTrueIR* ir, const std::span<const CTrueIR::SMaterialHandle> materials);

private:
    std::string getHashAs4UintsString(const CTrueIR::INode* node, const CTrueIR* ir, const std::string& separator = ",") const;

    void getAlbedoHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getNormalHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getAOVThroughputHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getTransparencyHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getGenerateHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getQuotientWeightHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getEvalWeightHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);

    void getEmissionHLSLCode(std::ostringstream& sstr, const CTrueIR::INode* node, const CTrueIR* ir);
};

}

#endif
