// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/utilities/CReduce.h"

namespace nbl::video
{

asset::ICPUShader* CReduce::getDefaultShader(const CArithmeticOps::E_DATA_TYPE dataType, const CArithmeticOps::E_OPERATOR op, const uint32_t scratchElCount)
{
    if (!m_shaders[dataType][op])
        m_shaders[dataType][op] = createShader(dataType,op,scratchElCount);
    return m_shaders[dataType][op].get();
}

IGPUShader* CReduce::getDefaultSpecializedShader(const CArithmeticOps::E_DATA_TYPE dataType, const CArithmeticOps::E_OPERATOR op, const uint32_t scratchElCount)
{
    if (!m_specialized_shaders[dataType][op])
    {
        auto cpuShader = core::smart_refctd_ptr<asset::ICPUShader>(getDefaultShader(dataType,op,scratchElCount));
        cpuShader->setFilePathHint("nbl/builtin/hlsl/scan/direct.hlsl");
        cpuShader->setShaderStage(asset::IShader::ESS_COMPUTE);

        auto gpushader = m_device->createShader(cpuShader.get());

        m_specialized_shaders[dataType][op] = gpushader;
    }
    return m_specialized_shaders[dataType][op].get();
}

IGPUComputePipeline* CReduce::getDefaultPipeline(const CArithmeticOps::E_DATA_TYPE dataType, const CArithmeticOps::E_OPERATOR op, const uint32_t scratchElCount)
{
    // ondemand
    if (!m_pipelines[dataType][op]) {
        IGPUComputePipeline::SCreationParams params = {};
        params.layout = m_pipeline_layout.get();
        // Theoretically a blob of SPIR-V can contain multiple named entry points and one has to be chosen, in practice most compilers only support outputting one (and glslang used to require it be called "main")
        params.shader.entryPoint = "main";
        params.shader.shader = getDefaultSpecializedShader(dataType,op,scratchElCount);

        m_device->createComputePipelines(
            nullptr, { &params,1 },
            & m_pipelines[dataType][op]
        );
    }
    return m_pipelines[dataType][op].get();
}

core::smart_refctd_ptr<asset::ICPUShader> CReduce::createShader(const CArithmeticOps::E_DATA_TYPE dataType, const CArithmeticOps::E_OPERATOR op, const uint32_t scratchElCount) const
{
    core::smart_refctd_ptr<asset::ICPUShader> base = createBaseShader("nbl/builtin/hlsl/scan/direct.hlsl", dataType, op, scratchElCount);
    return asset::CHLSLCompiler::createOverridenCopy(base.get(), "#define IS_EXCLUSIVE %s\n#define IS_SCAN %s\n", "false", "false");
}

}