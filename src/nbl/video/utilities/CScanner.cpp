#include "nbl/video/utilities/CScanner.h"

namespace nbl::video
{
    
asset::ICPUShader* CScanner::getDefaultShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op, const uint32_t scratchSz)
{
    if (!m_shaders[scanType][dataType][op])
        m_shaders[scanType][dataType][op] = createShader(scanType,dataType,op,scratchSz);
    return m_shaders[scanType][dataType][op].get();
}

IGPUShader* CScanner::getDefaultSpecializedShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op, const uint32_t scratchSz)
{
    if (!m_specialized_shaders[scanType][dataType][op])
    {
        auto cpuShader = core::smart_refctd_ptr<asset::ICPUShader>(getDefaultShader(scanType,dataType,op,scratchSz));
        cpuShader->setFilePathHint("nbl/builtin/hlsl/scan/direct.hlsl");
        cpuShader->setShaderStage(asset::IShader::ESS_COMPUTE);

        auto gpushader = m_device->createShader(cpuShader.get());

        m_specialized_shaders[scanType][dataType][op] = gpushader;
    }
    return m_specialized_shaders[scanType][dataType][op].get();
}

IGPUComputePipeline* CScanner::getDefaultPipeline(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op, const uint32_t scratchSz)
{
    // ondemand
    if (!m_pipelines[scanType][dataType][op]) {
        IGPUComputePipeline::SCreationParams params = {};
        params.layout = m_pipeline_layout.get();
        // Theoretically a blob of SPIR-V can contain multiple named entry points and one has to be chosen, in practice most compilers only support outputting one (and glslang used to require it be called "main")
        params.shader.entryPoint = "main";
        params.shader.shader = getDefaultSpecializedShader(scanType, dataType, op, scratchSz);

        m_device->createComputePipelines(
            nullptr, { &params,1 },
            & m_pipelines[scanType][dataType][op]
        );
    }
    return m_pipelines[scanType][dataType][op].get();
}
    
core::smart_refctd_ptr<asset::ICPUShader> CScanner::createShader(const E_SCAN_TYPE scanType, const E_DATA_TYPE dataType, const E_OPERATOR op, const uint32_t scratchSz) const
{
    core::smart_refctd_ptr<asset::ICPUShader> base = createBaseShader("nbl/builtin/hlsl/scan/direct.hlsl", dataType, op, scratchSz);
    return asset::CHLSLCompiler::createOverridenCopy(base.get(), "#define IS_EXCLUSIVE %s\n#define IS_SCAN %s\n", scanType == EST_EXCLUSIVE ? "true" : "false", "true");
}

}