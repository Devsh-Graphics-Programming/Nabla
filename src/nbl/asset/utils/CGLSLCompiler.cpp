// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/CGLSLCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"

#include <sstream>
#include <regex>
#include <iterator>


using namespace nbl;
using namespace nbl::asset;


CGLSLCompiler::CGLSLCompiler(core::smart_refctd_ptr<system::ISystem>&& system)
    : IShaderCompiler(std::move(system))
{
}

core::smart_refctd_ptr<ICPUShader> CGLSLCompiler::compileToSPIRV(const char* code, const IShaderCompiler::SOptions& options) const
{
    auto glslOptions = option_cast(options);

    if (!code)
    {
        glslOptions.logger.log("code is nullptr", system::ILogger::ELL_ERROR);
        return nullptr;
    }

    if (glslOptions.entryPoint.compare("main") != 0)
    {
        glslOptions.logger.log("shaderc requires entry point to be \"main\" in GLSL", system::ILogger::ELL_ERROR);
        return nullptr;
    }

    core::smart_refctd_ptr<asset::ICPUShader> cpuShader;
    if (glslOptions.includeFinder != nullptr)
    {
        cpuShader = resolveIncludeDirectives(code, glslOptions.stage, glslOptions.sourceIdentifier.data(), glslOptions.maxSelfInclusionCount, glslOptions.logger);
        if (cpuShader)
        {
            code = reinterpret_cast<const char*>(cpuShader->getContent()->getPointer());
        }
    }

    shaderc::Compiler comp;
    shaderc::CompileOptions shadercOptions; //default options
    shadercOptions.SetTargetSpirv(static_cast<shaderc_spirv_version>(glslOptions.targetSpirvVersion));
    const shaderc_shader_kind stage = glslOptions.stage == IShader::ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(glslOptions.stage);
    const size_t glsl_len = strlen(code);
    if (glslOptions.genDebugInfo)
        shadercOptions.SetGenerateDebugInfo();

    shaderc::SpvCompilationResult bin_res = comp.CompileGlslToSpv(code, glsl_len, stage, glslOptions.sourceIdentifier.data() ? glslOptions.sourceIdentifier.data() : "", glslOptions.entryPoint.data(), shadercOptions);

    if (bin_res.GetCompilationStatus() == shaderc_compilation_status_success)
    {
        auto outSpirv = core::make_smart_refctd_ptr<ICPUBuffer>(std::distance(bin_res.cbegin(), bin_res.cend()) * sizeof(uint32_t));
        memcpy(outSpirv->getPointer(), bin_res.cbegin(), outSpirv->getSize());

        if (glslOptions.spirvOptimizer)
            outSpirv = glslOptions.spirvOptimizer->optimize(outSpirv.get(), glslOptions.logger);
        return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(outSpirv), glslOptions.stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, glslOptions.sourceIdentifier.data());
    }
    else
    {
        glslOptions.logger.log(bin_res.GetErrorMessage(), system::ILogger::ELL_ERROR);
        return nullptr;
    }
}