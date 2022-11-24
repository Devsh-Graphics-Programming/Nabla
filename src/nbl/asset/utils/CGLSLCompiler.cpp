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

core::smart_refctd_ptr<ICPUShader> CGLSLCompiler::compileToSPIRV(const char* code, const CGLSLCompiler::SOptions& options) const
{
    if (!code)
    {
        options.logger.log("code is nullptr", system::ILogger::ELL_ERROR);
        return nullptr;
    }

    if (options.entryPoint.compare("main") != 0)
    {
        options.logger.log("shaderc requires entry point to be \"main\" in GLSL", system::ILogger::ELL_ERROR);
        return nullptr;
    }

    core::smart_refctd_ptr<asset::ICPUShader> cpuShader;
    if (options.includeFinder != nullptr)
    {
        cpuShader = resolveIncludeDirectives(code, options.stage, options.sourceIdentifier.data(), options.maxSelfInclusionCount, options.logger);
        if (cpuShader)
        {
            code = reinterpret_cast<const char*>(cpuShader->getContent()->getPointer());
        }
    }

    shaderc::Compiler comp;
    shaderc::CompileOptions shadercOptions; //default options
    shadercOptions.SetTargetSpirv(static_cast<shaderc_spirv_version>(options.targetSpirvVersion));
    const shaderc_shader_kind stage = options.stage == IShader::ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(options.stage);
    const size_t glsl_len = strlen(code);
    if (options.genDebugInfo)
        shadercOptions.SetGenerateDebugInfo();

    shaderc::SpvCompilationResult bin_res = comp.CompileGlslToSpv(code, glsl_len, stage, options.sourceIdentifier.data() ? options.sourceIdentifier.data() : "", options.entryPoint.data(), shadercOptions);

    if (bin_res.GetCompilationStatus() == shaderc_compilation_status_success)
    {
        auto outSpirv = core::make_smart_refctd_ptr<ICPUBuffer>(std::distance(bin_res.cbegin(), bin_res.cend()) * sizeof(uint32_t));
        memcpy(outSpirv->getPointer(), bin_res.cbegin(), outSpirv->getSize());

        if (options.spirvOptimizer)
            outSpirv = options.spirvOptimizer->optimize(outSpirv.get(), options.logger);
        return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(outSpirv), options.stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, options.sourceIdentifier.data());
    }
    else
    {
        options.logger.log(bin_res.GetErrorMessage(), system::ILogger::ELL_ERROR);
        return nullptr;
    }
}

core::smart_refctd_ptr<ICPUShader> CGLSLCompiler::compileToSPIRV(system::IFile* sourceFile, const CGLSLCompiler::SOptions& options) const
{
    size_t fileSize = sourceFile->getSize();
    std::string code(fileSize, '\0');

    system::IFile::success_t success;
    sourceFile->read(success, code.data(), 0, fileSize);
    if (success)
        return compileToSPIRV(code.c_str(), options);
    else
        return nullptr;
}