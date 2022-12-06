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

core::smart_refctd_ptr<ICPUShader> CGLSLCompiler::compileToSPIRV(const char* code, const IShaderCompiler::SCompilerOptions& options) const
{
    auto glslOptions = option_cast(options);

    if (!code)
    {
        glslOptions.preprocessorOptions.logger.log("code is nullptr", system::ILogger::ELL_ERROR);
        return nullptr;
    }

    if (glslOptions.entryPoint.compare("main") != 0)
    {
        glslOptions.preprocessorOptions.logger.log("shaderc requires entry point to be \"main\" in GLSL", system::ILogger::ELL_ERROR);
        return nullptr;
    }

    auto newCode = preprocessShader(code, glslOptions.stage, glslOptions.preprocessorOptions);

    shaderc::Compiler comp;
    shaderc::CompileOptions shadercOptions; //default options
    shadercOptions.SetTargetSpirv(static_cast<shaderc_spirv_version>(glslOptions.targetSpirvVersion));
    const shaderc_shader_kind stage = glslOptions.stage == IShader::ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(glslOptions.stage);
    if (glslOptions.genDebugInfo)
        shadercOptions.SetGenerateDebugInfo();

    shaderc::SpvCompilationResult bin_res = comp.CompileGlslToSpv(newCode.c_str(), newCode.size(), stage, glslOptions.preprocessorOptions.sourceIdentifier.data() ? glslOptions.preprocessorOptions.sourceIdentifier.data() : "", glslOptions.entryPoint.data(), shadercOptions);

    if (bin_res.GetCompilationStatus() == shaderc_compilation_status_success)
    {
        auto outSpirv = core::make_smart_refctd_ptr<ICPUBuffer>(std::distance(bin_res.cbegin(), bin_res.cend()) * sizeof(uint32_t));
        memcpy(outSpirv->getPointer(), bin_res.cbegin(), outSpirv->getSize());

        if (glslOptions.spirvOptimizer)
            outSpirv = glslOptions.spirvOptimizer->optimize(outSpirv.get(), glslOptions.preprocessorOptions.logger);
        return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(outSpirv), glslOptions.stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, glslOptions.preprocessorOptions.sourceIdentifier.data());
    }
    else
    {
        glslOptions.preprocessorOptions.logger.log(bin_res.GetErrorMessage(), system::ILogger::ELL_ERROR);
        return nullptr;
    }
}

void CGLSLCompiler::insertIntoStart(std::string& code, std::ostringstream&& ins) const
{
    auto findLineJustAfterVersionOrPragmaShaderStageDirective = [&code]() -> size_t
    {
        size_t hashPos = code.find_first_of('#');
        if (hashPos >= code.length())
            return code.npos;
        if (code.compare(hashPos, 8, "#version"))
            return code.npos;

        size_t searchPos = hashPos + 8ull;

        size_t hashPos2 = code.find_first_of('#', hashPos + 8ull);
        if (hashPos2 < code.length())
        {
            char pragma_stage_str[] = "#pragma shader_stage";
            if (code.compare(hashPos2, sizeof(pragma_stage_str) - 1ull, pragma_stage_str) == 0)
                searchPos = hashPos2 + sizeof(pragma_stage_str) - 1ull;
        }
        size_t nlPos = code.find_first_of('\n', searchPos);

        return (nlPos >= code.length()) ? code.npos : nlPos + 1ull;
    };

    const size_t pos = findLineJustAfterVersionOrPragmaShaderStageDirective();
    if (pos == code.npos)
        return;

    const size_t ln = std::count(code.begin(), code.begin() + pos, '\n') + 1;//+1 to count from 1

    ins << "#line " << std::to_string(ln) << "\n";
    code.insert(pos, ins.str());
}
