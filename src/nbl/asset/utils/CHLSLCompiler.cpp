// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/CHLSLCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"

#include <sstream>
#include <regex>
#include <iterator>


using namespace nbl;
using namespace nbl::asset;


CHLSLCompiler::CHLSLCompiler(core::smart_refctd_ptr<system::ISystem>&& system)
    : IShaderCompiler(std::move(system))
{
}

core::smart_refctd_ptr<ICPUShader> CHLSLCompiler::compileToSPIRV(const char* code, const IShaderCompiler::SCompilerOptions& options) const
{
    auto hlslOptions = option_cast(options);

    if (!code)
    {
        options.preprocessorOptions.logger.log("code is nullptr", system::ILogger::ELL_ERROR);
        return nullptr;
    }

    core::smart_refctd_ptr<ICPUBuffer> spirv = nullptr;
    // TODO: Use DXC
    return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(spirv), options.stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, options.preprocessorOptions.sourceIdentifier.data());
}

void CHLSLCompiler::insertExtraDefines(std::string& code, const core::SRange<const char* const>& defines) const
{
    if (defines.empty())
        return;

    std::ostringstream insertion;
    for (auto def : defines)
    {
        insertion << "#define " << def << "\n";
    }
    code.insert(0u, insertion.str());
}