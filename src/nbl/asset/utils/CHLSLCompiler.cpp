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

core::smart_refctd_ptr<ICPUBuffer> CHLSLCompiler::compileToSPIRV(const char* code, const CHLSLCompiler::SOptions& options) const
{
    core::smart_refctd_ptr<ICPUBuffer> outSpirv = nullptr;
    if (code)
    {
        // TODO: Use DXC
    }
    else
    {
        options.logger.log("code is nullptr", system::ILogger::ELL_ERROR);
    }
    return outSpirv;
}

core::smart_refctd_ptr<ICPUShader> CHLSLCompiler::createSPIRVShader(const char* code, const CHLSLCompiler::SOptions& options) const
{
    auto spirv = compileToSPIRV(code, options);
    if (spirv)
        return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(spirv), options.stage, IShader::E_CONTENT_TYPE::ECT_HLSL, options.sourceIdentifier);
    else
        return nullptr;
}

core::smart_refctd_ptr<ICPUShader> CHLSLCompiler::createSPIRVShader(system::IFile* sourceFile, const CHLSLCompiler::SOptions& options) const
{
    size_t fileSize = sourceFile->getSize();
    std::string code(fileSize, '\0');

    system::IFile::success_t success;
    sourceFile->read(success, code.data(), 0, fileSize);
    if (success)
        return createSPIRVShader(code.c_str(), options);
    else
        return nullptr;
}