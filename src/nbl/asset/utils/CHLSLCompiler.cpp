// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/CHLSLCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"

#include <sstream>
#include <regex>
#include <iterator>

#include <dxc/dxcapi.h>
#include <combaseapi.h>

using namespace nbl;
using namespace nbl::asset;


CHLSLCompiler::CHLSLCompiler(core::smart_refctd_ptr<system::ISystem>&& system)
    : IShaderCompiler(std::move(system))
{
    IDxcUtils* utils;
    auto res = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
    assert(SUCCEEDED(res));

    IDxcCompiler3* compiler;
    res = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));
    assert(SUCCEEDED(res));

    m_dxcUtils = std::unique_ptr<IDxcUtils>(utils);
    m_dxcCompiler = std::unique_ptr<IDxcCompiler3>(compiler);
}

CHLSLCompiler::~CHLSLCompiler()
{
    m_dxcUtils->Release();
    m_dxcCompiler->Release();
}

CHLSLCompiler::DxcCompilationResult CHLSLCompiler::dxcCompile(std::string& source, LPCWSTR* args, uint32_t argCount, const SOptions& options) const
{
    if (options.genDebugInfo)
    {
        std::ostringstream insertion;
        insertion << "// commandline compiler options : ";

        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> conv;
        for (uint32_t arg = 0; arg < argCount; arg ++)
        {
            auto str = conv.to_bytes(args[arg]);
            insertion << str.c_str() << " ";
        }

        insertion << "\n";
        insertIntoStart(source, std::move(insertion));
    }

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = source.data();
    sourceBuffer.Size = source.size();
    sourceBuffer.Encoding = 0;

    IDxcResult* compileResult;
    auto res = m_dxcCompiler->Compile(&sourceBuffer, args, argCount, nullptr, IID_PPV_ARGS(&compileResult));
    // If the compilation failed, this should still be a successful result
    assert(SUCCEEDED(res));

    HRESULT compilationStatus = 0;
    res = compileResult->GetStatus(&compilationStatus);
    assert(SUCCEEDED(res));

    IDxcBlobEncoding* errorBuffer;
    res = compileResult->GetErrorBuffer(&errorBuffer);
    assert(SUCCEEDED(res));

    DxcCompilationResult result;
    result.errorMessages = std::unique_ptr<IDxcBlobEncoding>(errorBuffer);
    result.compileResult = std::unique_ptr<IDxcResult>(compileResult);
    result.objectBlob = nullptr;

    if (!SUCCEEDED(compilationStatus))
    {
        options.preprocessorOptions.logger.log(result.GetErrorMessagesString(), system::ILogger::ELL_ERROR);
        return result;
    }

    IDxcBlob* resultingBlob;
    res = compileResult->GetResult(&resultingBlob);
    assert(SUCCEEDED(res));

    result.objectBlob = std::unique_ptr<IDxcBlob>(resultingBlob);

    return result;
}

core::smart_refctd_ptr<ICPUShader> CHLSLCompiler::compileToSPIRV(const char* code, const IShaderCompiler::SCompilerOptions& options) const
{
    auto hlslOptions = option_cast(options);

    if (!code)
    {
        hlslOptions.preprocessorOptions.logger.log("code is nullptr", system::ILogger::ELL_ERROR);
        return nullptr;
    }

    auto newCode = preprocessShader(code, hlslOptions.stage, hlslOptions.preprocessorOptions);

    LPCWSTR arguments[] = {
        // These will always be present
        L"-spirv",
        L"-HLSL2021",

        // These are debug only
        L"-Qembed_debug"
    };

    const uint32_t nonDebugArgs = 2;
    const uint32_t allArgs = nonDebugArgs + 0;

    DxcCompilationResult compileResult = dxcCompile(newCode, &arguments[0], hlslOptions.genDebugInfo ? allArgs : nonDebugArgs, hlslOptions);

    if (!compileResult.objectBlob)
    {
        return nullptr;
    }

    auto outSpirv = core::make_smart_refctd_ptr<ICPUBuffer>(compileResult.objectBlob->GetBufferSize());
    memcpy(outSpirv->getPointer(), compileResult.objectBlob->GetBufferPointer(), compileResult.objectBlob->GetBufferSize());

    return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(outSpirv), hlslOptions.stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, hlslOptions.preprocessorOptions.sourceIdentifier.data());
}

void CHLSLCompiler::insertIntoStart(std::string& code, std::ostringstream&& ins) const
{
    code.insert(0u, ins.str());
}
