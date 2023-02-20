// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/CHLSLCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"

#include <wrl.h>
#include <combaseapi.h>

#include <dxc/dxcapi.h>

#include <sstream>
#include <regex>
#include <iterator>

#define TCPP_IMPLEMENTATION
#include <tcpp/source/tcppLibrary.hpp>
#undef TCPP_IMPLEMENTATION

using namespace nbl;
using namespace nbl::asset;
using Microsoft::WRL::ComPtr;

namespace nbl::asset::hlsl::impl
{
    struct DXC {
        Microsoft::WRL::ComPtr<IDxcUtils> m_dxcUtils;
        Microsoft::WRL::ComPtr<IDxcCompiler3> m_dxcCompiler;
    };
}

CHLSLCompiler::CHLSLCompiler(core::smart_refctd_ptr<system::ISystem>&& system)
    : IShaderCompiler(std::move(system))
{
    ComPtr<IDxcUtils> utils;
    auto res = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(utils.GetAddressOf()));
    assert(SUCCEEDED(res));

    ComPtr<IDxcCompiler3> compiler;
    res = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(compiler.GetAddressOf()));
    assert(SUCCEEDED(res));

    m_dxcCompilerTypes = std::unique_ptr<nbl::asset::hlsl::impl::DXC>(new nbl::asset::hlsl::impl::DXC{
        utils,
        compiler
    });
}

static tcpp::IInputStream* getInputStreamInclude(
    const IShaderCompiler::CIncludeFinder* _inclFinder,
    const system::ISystem* _fs,
    uint32_t _maxInclCnt,
    const char* _requesting_source,
    const char* _requested_source,
    bool _type // true for #include "string"; false for #include <string>
)
{
    std::string res_str;

    std::filesystem::path relDir;
    const bool reqFromBuiltin = builtin::hasPathPrefix(_requesting_source);
    const bool reqBuiltin = builtin::hasPathPrefix(_requested_source);
    if (!reqFromBuiltin && !reqBuiltin)
    {
        //While #includ'ing a builtin, one must specify its full path (starting with "nbl/builtin" or "/nbl/builtin").
        //  This rule applies also while a builtin is #includ`ing another builtin.
        //While including a filesystem file it must be either absolute path (or relative to any search dir added to asset::iIncludeHandler; <>-type),
        //  or path relative to executable's working directory (""-type).
        relDir = std::filesystem::path(_requesting_source).parent_path();
    }
    std::filesystem::path name = _type ? (relDir / _requested_source) : (_requested_source);

    if (std::filesystem::exists(name) && !reqBuiltin)
        name = std::filesystem::absolute(name);

    if (_type)
        res_str = _inclFinder->getIncludeRelative(relDir, _requested_source);
    else //shaderc_include_type_standard
        res_str = _inclFinder->getIncludeStandard(relDir, _requested_source);

    if (!res_str.size()) {
        return new tcpp::StringInputStream("#error File not found");
    }

    IShaderCompiler::disableAllDirectivesExceptIncludes(res_str);
    res_str = IShaderCompiler::encloseWithinExtraInclGuards(std::move(res_str), _maxInclCnt, name.string().c_str());

    return new tcpp::StringInputStream(std::move(res_str));
}

class DxcCompilationResult
{
public:
    ComPtr<IDxcBlobEncoding> errorMessages;
    ComPtr<IDxcBlob> objectBlob;
    ComPtr<IDxcResult> compileResult;

    char* GetErrorMessagesString()
    {
        return reinterpret_cast<char*>(errorMessages->GetBufferPointer());
    }
};

DxcCompilationResult dxcCompile(const CHLSLCompiler* compiler, nbl::asset::hlsl::impl::DXC* dxc, std::string& source, LPCWSTR* args, uint32_t argCount, const CHLSLCompiler::SOptions& options)
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
        compiler->insertIntoStart(source, std::move(insertion));
    }
    
    ComPtr<IDxcBlobEncoding> src;
    auto res = dxc->m_dxcUtils->CreateBlob(reinterpret_cast<const void*>(source.data()), source.size(), CP_UTF8, &src);
    assert(SUCCEEDED(res));

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = src->GetBufferPointer();
    sourceBuffer.Size = src->GetBufferSize();
    sourceBuffer.Encoding = 0;

    ComPtr<IDxcResult> compileResult;
    res = dxc->m_dxcCompiler->Compile(&sourceBuffer, args, argCount, nullptr, IID_PPV_ARGS(compileResult.GetAddressOf()));
    // If the compilation failed, this should still be a successful result
    assert(SUCCEEDED(res));

    HRESULT compilationStatus = 0;
    res = compileResult->GetStatus(&compilationStatus);
    assert(SUCCEEDED(res));

    ComPtr<IDxcBlobEncoding> errorBuffer;
    res = compileResult->GetErrorBuffer(errorBuffer.GetAddressOf());
    assert(SUCCEEDED(res));

    DxcCompilationResult result;
    result.errorMessages = errorBuffer;
    result.compileResult = compileResult;
    result.objectBlob = nullptr;

    if (!SUCCEEDED(compilationStatus))
    {
        options.preprocessorOptions.logger.log(result.GetErrorMessagesString(), system::ILogger::ELL_ERROR);
        return result;
    }

    ComPtr<IDxcBlob> resultingBlob;
    res = compileResult->GetResult(resultingBlob.GetAddressOf());
    assert(SUCCEEDED(res));

    result.objectBlob = resultingBlob;

    return result;
}


std::string CHLSLCompiler::preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions) const
{
    if (preprocessOptions.extraDefines.size())
    {
        insertExtraDefines(code, preprocessOptions.extraDefines);
    }

    IShaderCompiler::disableAllDirectivesExceptIncludes(code);

    tcpp::StringInputStream codeIs = tcpp::StringInputStream(code);
    tcpp::Lexer lexer(codeIs);
    tcpp::Preprocessor proc(
        lexer,
        [&](auto errorInfo) {
            preprocessOptions.logger.log("Pre-processor error at line %i:\n%s", nbl::system::ILogger::ELL_ERROR, errorInfo.mLine, tcpp::ErrorTypeToString(errorInfo.mType).c_str());
        },
        [&](auto path, auto isSystemPath) {
            if (preprocessOptions.includeFinder)
            {
                return getInputStreamInclude(
                    preprocessOptions.includeFinder, m_system.get(), preprocessOptions.maxSelfInclusionCount + 1u,
                    preprocessOptions.sourceIdentifier.data(), path.c_str(), !isSystemPath
                );
            }
            else
            {
                return static_cast<tcpp::IInputStream*>(new tcpp::StringInputStream(std::string("#error No include handler")));
            }
        }
    );

    proc.AddCustomDirectiveHandler(std::string("pragma shader_stage"), [&](tcpp::Preprocessor& preprocessor, tcpp::Lexer& lexer, const std::string& text) {
        if (!lexer.HasNextToken()) return std::string("#error Malformed shader_stage pragma");
        auto token = lexer.GetNextToken();
        if (token.mType != tcpp::E_TOKEN_TYPE::OPEN_BRACKET) return std::string("#error Malformed shader_stage pragma");

        if (!lexer.HasNextToken()) return std::string("#error Malformed shader_stage pragma");
        token = lexer.GetNextToken();
        if (token.mType != tcpp::E_TOKEN_TYPE::IDENTIFIER) return std::string("#error Malformed shader_stage pragma");

        auto& shaderStageIdentifier = token.mRawView;
        core::unordered_map<std::string, IShader::E_SHADER_STAGE> stageFromIdent = {
            { "vertex", IShader::ESS_VERTEX },
            { "fragment", IShader::ESS_FRAGMENT },
            { "tesscontrol", IShader::ESS_TESSELLATION_CONTROL },
            { "tesseval", IShader::ESS_TESSELLATION_EVALUATION },
            { "geometry", IShader::ESS_GEOMETRY },
            { "compute", IShader::ESS_COMPUTE }
        };

        auto found = stageFromIdent.find(shaderStageIdentifier);
        if (found == stageFromIdent.end())
        {
            return std::string("#error Malformed shader_stage pragma, unknown stage");
        }
        stage = found->second;

        if (!lexer.HasNextToken()) return std::string("#error Malformed shader_stage pragma");
        token = lexer.GetNextToken();
        if (token.mType != tcpp::E_TOKEN_TYPE::CLOSE_BRACKET) return std::string("#error Malformed shader_stage pragma");

        while (lexer.HasNextToken()) {
            auto token = lexer.GetNextToken();
            if (token.mType == tcpp::E_TOKEN_TYPE::NEWLINE) break;
            if (token.mType != tcpp::E_TOKEN_TYPE::SPACE) return std::string("#error Malformed shader_stage pragma");
        }

        return std::string("");
    });

    auto resolvedString = proc.Process();
    IShaderCompiler::reenableDirectives(resolvedString);
    return resolvedString;
}

core::smart_refctd_ptr<ICPUShader> CHLSLCompiler::compileToSPIRV(const char* code, const IShaderCompiler::SCompilerOptions& options) const
{
    auto hlslOptions = option_cast(options);

    if (!code)
    {
        hlslOptions.preprocessorOptions.logger.log("code is nullptr", system::ILogger::ELL_ERROR);
        return nullptr;
    }

    auto stage = hlslOptions.stage;
    auto newCode = preprocessShader(code, stage, hlslOptions.preprocessorOptions);

    // Suffix is the shader model version
    std::wstring targetProfile(L"XX_6_2");

    // Set profile two letter prefix based on stage
    switch (stage) {
    case asset::IShader::ESS_VERTEX:
        targetProfile.replace(0, 2, L"vs");
        break;
    case asset::IShader::ESS_TESSELLATION_CONTROL:
        targetProfile.replace(0, 2, L"ds");
        break;
    case asset::IShader::ESS_TESSELLATION_EVALUATION:
        targetProfile.replace(0, 2, L"hs");
        break;
    case asset::IShader::ESS_GEOMETRY:
        targetProfile.replace(0, 2, L"gs");
        break;
    case asset::IShader::ESS_FRAGMENT:
        targetProfile.replace(0, 2, L"ps");
        break;
    case asset::IShader::ESS_COMPUTE:
        targetProfile.replace(0, 2, L"cs");
        break;
    case asset::IShader::ESS_TASK:
        targetProfile.replace(0, 2, L"as");
        break;
    case asset::IShader::ESS_MESH:
        targetProfile.replace(0, 2, L"ms");
        break;
    default:
        hlslOptions.preprocessorOptions.logger.log("invalid shader stage %i", system::ILogger::ELL_ERROR, stage);
        return nullptr;
    };

    LPCWSTR arguments[] = {
        // These will always be present
        L"-spirv",
        L"-HV", L"2021",
        L"-T", targetProfile.c_str(),

        // These are debug only
        DXC_ARG_DEBUG,
        L"-Qembed_debug",
        L"-fspv-debug=vulkan-with-source",
        L"-fspv-debug=file"
    };

    const uint32_t nonDebugArgs = 5;
    const uint32_t allArgs = nonDebugArgs + 4;

    // const CHLSLCompiler* compiler, nbl::asset::hlsl::impl::DXC* compilerTypes, std::string& source, LPCWSTR* args, uint32_t argCount, const CHLSLCompiler::SOptions& options
    auto compileResult = dxcCompile(
        this, 
        m_dxcCompilerTypes.get(), 
        newCode,
        &arguments[0], 
        hlslOptions.genDebugInfo ? allArgs : nonDebugArgs, 
        hlslOptions
    );

    if (!compileResult.objectBlob)
    {
        return nullptr;
    }

    auto outSpirv = core::make_smart_refctd_ptr<ICPUBuffer>(compileResult.objectBlob->GetBufferSize());
    memcpy(outSpirv->getPointer(), compileResult.objectBlob->GetBufferPointer(), compileResult.objectBlob->GetBufferSize());

    return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(outSpirv), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, hlslOptions.preprocessorOptions.sourceIdentifier.data());
}

void CHLSLCompiler::insertIntoStart(std::string& code, std::ostringstream&& ins) const
{
    code.insert(0u, ins.str());
}
