// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/CHLSLCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"
#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/builtin/CArchive.h"
#include "spirv/builtin/CArchive.h"
#endif // NBL_EMBED_BUILTIN_RESOURCES


#ifdef _NBL_PLATFORM_WINDOWS_

#include <wrl.h>
#include <combaseapi.h>

#include <dxc/dxcapi.h>

#include <sstream>
#include <regex>
#include <iterator>
#include <codecvt>

#define TCPP_IMPLEMENTATION
#include <tcpp/source/tcppLibrary.hpp>
#undef TCPP_IMPLEMENTATION

using namespace nbl;
using namespace nbl::asset;
using Microsoft::WRL::ComPtr;

static constexpr const wchar_t* SHADER_MODEL_PROFILE = L"XX_6_7";

namespace nbl::asset::hlsl::impl
{
    struct DXC {
        ComPtr<IDxcUtils> m_dxcUtils;
        ComPtr<IDxcCompiler3> m_dxcCompiler;
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

    m_dxcCompilerTypes = new nbl::asset::hlsl::impl::DXC{
        utils,
        compiler
    };
}

CHLSLCompiler::~CHLSLCompiler()
{
    delete m_dxcCompilerTypes;
}

static tcpp::TInputStreamUniquePtr getInputStreamInclude(
    const IShaderCompiler::CIncludeFinder* inclFinder,
    const system::ISystem* fs,
    uint32_t maxInclCnt,
    const char* requestingSource,
    const char* requestedSource,
    bool isRelative, // true for #include "string"; false for #include <string>
    uint32_t lexerLineIndex,
    uint32_t leadingLinesImports,
    std::vector<std::pair<uint32_t, std::string>>& includeStack
)
{

    std::filesystem::path relDir;
    #ifdef NBL_EMBED_BUILTIN_RESOURCES
    const bool reqFromBuiltin = nbl::builtin::hasPathPrefix(requestingSource) || spirv::builtin::hasPathPrefix(requestingSource);
    const bool reqBuiltin = nbl::builtin::hasPathPrefix(requestedSource) || spirv::builtin::hasPathPrefix(requestedSource);
    if (!reqFromBuiltin && !reqBuiltin)
    {
        //While #includ'ing a builtin, one must specify its full path (starting with "nbl/builtin" or "/nbl/builtin").
        //  This rule applies also while a builtin is #includ`ing another builtin.
        //While including a filesystem file it must be either absolute path (or relative to any search dir added to asset::iIncludeHandler; <>-type),
        //  or path relative to executable's working directory (""-type).
        relDir = std::filesystem::path(requestingSource).parent_path();
    }
    #else
    const bool reqBuiltin = false;
    #endif // NBL_EMBED_BUILTIN_RESOURCES
    std::filesystem::path name = isRelative ? (relDir / requestedSource) : (requestedSource);

    if (std::filesystem::exists(name) && !reqBuiltin)
        name = std::filesystem::absolute(name);

    std::optional<std::string> result;
    if (isRelative)
        result = inclFinder->getIncludeRelative(relDir, requestedSource);
    else //shaderc_include_type_standard
        result = inclFinder->getIncludeStandard(relDir, requestedSource);

    if (!result) 
    {
        /*
            Alternative could be found in the commit 2a66b4b20c579ea730aa3dd8af707847c01def64
            if ever HLSL gets system headers we might just want to let includes be includes and 
            DXC handle everything else we don't know about
        */
        std::string re(IShaderCompiler::PREPROC_DIRECTIVE_DISABLER);
        re.append("error ");
        re.append(requestedSource);
        re.append(" not found");
        includeStack.push_back(includeStack.back());
        return tcpp::TInputStreamUniquePtr(new tcpp::StringInputStream(re));
    }

    // Figure out what line in the current file this #include was
    // That would be the current lexer line, minus the line where the current file was included
    uint32_t lineInCurrentFileWithInclude = lexerLineIndex -
        // if this is 2 includes deep (include within include), subtract leading import lines
        // from the previous include
        (includeStack.size() > 1 ? leadingLinesImports : 0);
    auto lastItemInIncludeStack = includeStack.back();

    auto& res_str = *result;
    IShaderCompiler::disableAllDirectivesExceptIncludes(res_str);
    res_str = IShaderCompiler::encloseWithinExtraInclGuards(std::move(res_str), maxInclCnt, name.string().c_str());
    res_str = res_str + "\n" +
        IShaderCompiler::PREPROC_DIRECTIVE_DISABLER + "line " + std::to_string(lineInCurrentFileWithInclude - lastItemInIncludeStack.first - 1).c_str() + " \"" + lastItemInIncludeStack.second.c_str() + "\"\n";

    // Offset the lines this include takes up for subsequent includes
    includeStack.back().first += std::count(res_str.begin(), res_str.end(), '\n');

    includeStack.push_back(std::pair<uint32_t, std::string>(lineInCurrentFileWithInclude, IShaderCompiler::escapeFilename(name.string())));

    return tcpp::TInputStreamUniquePtr(new tcpp::StringInputStream(std::move(res_str)));
}

class DxcCompilationResult
{
public:
    ComPtr<IDxcBlobEncoding> errorMessages;
    ComPtr<IDxcBlob> objectBlob;
    ComPtr<IDxcResult> compileResult;

    std::string GetErrorMessagesString()
    {
        return std::string(reinterpret_cast<char*>(errorMessages->GetBufferPointer()), errorMessages->GetBufferSize());
    }
};

DxcCompilationResult dxcCompile(const CHLSLCompiler* compiler, nbl::asset::hlsl::impl::DXC* dxc, std::string& source, LPCWSTR* args, uint32_t argCount, const CHLSLCompiler::SOptions& options)
{
    // Append Commandline options into source only if debugInfoFlags will emit source
    auto sourceEmittingFlags =
        CHLSLCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT |
        CHLSLCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT |
        CHLSLCompiler::E_DEBUG_INFO_FLAGS::EDIF_NON_SEMANTIC_BIT;
    if ((options.debugInfoFlags.value & sourceEmittingFlags) != CHLSLCompiler::E_DEBUG_INFO_FLAGS::EDIF_NONE)
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

    auto errorMessagesString = result.GetErrorMessagesString();
    if (SUCCEEDED(compilationStatus))
    {
        if (errorMessagesString.length() > 0)
        {
            options.preprocessorOptions.logger.log("DXC Compilation Warnings:\n%s", system::ILogger::ELL_WARNING, errorMessagesString.c_str());
        }
    } 
    else
    {
        options.preprocessorOptions.logger.log("DXC Compilation Failed:\n%s", system::ILogger::ELL_ERROR, errorMessagesString.c_str());
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
    // Line 1 comes before all the extra defines in the main shader
    insertIntoStart(code, std::ostringstream(std::string(IShaderCompiler::PREPROC_DIRECTIVE_ENABLER) + "line 1\n"));

    uint32_t defineLeadingLinesMain = 1;
    uint32_t leadingLinesImports = IShaderCompiler::encloseWithinExtraInclGuardsLeadingLines(preprocessOptions.maxSelfInclusionCount + 1u);
    if (preprocessOptions.extraDefines.size())
    {
        insertExtraDefines(code, preprocessOptions.extraDefines);
        defineLeadingLinesMain += preprocessOptions.extraDefines.size();
    }

    IShaderCompiler::disableAllDirectivesExceptIncludes(code);

    // Keep track of the line in the original file where each #include was on each level of the include stack
    std::vector<std::pair<uint32_t, std::string>> lineOffsetStack = { std::pair<uint32_t, std::string>(defineLeadingLinesMain, preprocessOptions.sourceIdentifier) };
    
    tcpp::Lexer lexer(tcpp::TInputStreamUniquePtr(new tcpp::StringInputStream(code)));

    tcpp::Preprocessor::TPreprocessorConfigInfo info;
    {
        info.mOnErrorCallback = [&](auto errorInfo) {
            preprocessOptions.logger.log("Pre-processor error at line %i:\n%s", nbl::system::ILogger::ELL_ERROR, errorInfo.mLine, tcpp::ErrorTypeToString(errorInfo.mType).c_str());
        };
        info.mOnIncludeCallback = [&](auto path, auto isSystemPath) {
            if (preprocessOptions.includeFinder)
            {
                return getInputStreamInclude(
                    preprocessOptions.includeFinder, m_system.get(), preprocessOptions.maxSelfInclusionCount + 1u,
                    preprocessOptions.sourceIdentifier.data(), path.c_str(), !isSystemPath,
                    lexer.GetCurrLineIndex(), leadingLinesImports, lineOffsetStack
                );
            }
            else
            {
                return tcpp::TInputStreamUniquePtr(new tcpp::StringInputStream(std::string("#error No include handler")));
            }
        };
        info.mOnPopIncludeCallback = [&]() {
            lineOffsetStack.pop_back();
        };
    }
    tcpp::Preprocessor proc(
        lexer, info        
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
    // TODO: Figure out a way to get the shader model version automatically
    // 
    // We can't get it from the DXC library itself, as the different versions and the parsing
    // use a weird lexer based system that resolves to a hash, and all of that is in a scoped variable
    // (lib/DXIL/DxilShaderModel.cpp:83)
    // 
    // Another option is trying to fetch it from the commandline tool, either from parsing the help message
    // or from brute forcing every -T option until one isn't accepted
    //
    std::wstring targetProfile(SHADER_MODEL_PROFILE);

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

    std::vector<LPCWSTR> arguments = {
        L"-spirv",
        L"-HV", L"202x",
        L"-T", targetProfile.c_str(),
        L"-Zpr", // Packs matrices in row-major order by default
        L"-enable-16bit-types",
        L"-fvk-use-scalar-layout",
        L"-Wno-c++11-extensions"
    };

    // If a custom SPIR-V optimizer is specified, use that instead of DXC's spirv-opt.
    // This is how we can get more optimizer options.
    // 
    // Optimization is also delegated to SPIRV-Tools. Right now there are no difference between 
    // optimization levels greater than zero; they will all invoke the same optimization recipe. 
    // https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst#optimization
    if (hlslOptions.spirvOptimizer)
    {
        arguments.push_back(L"-O0");
    }

    // Debug only values
    if (hlslOptions.debugInfoFlags.hasFlags(E_DEBUG_INFO_FLAGS::EDIF_FILE_BIT))
        arguments.push_back(L"-fspv-debug=file");
    if (hlslOptions.debugInfoFlags.hasFlags(E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT))
        arguments.push_back(L"-fspv-debug=source");
    if (hlslOptions.debugInfoFlags.hasFlags(E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT))
        arguments.push_back(L"-fspv-debug=line");
    if (hlslOptions.debugInfoFlags.hasFlags(E_DEBUG_INFO_FLAGS::EDIF_TOOL_BIT))
        arguments.push_back(L"-fspv-debug=tool");
    if (hlslOptions.debugInfoFlags.hasFlags(E_DEBUG_INFO_FLAGS::EDIF_NON_SEMANTIC_BIT))
        arguments.push_back(L"-fspv-debug=vulkan-with-source");

    auto compileResult = dxcCompile(
        this, 
        m_dxcCompilerTypes, 
        newCode,
        &arguments[0],
        arguments.size(),
        hlslOptions
    );

    if (!compileResult.objectBlob)
    {
        return nullptr;
    }

    auto outSpirv = core::make_smart_refctd_ptr<ICPUBuffer>(compileResult.objectBlob->GetBufferSize());
    memcpy(outSpirv->getPointer(), compileResult.objectBlob->GetBufferPointer(), compileResult.objectBlob->GetBufferSize());
    
    // Optimizer step
    if (hlslOptions.spirvOptimizer)
        outSpirv = hlslOptions.spirvOptimizer->optimize(outSpirv.get(), hlslOptions.preprocessorOptions.logger);


    return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(outSpirv), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, hlslOptions.preprocessorOptions.sourceIdentifier.data());
}

void CHLSLCompiler::insertIntoStart(std::string& code, std::ostringstream&& ins) const
{
    code.insert(0u, ins.str());
}

#endif
