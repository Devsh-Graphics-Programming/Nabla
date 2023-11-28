// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/CHLSLCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"
// TODO: review
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


using namespace nbl;
using namespace nbl::asset;
using Microsoft::WRL::ComPtr;

static constexpr const wchar_t* SHADER_MODEL_PROFILE = L"XX_6_7";


namespace nbl::asset::impl
{
struct DXC 
{
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

    m_dxcCompilerTypes = new impl::DXC{
        utils,
        compiler
    };
}

CHLSLCompiler::~CHLSLCompiler()
{
    delete m_dxcCompilerTypes;
}


struct DxcCompilationResult
{
    ComPtr<IDxcBlobEncoding> errorMessages;
    ComPtr<IDxcBlob> objectBlob;
    ComPtr<IDxcResult> compileResult;

    std::string GetErrorMessagesString()
    {
        return std::string(reinterpret_cast<char*>(errorMessages->GetBufferPointer()), errorMessages->GetBufferSize());
    }
};

DxcCompilationResult dxcCompile(const CHLSLCompiler* compiler, nbl::asset::impl::DXC* dxc, std::string& source, LPCWSTR* args, uint32_t argCount, const CHLSLCompiler::SOptions& options)
{
    // Append Commandline options into source only if debugInfoFlags will emit source
    auto sourceEmittingFlags =
        CHLSLCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT |
        CHLSLCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT |
        CHLSLCompiler::E_DEBUG_INFO_FLAGS::EDIF_NON_SEMANTIC_BIT;
    if ((options.debugInfoFlags.value & sourceEmittingFlags) != CHLSLCompiler::E_DEBUG_INFO_FLAGS::EDIF_NONE)
    {
        std::ostringstream insertion;
        insertion << "//#pragma compile_flags ";

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


#include "nbl/asset/utils/waveContext.h"
using wave_context_t = nbl::wave::context<std::string::iterator>;

std::string CHLSLCompiler::preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions) const
{
    wave_context_t context(code.begin(),code.end(),preprocessOptions.sourceIdentifier.data(),{preprocessOptions});
    auto language = boost::wave::support_cpp20 | boost::wave::support_option_preserve_comments | boost::wave::support_option_emit_line_directives;
    context.set_language(static_cast<boost::wave::language_support>(language));
    context.add_macro_definition("__HLSL_VERSION");
   
     // instead of defining extraDefines as "NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_1D 32768", 
    // now define them as "NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_1D=32768" 
    // to match boost wave syntax
    // https://www.boost.org/doc/libs/1_82_0/libs/wave/doc/class_reference_context.html#:~:text=Maintain%20defined%20macros-,add_macro_definition,-bool%20add_macro_definition
    for (const auto& define : preprocessOptions.extraDefines)
        context.add_macro_definition(define.identifier.data()+core::string("=")+define.definition.data());

    // preprocess
    core::string resolvedString;
    try
    {
        std::stringstream stream = std::stringstream();
        for (auto i=context.begin(); i!=context.end(); i++)
            stream << i->get_value();
        resolvedString = stream.str();
    }
    catch (...)
    {
        preprocessOptions.logger.log("Boost.Wave exception caught!",system::ILogger::ELL_ERROR);
    }
    
    // for debugging cause MSVC doesn't like to show more than 21k LoC in TextVisualizer
    if constexpr (true)
    {
        system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
        m_system->createFile(future,system::path(preprocessOptions.sourceIdentifier).parent_path()/"preprocessed.hlsl",system::IFileBase::ECF_WRITE);
        if (auto file=future.acquire(); file&&bool(*file))
        {
            system::IFile::success_t succ;
            (*file)->write(succ,resolvedString.data(),0,resolvedString.size()+1);
            succ.getBytesProcessed(true);
        }
    }

    if(context.get_hooks().m_pragmaStage != IShader::ESS_UNKNOWN)
        stage = context.get_hooks().m_pragmaStage;

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
    switch (stage)
    {
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
        L"-Wno-c++11-extensions",
        L"-Wno-c++1z-extensions",
        L"-Wno-gnu-static-float-init",
        L"-fspv-target-env=vulkan1.3"
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