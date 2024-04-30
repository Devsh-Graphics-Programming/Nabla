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

#include <regex>
#include <iterator>
#include <codecvt>
#include <wrl.h>
#include <combaseapi.h>
#include <sstream>
#include <dxc/dxcapi.h>

using namespace nbl;
using namespace nbl::asset;
using Microsoft::WRL::ComPtr;

static constexpr const wchar_t* SHADER_MODEL_PROFILE = L"XX_6_7";
static const wchar_t* ShaderStageToString(asset::IShader::E_SHADER_STAGE stage) {
    switch (stage)
    {
    case asset::IShader::ESS_VERTEX:
        return L"vs";
    case asset::IShader::ESS_TESSELLATION_CONTROL:
        return L"ds";
    case asset::IShader::ESS_TESSELLATION_EVALUATION:
        return L"hs";
    case asset::IShader::ESS_GEOMETRY:
        return L"gs";
    case asset::IShader::ESS_FRAGMENT:
        return L"ps";
    case asset::IShader::ESS_COMPUTE:
        return L"cs";
    case asset::IShader::ESS_TASK:
        return L"as";
    case asset::IShader::ESS_MESH:
        return L"ms";
    default:
        return nullptr;
    };
}

namespace nbl::asset::impl
{
struct DXC 
{
    ComPtr<IDxcUtils> m_dxcUtils;
    ComPtr<IDxcCompiler3> m_dxcCompiler;
};
}

struct DxcCompilationResult
{
    Microsoft::WRL::ComPtr<IDxcBlobEncoding> errorMessages;
    Microsoft::WRL::ComPtr<IDxcBlob> objectBlob;
    Microsoft::WRL::ComPtr<IDxcResult> compileResult;

    std::string GetErrorMessagesString()
    {
        return std::string(reinterpret_cast<char*>(errorMessages->GetBufferPointer()), errorMessages->GetBufferSize());
    }
};


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


static void try_upgrade_hlsl_version(std::vector<std::wstring>& arguments, system::logger_opt_ptr& logger)
{
    auto stageArgumentPos = std::find(arguments.begin(), arguments.end(), L"-HV");
    if (stageArgumentPos != arguments.end() && stageArgumentPos + 1 != arguments.end())
    {
        std::wstring version = *(++stageArgumentPos);
        if (!isalpha(version.back()))
        {
            try
            {
                if (version.length() != 4)
                    throw std::invalid_argument("-HV argument is of incorrect length, expeciting 4. Fallign back to 2021");
                if (std::stoi(version) < 2021)
                    throw std::invalid_argument("-HV argument is too low");
            }
            catch (const std::exception& ex)
            {
                version = L"2021";
            }
        }
        *stageArgumentPos = version;
    }
    else
    {
        logger.log("Compile flag error: Required compile flag not found -HV. Force enabling -HV 202x, as it is required by Nabla.", system::ILogger::ELL_WARNING);
        arguments.push_back(L"-HV");
        arguments.push_back(L"202x");
    }
}

static void try_upgrade_shader_stage(std::vector<std::wstring>& arguments, asset::IShader::E_SHADER_STAGE shaderStageOverrideFromPragma, system::logger_opt_ptr& logger) {

    constexpr int MajorReqVersion = 6, MinorReqVersion = 7;
    auto overrideStageStr = ShaderStageToString(shaderStageOverrideFromPragma);
    if (shaderStageOverrideFromPragma != IShader::ESS_UNKNOWN && !overrideStageStr)
    {
        logger.log("Invalid shader stage with int value '%i'.\nThis value does not have a known string representation.",
            system::ILogger::ELL_ERROR, shaderStageOverrideFromPragma);
        return;
    }
    bool setDefaultValue = true;
    auto foundShaderStageArgument = std::find(arguments.begin(), arguments.end(), L"-T");
    if (foundShaderStageArgument != arguments.end() && foundShaderStageArgument +1 != arguments.end()) {
        auto foundShaderStageArgumentValueIdx = foundShaderStageArgument - arguments.begin() + 1;
        std::wstring stageStr;
        std::wstring s = arguments[foundShaderStageArgumentValueIdx];
        if (s.length() >= 6) {
            std::vector<std::wstring::iterator> underscorePositions = {};
            auto it = std::find(s.begin(), s.end(), '_');
            while (it != s.end()) {
                underscorePositions.push_back(it);
                it = std::find(it + 1, s.end(), '_');
            }

            if (underscorePositions.size() == 2) 
            {   
                stageStr = shaderStageOverrideFromPragma != IShader::ESS_UNKNOWN ? std::wstring(overrideStageStr) : std::wstring(s.begin(), underscorePositions[0]);
                // Version
                std::wstring majorVersionString, minorVersionString;
                int size = underscorePositions.size();
                auto secondLastUnderscore = underscorePositions[size - 2];
                auto lastUnderscore = underscorePositions[size - 1];
                majorVersionString = std::wstring(secondLastUnderscore + 1, lastUnderscore);
                minorVersionString = std::wstring(lastUnderscore + 1, s.end());
                try
                {
                    int major = std::stoi(majorVersionString);
                    int minor = std::stoi(minorVersionString);
                    if (major < MajorReqVersion || (major == MajorReqVersion && minor < MinorReqVersion))
                    {
                        // Overwrite the version 
                        logger.log("Upgrading shader stage version number to %i %i", system::ILogger::ELL_DEBUG, MajorReqVersion, MinorReqVersion);
                        arguments[foundShaderStageArgumentValueIdx] = stageStr + L"_" + std::to_wstring(MajorReqVersion) + L"_" + std::to_wstring(MinorReqVersion);
                    }
                    else
                    {
                        // keep the version as it was
                        arguments[foundShaderStageArgumentValueIdx] = stageStr + L"_" + majorVersionString + L"_" + minorVersionString;
                    }
                    return;
                }
                catch (const std::invalid_argument& e)
                {
                    logger.log("Parsing shader version failed, invalid argument exception: %s", system::ILogger::ELL_ERROR, e.what());
                }
                catch (const std::out_of_range& e)
                {
                    logger.log("Parsing shader version failed, out of range exception: %s", system::ILogger::ELL_ERROR, e.what());
                }
            }
            else
            {
                logger.log("Incorrect -T argument value.\nExpecting string with exactly 2 '_' delimiters: between shader stage, version major and version minor.",
                    system::ILogger::ELL_ERROR);
            }
        }
        else 
        {
            logger.log("invalid shader stage '%s' argument, expecting a string of length >= 6 ", system::ILogger::ELL_ERROR, s);
        } 
        // In case of an exception or str < 6
        arguments[foundShaderStageArgumentValueIdx] = stageStr + L"_" + std::to_wstring(MajorReqVersion) + L"_" + std::to_wstring(MinorReqVersion);
        setDefaultValue = false;
    }
    if (setDefaultValue) 
    { 
        // in case of no -T
        // push back default values for -T argument
        // can be safely pushed to the back of argument list as output files should be evicted from args before passing to this func
        // leaving only compiler flags
        arguments.push_back(L"-T");
        arguments.push_back(std::wstring(overrideStageStr) + L"_" + std::to_wstring(MajorReqVersion) + L"_" + std::to_wstring(MinorReqVersion));
    }
}

static void add_required_arguments_if_not_present(std::vector<std::wstring>& arguments, system::logger_opt_ptr &logger) {
    auto set = std::unordered_set<std::wstring>();
    for (int i = 0; i < arguments.size(); i++)
        set.insert(arguments[i]);
    for (int j = 0; j < CHLSLCompiler::RequiredArgumentCount; j++)
    {
        bool missing = set.find(CHLSLCompiler::RequiredArguments[j]) == set.end();
        if (missing) {
            logger.log("Compile flag error: Required compile flag not found %ls. This flag will be force enabled, as it is required by Nabla.", system::ILogger::ELL_WARNING, CHLSLCompiler::RequiredArguments[j]);
            arguments.push_back(CHLSLCompiler::RequiredArguments[j]);
        }
    }
}

// adds missing required arguments
// converts arguments from std::string to std::wstring
template <typename T>
static void populate_arguments_with_type_conversion(std::vector<std::wstring> &arguments, T &iterable_collection, system::logger_opt_ptr &logger) {
    size_t arg_size = iterable_collection.size();
    arguments.reserve(arg_size);
    for (auto it = iterable_collection.begin(); it != iterable_collection.end(); it++) {
        auto temp = std::wstring(it->begin(), it->end()); // *it is string, therefore create wstring
        arguments.push_back(temp);
    }
    add_required_arguments_if_not_present(arguments, logger);
}


static DxcCompilationResult dxcCompile(const CHLSLCompiler* compiler, nbl::asset::impl::DXC* dxc, std::string& source, LPCWSTR* args, uint32_t argCount, const CHLSLCompiler::SOptions& options)
{
    // Emit compile flags as a #pragma directive
    // "#pragma wave dxc_compile_flags allows" intended use is to be able to recompile a shader with the same* flags as initial compilation
    // mainly meant to be used while debugging in RenderDoc.
    // * (except "-no-nbl-builtins") 
    {
        std::ostringstream insertion;
        insertion << "#pragma wave dxc_compile_flags( ";

        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> conv;
        for (uint32_t arg = 0; arg < argCount; arg ++)
        {
            auto str = conv.to_bytes(args[arg]);
            insertion << str.c_str() << " ";
        }

        insertion << ")\n";
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


std::string CHLSLCompiler::preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions, std::vector<std::string>& dxc_compile_flags_override, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies) const
{
    nbl::wave::context context(code.begin(),code.end(),preprocessOptions.sourceIdentifier.data(),{preprocessOptions});
    // If dependencies were passed, we assume we want caching
    context.set_caching(bool(dependencies));
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
    catch (boost::wave::preprocess_exception& e)
    {
        preprocessOptions.logger.log("Boost.Wave %s exception caught!",system::ILogger::ELL_ERROR,e.what());
    }
    catch (...)
    {
        preprocessOptions.logger.log("Unknown exception caught!",system::ILogger::ELL_ERROR);
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

    if (context.get_hooks().m_dxc_compile_flags_override.size() != 0)
        dxc_compile_flags_override = context.get_hooks().m_dxc_compile_flags_override;

    if(context.get_hooks().m_pragmaStage != IShader::ESS_UNKNOWN)
        stage = context.get_hooks().m_pragmaStage;

    if (dependencies) {
        *dependencies = std::move(context.get_dependencies());
    }

    return resolvedString;
}

std::string CHLSLCompiler::preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies) const
{
    std::vector<std::string> extra_dxc_compile_flags = {};
    return preprocessShader(std::move(code), stage, preprocessOptions, extra_dxc_compile_flags);
}

core::smart_refctd_ptr<ICPUShader> CHLSLCompiler::compileToSPIRV_impl(const std::string_view code, const IShaderCompiler::SCompilerOptions& options, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies) const
{
    auto hlslOptions = option_cast(options);
    auto logger = hlslOptions.preprocessorOptions.logger;
    if (code.empty())
    {
        logger.log("code is nullptr", system::ILogger::ELL_ERROR);
        return nullptr;
    }
    std::vector<std::string> dxc_compile_flags = {};
    IShader::E_SHADER_STAGE stage = options.stage;
    auto newCode = preprocessShader(std::string(code), stage, hlslOptions.preprocessorOptions, dxc_compile_flags, dependencies);

    // Suffix is the shader model version
    std::wstring targetProfile(SHADER_MODEL_PROFILE);
   
    std::vector<std::wstring> arguments = {};
    if (dxc_compile_flags.size()) { // #pragma dxc_compile_flags takes priority
        populate_arguments_with_type_conversion(arguments, dxc_compile_flags, logger);
    }
    else if (hlslOptions.dxcOptions.size()) { // second in order of priority is command line arguments
        populate_arguments_with_type_conversion(arguments, hlslOptions.dxcOptions, logger);
    }
    else { // lastly default arguments
        arguments = {};
        for (size_t i = 0; i < RequiredArgumentCount; i++)
            arguments.push_back(RequiredArguments[i]);
        arguments.push_back(L"-HV");
        arguments.push_back(L"202x");
        // TODO: add this to `CHLSLCompiler::SOptions` and handle it properly in `dxc_compile_flags.empty()`
        arguments.push_back(L"-E");
        arguments.push_back(L"main");
        // If a custom SPIR-V optimizer is specified, use that instead of DXC's spirv-opt.
        // This is how we can get more optimizer options.
        // 
        // Optimization is also delegated to SPIRV-Tools. Right now there are no difference between 
        // optimization levels greater than zero; they will all invoke the same optimization recipe. 
        // https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst#optimization
        if (hlslOptions.spirvOptimizer)
            arguments.push_back(L"-O0");
    }
    if (dxc_compile_flags.empty())
    {
        auto set = std::unordered_set<std::wstring>();
        for (int i = 0; i < arguments.size(); i++)
            set.insert(arguments[i]);
        auto add_if_missing = [&arguments, &set, logger](std::wstring flag) {
            if (set.find(flag) == set.end()) {
                logger.log("Adding debug flag %ls", nbl::system::ILogger::ELL_DEBUG, flag.c_str());
                arguments.push_back(flag);
            }
        };
        // Debug only values
        if (hlslOptions.debugInfoFlags.hasFlags(E_DEBUG_INFO_FLAGS::EDIF_FILE_BIT))
            add_if_missing(L"-fspv-debug=file");
        if (hlslOptions.debugInfoFlags.hasFlags(E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT))
            add_if_missing(L"-fspv-debug=source");
        if (hlslOptions.debugInfoFlags.hasFlags(E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT))
            add_if_missing(L"-fspv-debug=line");
        if (hlslOptions.debugInfoFlags.hasFlags(E_DEBUG_INFO_FLAGS::EDIF_TOOL_BIT))
            add_if_missing(L"-fspv-debug=tool");
        if (hlslOptions.debugInfoFlags.hasFlags(E_DEBUG_INFO_FLAGS::EDIF_NON_SEMANTIC_BIT))
            add_if_missing(L"-fspv-debug=vulkan-with-source");
    }

    try_upgrade_shader_stage(arguments, stage, logger);
    try_upgrade_hlsl_version(arguments, logger);
    
    uint32_t argc = arguments.size();
    LPCWSTR* argsArray = new LPCWSTR[argc];
    for (size_t i = 0; i < argc; i++)
        argsArray[i] = arguments[i].c_str();
    
    auto compileResult = dxcCompile( 
        this,
        m_dxcCompilerTypes,
        newCode,
        argsArray,
        argc,
        hlslOptions
    );

    if (argsArray)
        delete[] argsArray;

    if (!compileResult.objectBlob)
    {
        return nullptr;
    }

    auto outSpirv = core::make_smart_refctd_ptr<ICPUBuffer>(compileResult.objectBlob->GetBufferSize());
    memcpy(outSpirv->getPointer(), compileResult.objectBlob->GetBufferPointer(), compileResult.objectBlob->GetBufferSize());
    
    // Optimizer step
    if (hlslOptions.spirvOptimizer)
        outSpirv = hlslOptions.spirvOptimizer->optimize(outSpirv.get(), logger);

    return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(outSpirv), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, hlslOptions.preprocessorOptions.sourceIdentifier.data());
}


void CHLSLCompiler::insertIntoStart(std::string& code, std::ostringstream&& ins) const
{
    code.insert(0u, ins.str());
}


#endif