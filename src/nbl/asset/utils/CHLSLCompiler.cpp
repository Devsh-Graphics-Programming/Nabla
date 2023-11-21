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

#include <boost/wave.hpp>
#include <boost/wave/cpplexer/cpp_lex_token.hpp>
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>

using namespace nbl;
using namespace nbl::asset;
using Microsoft::WRL::ComPtr;

static constexpr const wchar_t* SHADER_MODEL_PROFILE = L"XX_6_7";



namespace nbl::asset::hlsl::impl
{
    struct DXC 
    {
        ComPtr<IDxcUtils> m_dxcUtils;
        ComPtr<IDxcCompiler3> m_dxcCompiler;
    };


    // for including builtins 
    struct load_file_or_builtin_to_string
    {

    template <typename IterContextT>
    class inner
    {
    public:
        template <typename PositionT>
        static void init_iterators(IterContextT& iter_ctx, PositionT const& act_pos, boost::wave::language_support language)
        {
            using iterator_type = typename IterContextT::iterator_type;

            std::string filepath(iter_ctx.filename.begin(), iter_ctx.filename.end());
            auto inclFinder = iter_ctx.ctx.get_hooks().m_preprocessOptions.includeFinder;
            if (inclFinder) 
            {
                std::optional<std::string> result;
                system::path requestingSourceDir(iter_ctx.ctx.get_current_directory().string());
                if (iter_ctx.type == IterContextT::base_type::file_type::system_header) // is it a sys include (#include <...>)?
                    result = inclFinder->getIncludeStandard(requestingSourceDir, filepath);
                else // regular #include "..."
                    result = inclFinder->getIncludeRelative(requestingSourceDir, filepath);

                if (!result)
                    BOOST_WAVE_THROW_CTX(iter_ctx.ctx, boost::wave::preprocess_exception,
                        bad_include_file, iter_ctx.filename.c_str(), act_pos);
                auto& res_str = *result;
                iter_ctx.instring = res_str;
            }
            else // include finder not provided
            {
                auto builtin_pair = nbl::builtin::get_resource_runtime(filepath);
                if (builtin_pair.first) // builtin exists
                {
                    iter_ctx.instring = std::string(builtin_pair.first, builtin_pair.first + builtin_pair.second);
                }
                else // default boost behavior
                {
                    // read in the file
                    boost::filesystem::ifstream instream(iter_ctx.filename.c_str());
                    if (!instream.is_open()) {
                        BOOST_WAVE_THROW_CTX(iter_ctx.ctx, boost::wave::preprocess_exception,
                            bad_include_file, iter_ctx.filename.c_str(), act_pos);
                        return;
                    }
                    instream.unsetf(std::ios::skipws);

                    iter_ctx.instring.assign(
                        std::istreambuf_iterator<char>(instream.rdbuf()),
                        std::istreambuf_iterator<char>());
                }
            }
            iter_ctx.first = iterator_type(
                iter_ctx.instring.begin(), iter_ctx.instring.end(),
                PositionT(iter_ctx.filename), language);
            iter_ctx.last = iterator_type();
        }

    private:
        std::string instring;
    };
    };


    struct custom_preprocessing_hooks : public boost::wave::context_policies::default_preprocessing_hooks
    {

        custom_preprocessing_hooks(const IShaderCompiler::SPreprocessorOptions& _preprocessOptions, IShader::E_SHADER_STAGE& _stage) 
            : m_preprocessOptions(_preprocessOptions), m_stage(_stage) {}

        IShaderCompiler::SPreprocessorOptions m_preprocessOptions;
        IShader::E_SHADER_STAGE m_stage;


        template <typename ContextT>
        bool locate_include_file(ContextT& ctx, std::string& file_path, bool is_system, char const* current_name, std::string& dir_path, std::string& native_name) 
        {
            //on builtin return true
            //default returns false if file does not exist
            //if (builtin::hasPathPrefix(file_path))
            //{
            //    file_path = file_path.substr(builtin::pathPrefix.size() + 1); //trim path
            //    return true;
            //}
            auto inclFinder = m_preprocessOptions.includeFinder;
            dir_path = ctx.get_current_directory().string();
            std::optional<std::string> result;
            if (is_system)
                result = inclFinder->getIncludeStandard(dir_path, file_path);
            else //shaderc_include_type_standard
                result = inclFinder->getIncludeRelative(dir_path, file_path);
            if (!result)
                return false;
            else {
                native_name = file_path;
                return true;
            }

            namespace fs = boost::filesystem;

            fs::path native_path(boost::wave::util::create_path(file_path));
            if (!fs::exists(native_path)) {

                m_preprocessOptions.logger.log("Pre-processor error: Bad include file.\n'%s' does not exist.", nbl::system::ILogger::ELL_ERROR, file_path.c_str());

                return false;
            }
            return true;
        }


        // interpretation of #pragma's of the form 'wave option[(value)]'
        template <typename ContextT, typename ContainerT>
        bool
            interpret_pragma(ContextT const& ctx, ContainerT& pending,
                typename ContextT::token_type const& option, ContainerT const& values,
                typename ContextT::token_type const& act_token)
        {
            auto optionStr = option.get_value().c_str();
            if (strcmp(optionStr, "shader_stage") == 0) 
            {
                auto valueIter = values.begin();
                if (valueIter == values.end()) {
                    m_preprocessOptions.logger.log("Pre-processor error:\nMalformed shader_stage pragma. No shaderstage option given", nbl::system::ILogger::ELL_ERROR);
                    return false;
                }
                auto shaderStageIdentifier = std::string(valueIter->get_value().c_str());
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
                    m_preprocessOptions.logger.log("Pre-processor error:\nMalformed shader_stage pragma. Unknown stage '%s'", nbl::system::ILogger::ELL_ERROR, shaderStageIdentifier);
                    return false;
                }
                valueIter++;
                if (valueIter != values.end()) {
                    m_preprocessOptions.logger.log("Pre-processor error:\nMalformed shader_stage pragma. Too many arguments", nbl::system::ILogger::ELL_ERROR);
                    return false;
                }
                m_stage = found->second;
                return true;
            }
            return false;
        }


        template <typename ContextT, typename ContainerT>
        bool found_error_directive(ContextT const& ctx, ContainerT const& message) {
            m_preprocessOptions.logger.log("Pre-processor error:\n%s", nbl::system::ILogger::ELL_ERROR, message);
            return true;
        }

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

    m_dxcCompilerTypes = new hlsl::impl::DXC{
        utils,
        compiler
    };
}


CHLSLCompiler::~CHLSLCompiler()
{
    delete m_dxcCompilerTypes;
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


DxcCompilationResult dxcCompile(const CHLSLCompiler* compiler, hlsl::impl::DXC* dxc, std::string& source, LPCWSTR* args, uint32_t argCount, const CHLSLCompiler::SOptions& options)
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
    using lex_token_t = boost::wave::cpplexer::lex_token<>;
    using lex_iterator_t = boost::wave::cpplexer::lex_iterator<lex_token_t>;
    using wave_context_t = boost::wave::context<core::string::iterator, lex_iterator_t, hlsl::impl::load_file_or_builtin_to_string, hlsl::impl::custom_preprocessing_hooks>;

    hlsl::impl::custom_preprocessing_hooks hooks(preprocessOptions, stage);
    std::string startingFileIdentifier = std::string("../") + preprocessOptions.sourceIdentifier.data();
    wave_context_t context(code.begin(), code.end(), startingFileIdentifier.data(), hooks);
    auto language = boost::wave::support_cpp20 | boost::wave::support_option_preserve_comments | boost::wave::support_option_emit_line_directives;
    context.set_language(static_cast<boost::wave::language_support>(language));
    context.add_macro_definition("__HLSL_VERSION");
    //TODO fix bad syntax and uncomment
    // instead of defining extraDefines as "NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_1D 32768", 
    // now define them as "NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_1D=32768" 
    // to match boost wave syntax
    // https://www.boost.org/doc/libs/1_82_0/libs/wave/doc/class_reference_context.html#:~:text=Maintain%20defined%20macros-,add_macro_definition,-bool%20add_macro_definition
    for (auto iter = preprocessOptions.extraDefines.begin(); iter != preprocessOptions.extraDefines.end(); iter++)
    {
        std::string s = *iter;
        size_t firstParenthesis = s.find(')');
        if (firstParenthesis == -1) firstParenthesis = 0;
        size_t firstWhitespace = s.find(' ', firstParenthesis);
        if (firstWhitespace != -1)
            s[firstWhitespace] = '=';
        context.add_macro_definition(s);
    }

    // preprocess
    std::stringstream stream = std::stringstream();
    for (auto i = context.begin(); i != context.end(); i++) {
        stream << i->get_value();
        std::cout<< i->get_value();
    }
    core::string resolvedString = stream.str();
    
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
    stage = context.get_hooks().m_stage;
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
