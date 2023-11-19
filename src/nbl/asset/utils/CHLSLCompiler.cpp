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

//#define TCPP_IMPLEMENTATION
//#include <tcpp/source/tcppLibrary.hpp>
//#undef TCPP_IMPLEMENTATION

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


    // for including builtins 
    struct load_file_or_builtin_to_string
    {
        template <typename IterContextT>
        class inner
        {
        public:
            template <typename PositionT>
            static void init_iterators(IterContextT& iter_ctx,
                PositionT const& act_pos, boost::wave::language_support language)
            {
                typedef typename IterContextT::iterator_type iterator_type;

                std::string s(iter_ctx.filename.begin(), iter_ctx.filename.end());
                auto icf = iter_ctx.ctx.get_hooks().preprocessOptions.includeFinder; // TODO it's null?
                //std::optional<std::string> st = icf.getIncludeStandard(p, s);
                //unfortunately have to call get_resource_runtime here again
                auto builtin_pair = nbl::builtin::get_resource_runtime(s);
                if (builtin_pair.first) //not a nullptr
                {
                    iter_ctx.instring = std::string(builtin_pair.first, builtin_pair.first+builtin_pair.second);
                }
                else {
                    //TODO use nbl filesystem
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
                iter_ctx.first = iterator_type(
                    iter_ctx.instring.begin(), iter_ctx.instring.end(),
                    PositionT(iter_ctx.filename), language);
                iter_ctx.last = iterator_type();
            }

        private:
            std::string instring; //todo change to IReferenceCounted file
        };
    };

    struct custom_preprocessing_hooks : public boost::wave::context_policies::default_preprocessing_hooks{

        custom_preprocessing_hooks(const nbl::asset::IShaderCompiler::SPreprocessorOptions& _preprocessOptions, std::vector<std::pair<uint32_t, std::string>>&& _lineOffsetStack) : preprocessOptions(_preprocessOptions), lineOffsetStack(_lineOffsetStack) {}

        nbl::asset::IShaderCompiler::SPreprocessorOptions preprocessOptions;
        std::vector<std::pair<uint32_t, std::string>> lineOffsetStack;  //TODO no longer needed


        template <typename ContextT>
        bool
            locate_include_file(ContextT& ctx, std::string& file_path,
                bool is_system, char const* current_name, std::string& dir_path,
                std::string& native_name) {

            //on builtin return true
            //default returns false if file does not exist
            if (nbl::builtin::hasPathPrefix(file_path))
            {
                file_path = file_path.substr(nbl::builtin::pathPrefix.size() + 1);
                return true;
            }
            if (!ctx.find_include_file(file_path, dir_path, is_system, current_name))
                return false;   // could not locate file

            namespace fs = boost::filesystem;

            fs::path native_path(boost::wave::util::create_path(file_path));
            if (!fs::exists(native_path)) {
                //TODO throw error or log message?
                /*BOOST_WAVE_THROW_CTX(ctx, boost::wave::preprocess_exception, boost::wave::bad_include_file,
                    file_path.c_str(), ctx.get_main_pos());*/
                //preprocessOptions.logger.log("Pre-processor error ...:\n%s", nbl::system::ILogger::ELL_ERROR, message);

                return false;
            }
            return true;
        }

        //template <typename ContextT>
        //void opened_include_file(ContextT const& ctx, std::string const& relname, std::string const& absname, bool is_system_include) {
        //    //if (preprocessOptions.includeFinder)
        //    //{
        //    //    return getInputStreamInclude(
        //    //        preprocessOptions.includeFinder, m_system.get(), preprocessOptions.maxSelfInclusionCount + 1u,
        //    //        preprocessOptions.sourceIdentifier.data(), path.c_str(), !isSystemPath,
        //    //        lexer.GetCurrLineIndex(), leadingLinesImports, lineOffsetStack
        //    //    );
        //    //}
        //    //else
        //    //{
        //    //    lineOffsetStack.push_back(lineOffsetStack.back()); // have to push something because pop-include expects something
        //    //    return tcpp::TInputStreamUniquePtr(new tcpp::StringInputStream(std::string("#error No include handler\n")));
        //    //}
        //    _NBL_DEBUG_BREAK_IF(true);
        //}



        // interpretation of #pragma's of the form 
        // 'wave option[(value)]'
        template <typename ContextT, typename ContainerT>
        bool interpret_pragma(ContextT const& ctx, ContainerT& pending, typename ContextT::token_type const& option, ContainerT const& values, typename ContextT::token_type const& pragma_token) {
            //"pragma shader_stage"
           /* if (lexer.HasNextToken()) return std::string("#error Malformed shader_stage pragma");
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
            }*/

            return true;
        }


        template <typename ContextT, typename ContainerT>
        bool found_error_directive(ContextT const& ctx, ContainerT const& message) {
            preprocessOptions.logger.log("Pre-processor error:\n%s", nbl::system::ILogger::ELL_ERROR, message);
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

    m_dxcCompilerTypes = new nbl::asset::hlsl::impl::DXC{
        utils,
        compiler
    };
}

CHLSLCompiler::~CHLSLCompiler()
{
    delete m_dxcCompilerTypes;
}
//
//static std::string getInputStreamInclude(
//    const IShaderCompiler::CIncludeFinder* inclFinder,
//    const system::ISystem* fs,
//    uint32_t maxInclCnt,
//    const char* requestingSource,
//    const char* requestedSource,
//    bool isRelative, // true for #include "string"; false for #include <string>
//    uint32_t lexerLineIndex,
//    uint32_t leadingLinesImports,
//    std::vector<std::pair<uint32_t, std::string>>& includeStack
//)
//{
//    std::filesystem::path relDir;
//    #ifdef NBL_EMBED_BUILTIN_RESOURCES
//    const bool reqFromBuiltin = nbl::builtin::hasPathPrefix(requestingSource) || spirv::builtin::hasPathPrefix(requestingSource);
//    const bool reqBuiltin = nbl::builtin::hasPathPrefix(requestedSource) || spirv::builtin::hasPathPrefix(requestedSource);
//    if (!reqFromBuiltin && !reqBuiltin)
//    {
//        //While #includ'ing a builtin, one must specify its full path (starting with "nbl/builtin" or "/nbl/builtin").
//        //  This rule applies also while a builtin is #includ`ing another builtin.
//        //While including a filesystem file it must be either absolute path (or relative to any search dir added to asset::iIncludeHandler; <>-type),
//        //  or path relative to executable's working directory (""-type).
//        relDir = std::filesystem::path(requestingSource).parent_path();
//    }
//    #else
//    const bool reqBuiltin = false;
//    #endif // NBL_EMBED_BUILTIN_RESOURCES
//    std::filesystem::path name = isRelative ? (relDir / requestedSource) : (requestedSource);
//
//    if (std::filesystem::exists(name) && !reqBuiltin)
//        name = std::filesystem::absolute(name);
//
//    std::optional<std::string> result;
//    if (isRelative)
//        result = inclFinder->getIncludeRelative(relDir, requestedSource);
//    else //shaderc_include_type_standard
//        result = inclFinder->getIncludeStandard(relDir, requestedSource);
//
//    if (!result) 
//    {
//        /*
//            Alternative could be found in the commit 2a66b4b20c579ea730aa3dd8af707847c01def64
//            if ever HLSL gets system headers we might just want to let includes be includes and 
//            DXC handle everything else we don't know about
//        */
//        std::string re(IShaderCompiler::PREPROC_DIRECTIVE_DISABLER);
//        re.append("error ");
//        re.append(requestedSource);
//        re.append(" not found\n");
//        includeStack.push_back(includeStack.back());
//        return re;
//    }
//
//    // Figure out what line in the current file this #include was
//    // That would be the current lexer line, minus the line where the current file was included
//    uint32_t lineInCurrentFileWithInclude = lexerLineIndex -
//        // if this is 2 includes deep (include within include), subtract leading import lines
//        // from the previous include
//        (includeStack.size() > 1 ? leadingLinesImports : 0);
//    auto lastItemInIncludeStack = includeStack.back();
//
//    auto& res_str = *result;
//    IShaderCompiler::disableAllDirectivesExceptIncludes(res_str);
//    res_str = IShaderCompiler::encloseWithinExtraInclGuards(std::move(res_str), maxInclCnt, name.string().c_str());
//    res_str = res_str + "\n" +
//        IShaderCompiler::PREPROC_DIRECTIVE_DISABLER + "line " + std::to_string(lineInCurrentFileWithInclude - lastItemInIncludeStack.first - 1).c_str() + " \"" + lastItemInIncludeStack.second.c_str() + "\"\n";
//
//    // Offset the lines this include takes up for subsequent includes
//    includeStack.back().first += std::count(res_str.begin(), res_str.end(), '\n');
//
//    includeStack.push_back(std::pair<uint32_t, std::string>(lineInCurrentFileWithInclude, IShaderCompiler::escapeFilename(name.string())));
//
//    return res_str;
//}

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


typedef boost::wave::cpplexer::lex_token<> lex_token_t;
typedef boost::wave::cpplexer::lex_iterator<lex_token_t> lex_iterator_t;
typedef boost::wave::context<core::string::iterator, lex_iterator_t, nbl::asset::hlsl::impl::load_file_or_builtin_to_string, nbl::asset::hlsl::impl::custom_preprocessing_hooks> wave_context_t;

std::string CHLSLCompiler::preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions) const
{


    code.insert(0u, "#line 1\n");

    uint32_t defineLeadingLinesMain = 1;
    uint32_t leadingLinesImports = IShaderCompiler::encloseWithinExtraInclGuardsLeadingLines(preprocessOptions.maxSelfInclusionCount + 1u);
    if (preprocessOptions.extraDefines.size())
    {
        insertExtraDefines(code, preprocessOptions.extraDefines);
        defineLeadingLinesMain += preprocessOptions.extraDefines.size();
    }

    //IShaderCompiler::disableAllDirectivesExceptIncludes(code);
    // TODO: change `code` to `const core::string&` because its supposed to be immutable
    auto hooks = nbl::asset::hlsl::impl::custom_preprocessing_hooks(preprocessOptions, { std::pair<uint32_t, std::string>(defineLeadingLinesMain, preprocessOptions.sourceIdentifier) });
    wave_context_t context(code.begin(), code.end(), preprocessOptions.sourceIdentifier.data(), hooks);
    //      context.add_include_path
    //      context.add_sysinclude_path(preprocessOptions.includeFinder)/* <- for dem builtins! / preprocessOptions.includeFinder?*/ 
    //      context.add_macro_definition from preprocessOptions.extraDefines
    core::string resolvedString;
    
    {
        std::stringstream stream = std::stringstream();
        for (auto i = context.begin(); i != context.end(); i++)
        {
            stream << i->get_value();
        }
        resolvedString = stream.str();
    }
 
    
    IShaderCompiler::reenableDirectives(resolvedString);

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




// for including builtins 
template<>
inline bool boost::wave::impl::pp_iterator_functor<wave_context_t>::on_include_helper(char const* f, char const* s,
    bool is_system, bool include_next)
{
    namespace fs = boost::filesystem;

    // try to locate the given file, searching through the include path lists
    std::string file_path(s);
    std::string dir_path;
#if BOOST_WAVE_SUPPORT_INCLUDE_NEXT != 0
    char const* current_name = include_next ? iter_ctx->real_filename.c_str() : 0;
#else
    char const* current_name = 0; // never try to match current file name
#endif

    // call the 'found_include_directive' hook function
    if (ctx.get_hooks().found_include_directive(ctx.derived(), f, include_next))
        return true;    // client returned false: skip file to include

    file_path = util::impl::unescape_lit(file_path);
    std::string native_path_str;

    if (!ctx.get_hooks().locate_include_file(ctx, file_path, is_system,
        current_name, dir_path, native_path_str))
    {
        BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_include_file,
            file_path.c_str(), act_pos);
        return false;
    }

    // test, if this file is known through a #pragma once directive
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
    if (!ctx.has_pragma_once(native_path_str))
#endif
    {
        boost::shared_ptr<base_iteration_context_type> new_iter_ctx;
        auto builtin_pair = nbl::builtin::get_resource_runtime(file_path);
        if (builtin_pair.first) //not a nullptr
        {

            // preprocess the opened file
             new_iter_ctx = boost::shared_ptr<base_iteration_context_type>(
                new iteration_context_type(ctx, file_path.c_str(), act_pos,
                    boost::wave::enable_prefer_pp_numbers(ctx.get_language()),
                    is_system ? base_iteration_context_type::system_header :
                    base_iteration_context_type::user_header));

            // do not
            // call the include policy trace function
            //ctx.get_hooks().opened_include_file(ctx.derived(), dir_path, file_path, is_system);
        }
        else {
            // the new include file determines the actual current directory
            ctx.set_current_directory(native_path_str.c_str());
        
            // preprocess the opened file
            new_iter_ctx = boost::shared_ptr<base_iteration_context_type>(
                new iteration_context_type(ctx, native_path_str.c_str(), act_pos,
                    boost::wave::enable_prefer_pp_numbers(ctx.get_language()),
                    is_system ? base_iteration_context_type::system_header :
                    base_iteration_context_type::user_header));
        }

        


        // call the include policy trace function
        ctx.get_hooks().opened_include_file(ctx.derived(), dir_path, file_path,
            is_system);

        // store current file position
        iter_ctx->real_relative_filename = ctx.get_current_relative_filename().c_str();
        iter_ctx->filename = act_pos.get_file();
        iter_ctx->line = act_pos.get_line();
        iter_ctx->if_block_depth = ctx.get_if_block_depth();
        iter_ctx->emitted_lines = (unsigned int)(-1);   // force #line directive

        // push the old iteration context onto the stack and continue with the new
        ctx.push_iteration_context(act_pos, iter_ctx);
        iter_ctx = new_iter_ctx;
        seen_newline = true;        // fake a newline to trigger pp_directive
        must_emit_line_directive = true;

        act_pos.set_file(iter_ctx->filename);  // initialize file position
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
        fs::path rfp(wave::util::create_path(iter_ctx->real_filename.c_str()));
        std::string real_filename(rfp.string());
        ctx.set_current_filename(real_filename.c_str());
#endif

        ctx.set_current_relative_filename(dir_path.c_str());
        iter_ctx->real_relative_filename = dir_path.c_str();

        act_pos.set_line(iter_ctx->line);
        act_pos.set_column(0);
    }
    return true;
}


#endif
