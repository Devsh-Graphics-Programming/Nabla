// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/CGLSLCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"
#ifdef _NBL_EMBED_BUILTIN_RESOURCES_
#include "nbl/builtin/CArchive.h"
#endif // _NBL_EMBED_BUILTIN_RESOURCES_

#include <sstream>
#include <regex>
#include <iterator>

using namespace nbl;
using namespace nbl::asset;

static constexpr const char* PREPROC_GL__DISABLER = "_this_is_a_GL__prefix_";
static constexpr const char* PREPROC_GL__ENABLER = PREPROC_GL__DISABLER;
static constexpr const char* PREPROC_LINE_CONTINUATION_DISABLER = "_this_is_a_line_continuation_\n";
static constexpr const char* PREPROC_LINE_CONTINUATION_ENABLER = "_this_is_a_line_continuation_";

static void disableGlDirectives(std::string& _code)
{
    std::regex glMacro("[ \t\r\n\v\f]GL_");
    auto result = std::regex_replace(_code, glMacro, PREPROC_GL__DISABLER);
    std::regex lineContinuation("\\\\[ \t\r\n\v\f]*\n");
    _code = std::regex_replace(result, lineContinuation, PREPROC_LINE_CONTINUATION_DISABLER);
}

static void reenableGlDirectives(std::string& _code)
{
    std::regex lineContinuation(PREPROC_LINE_CONTINUATION_ENABLER);
    auto result = std::regex_replace(_code, lineContinuation, " \\");
    std::regex glMacro(PREPROC_GL__ENABLER);
    _code = std::regex_replace(result, glMacro, " GL_");
}


namespace nbl::asset::impl
{
    class Includer : public shaderc::CompileOptions::IncluderInterface
    {
        const IShaderCompiler::CIncludeFinder* m_defaultIncludeFinder;
        const system::ISystem* m_system;
        const uint32_t m_maxInclCnt;

    public:
        Includer(const IShaderCompiler::CIncludeFinder* _inclFinder, const system::ISystem* _fs, uint32_t _maxInclCnt) : m_defaultIncludeFinder(_inclFinder), m_system(_fs), m_maxInclCnt{ _maxInclCnt } {}

        //_requesting_source in top level #include's is what shaderc::Compiler's compiling functions get as `input_file_name` parameter
        //so in order for properly working relative #include's (""-type) `input_file_name` has to be path to file from which the GLSL source really come from
        //or at least path to not necessarily existing file whose directory will be base for ""-type #include's resolution
        shaderc_include_result* GetInclude(const char* _requested_source,
            shaderc_include_type _type,
            const char* _requesting_source,
            size_t _include_depth) override
        {
            shaderc_include_result* res = new shaderc_include_result;
            std::string res_str;

            std::filesystem::path relDir;
            #ifdef _NBL_EMBED_BUILTIN_RESOURCES_
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
            #else
            const bool reqBuiltin = false;
            #endif // _NBL_EMBED_BUILTIN_RESOURCES_
            std::filesystem::path name = (_type == shaderc_include_type_relative) ? (relDir / _requested_source) : (_requested_source);

            if (std::filesystem::exists(name) && !reqBuiltin)
                name = std::filesystem::absolute(name);

            if (_type == shaderc_include_type_relative)
                res_str = m_defaultIncludeFinder->getIncludeRelative(relDir, _requested_source);
            else //shaderc_include_type_standard
                res_str = m_defaultIncludeFinder->getIncludeStandard(relDir, _requested_source);

            if (!res_str.size()) {
                const char* error_str = "Could not open file";
                res->content_length = strlen(error_str);
                res->content = new char[res->content_length + 1u];
                strcpy(const_cast<char*>(res->content), error_str);
                res->source_name_length = 0u;
                res->source_name = "";
            }
            else {
                //employ encloseWithinExtraInclGuards() in order to prevent infinite loop of (not necesarilly direct) self-inclusions while other # directives (incl guards among them) are disabled
                IShaderCompiler::disableAllDirectivesExceptIncludes(res_str);
                disableGlDirectives(res_str);
                res_str = IShaderCompiler::encloseWithinExtraInclGuards(std::move(res_str), m_maxInclCnt, name.string().c_str());

                res->content_length = res_str.size();
                res->content = new char[res_str.size() + 1u];
                strcpy(const_cast<char*>(res->content), res_str.c_str());
                res->source_name_length = name.native().size();
                res->source_name = new char[name.native().size() + 1u];
                strcpy(const_cast<char*>(res->source_name), name.string().c_str());
            }

            return res;
        }

        void ReleaseInclude(shaderc_include_result* data) override
        {
            if (data->content_length > 0u)
                delete[] data->content;
            if (data->source_name_length > 0u)
                delete[] data->source_name;
            delete data;
        }
    };
}

CGLSLCompiler::CGLSLCompiler(core::smart_refctd_ptr<system::ISystem>&& system)
    : IShaderCompiler(std::move(system))
{
}



std::string CGLSLCompiler::preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions) const
{
    if (preprocessOptions.extraDefines.size())
    {
        insertExtraDefines(code, preprocessOptions.extraDefines);
    }
    IShaderCompiler::disableAllDirectivesExceptIncludes(code);
    disableGlDirectives(code);
    shaderc::Compiler comp;
    shaderc::CompileOptions options;
    options.SetTargetSpirv(shaderc_spirv_version_1_6);

    if (preprocessOptions.includeFinder != nullptr)
    {
        options.SetIncluder(std::make_unique<impl::Includer>(preprocessOptions.includeFinder, m_system.get(), preprocessOptions.maxSelfInclusionCount + 1u));//custom #include handler
    }
    const shaderc_shader_kind scstage = stage == IShader::ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(stage);
    auto res = comp.PreprocessGlsl(code, scstage, preprocessOptions.sourceIdentifier.data(), options);

    if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
        preprocessOptions.logger.log(res.GetErrorMessage(), system::ILogger::ELL_ERROR);
        return nullptr;
    }

    auto resolvedString = std::string(res.cbegin(), std::distance(res.cbegin(), res.cend()));
    IShaderCompiler::reenableDirectives(resolvedString);
    reenableGlDirectives(resolvedString);
    return resolvedString;
}

core::smart_refctd_ptr<ICPUShader> CGLSLCompiler::compileToSPIRV(const char* code, const IShaderCompiler::SCompilerOptions& options) const
{
    auto glslOptions = option_cast(options);

    if (!code)
    {
        glslOptions.preprocessorOptions.logger.log("code is nullptr", system::ILogger::ELL_ERROR);
        return nullptr;
    }

    auto newCode = preprocessShader(std::string(code), glslOptions.stage, glslOptions.preprocessorOptions);

    shaderc::Compiler comp;
    shaderc::CompileOptions shadercOptions; //default options
    shadercOptions.SetTargetSpirv(static_cast<shaderc_spirv_version>(glslOptions.targetSpirvVersion));
    const shaderc_shader_kind stage = glslOptions.stage == IShader::ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(glslOptions.stage);
    if (glslOptions.genDebugInfo)
        shadercOptions.SetGenerateDebugInfo();

    shaderc::SpvCompilationResult bin_res = comp.CompileGlslToSpv(newCode.c_str(), newCode.size(), stage, glslOptions.preprocessorOptions.sourceIdentifier.data() ? glslOptions.preprocessorOptions.sourceIdentifier.data() : "", "main", shadercOptions);

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
