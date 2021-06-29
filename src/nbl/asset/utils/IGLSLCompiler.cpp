// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <sstream>
#include <regex>
#include <iterator>

#include "nbl/asset/utils/IGLSLCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"
#include "nbl/asset/utils/CIncludeHandler.h"

#include "nbl/asset/utils/CGLSLVirtualTexturingBuiltinIncludeLoader.h"

#include "os.h"
namespace nbl
{
using namespace system;
namespace asset
{

static constexpr shaderc_spirv_version TARGET_SPIRV_VERSION = shaderc_spirv_version_1_5;

IGLSLCompiler::IGLSLCompiler(system::ISystem* _s) : m_inclHandler(core::make_smart_refctd_ptr<CIncludeHandler>(_s)), m_system(_s)
{
    m_inclHandler->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<asset::CGLSLVirtualTexturingBuiltinIncludeLoader>(_s));
}

core::smart_refctd_ptr<ICPUBuffer> IGLSLCompiler::compileSPIRVFromGLSL(const char* _glslCode, ISpecializedShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, bool _genDebugInfo, std::string* _outAssembly) const
{
    //shaderc requires entry point to be "main" in GLSL
    if (strcmp(_entryPoint, "main") != 0)
        return nullptr;

    shaderc::Compiler comp;
    shaderc::CompileOptions options;//default options
    options.SetTargetSpirv(TARGET_SPIRV_VERSION);
    const shaderc_shader_kind stage = _stage==ISpecializedShader::ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(_stage);
    const size_t glsl_len = strlen(_glslCode);
    if (_genDebugInfo)
        options.SetGenerateDebugInfo();

    shaderc::AssemblyCompilationResult asm_res;
    shaderc::SpvCompilationResult bin_res;
    if (_outAssembly) {
        asm_res = comp.CompileGlslToSpvAssembly(_glslCode, glsl_len, stage, _compilationId ? _compilationId : "", options);
        _outAssembly->resize(std::distance(asm_res.cbegin(), asm_res.cend()));
        memcpy(_outAssembly->data(), asm_res.cbegin(), _outAssembly->size());
        bin_res = comp.AssembleToSpv(_outAssembly->data(), _outAssembly->size(), options);
    }
    else {
        bin_res = comp.CompileGlslToSpv(_glslCode, glsl_len, stage, _compilationId ? _compilationId : "", _entryPoint, options);
    }

    if (bin_res.GetCompilationStatus() != shaderc_compilation_status_success) {
        os::Printer::log(bin_res.GetErrorMessage(), ELL_ERROR);
        return nullptr;
    }

    auto spirv = core::make_smart_refctd_ptr<ICPUBuffer>(std::distance(bin_res.cbegin(), bin_res.cend())*sizeof(uint32_t));
    memcpy(spirv->getPointer(), bin_res.cbegin(), spirv->getSize());
	return spirv;
}

core::smart_refctd_ptr<ICPUShader> IGLSLCompiler::createSPIRVFromGLSL(const char* _glslCode, ISpecializedShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, const ISPIRVOptimizer* _opt, bool _genDebugInfo, std::string* _outAssembly) const
{
    auto spirvBuffer = compileSPIRVFromGLSL(_glslCode,_stage,_entryPoint,_compilationId,_genDebugInfo,_outAssembly);
	if (!spirvBuffer)
		return nullptr;
    if (_opt)
        spirvBuffer = _opt->optimize(spirvBuffer.get());

    return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(spirvBuffer));
}

core::smart_refctd_ptr<ICPUShader> IGLSLCompiler::createSPIRVFromGLSL(system::IFile* _sourcefile, ISpecializedShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, const ISPIRVOptimizer* _opt, bool _genDebugInfo, std::string* _outAssembly) const
{
    size_t fileSize = _sourcefile->getSize();
    std::string glsl(fileSize, '\0');
    system::ISystem::future_t<uint32_t> future;
    m_system->readFile(future, _sourcefile, glsl.data(), 0, fileSize);
    future.get();
    return createSPIRVFromGLSL(glsl.c_str(), _stage, _entryPoint, _compilationId, _opt, _outAssembly);
}

namespace impl
{
    //string to be replaced with all "#" except those in "#include"
    static constexpr const char* PREPROC_DIRECTIVE_DISABLER = "_this_is_a_hash_";
    static constexpr const char* PREPROC_DIRECTIVE_ENABLER = PREPROC_DIRECTIVE_DISABLER;
    static constexpr const char* PREPROC_GL__DISABLER = "_this_is_a_GL__prefix_";
    static constexpr const char* PREPROC_GL__ENABLER = PREPROC_GL__DISABLER;
    static constexpr const char* PREPROC_LINE_CONTINUATION_DISABLER = "_this_is_a_line_continuation_\n";
    static constexpr const char* PREPROC_LINE_CONTINUATION_ENABLER = "_this_is_a_line_continuation_";
    static void disableAllDirectivesExceptIncludes(std::string& _glslCode)
    {
        // TODO: replace this with a proper-ish proprocessor and includer one day
        std::regex directive("#(?!(include|version|pragma shader_stage|line))");//all # not followed by "include" nor "version" nor "pragma shader_stage"
        //`#pragma shader_stage(...)` is needed for determining shader stage when `_stage` param of IGLSLCompiler functions is set to ESS_UNKNOWN
        auto result = std::regex_replace(_glslCode,directive,PREPROC_DIRECTIVE_DISABLER);
        std::regex glMacro("[ \t\r\n\v\f]GL_");
        result = std::regex_replace(result, glMacro, PREPROC_GL__DISABLER);
        std::regex lineContinuation("\\\\[ \t\r\n\v\f]*\n");
        _glslCode = std::regex_replace(result, lineContinuation, PREPROC_LINE_CONTINUATION_DISABLER);
    }
    static void reenableDirectives(std::string& _glslCode)
    {
        std::regex lineContinuation(PREPROC_LINE_CONTINUATION_ENABLER);
        auto result = std::regex_replace(_glslCode, lineContinuation, " \\");
        std::regex glMacro(PREPROC_GL__ENABLER);
        result = std::regex_replace(result,glMacro," GL_");
        std::regex directive(PREPROC_DIRECTIVE_ENABLER);
        _glslCode = std::regex_replace(result, directive, "#");
    }
    static std::string encloseWithinExtraInclGuards(std::string&& _glslCode, uint32_t _maxInclusions, const char* _identifier)
    {
        assert(_maxInclusions!=0u);

        using namespace std::string_literals;
        std::string defBase_ = "_GENERATED_INCLUDE_GUARD_"s + _identifier + "_";
        std::replace_if(defBase_.begin(), defBase_.end(), [](char c) ->bool { return !::isalpha(c) && !::isdigit(c); }, '_');

        auto genDefs = [&defBase_, _maxInclusions, _identifier] {
            auto defBase = [&defBase_](uint32_t n) { return defBase_ + std::to_string(n); };
            std::string defs = "#ifndef " + defBase(0) + "\n\t#define " + defBase(0) + "\n";
            for (uint32_t i = 1u; i <= _maxInclusions; ++i) {
                const std::string defname = defBase(i);
                defs += "#elif !defined(" + defname + ")\n\t#define " + defname + "\n";
            }
            defs += "#endif\n";
            return defs;
        };
        auto genUndefs = [&defBase_, _maxInclusions, _identifier] {
            auto defBase = [&defBase_](int32_t n) { return defBase_ + std::to_string(n); };
            std::string undefs = "#ifdef " + defBase(_maxInclusions) + "\n\t#undef " + defBase(_maxInclusions) + "\n";
            for (int32_t i = _maxInclusions-1; i >= 0; --i) {
                const std::string defname = defBase(i);
                undefs += "#elif defined(" + defname + ")\n\t#undef " + defname + "\n";
            }
            undefs += "#endif\n";
            return undefs;
        };
        
        return
            genDefs() +
            "\n"
            "#ifndef " + defBase_ + std::to_string(_maxInclusions) +
            "\n" +
            "#line 1 \"" + _identifier + "\"\n" +
            _glslCode +
            "\n"
            "#endif"
            "\n\n" +
            genUndefs();
    }

    class Includer : public shaderc::CompileOptions::IncluderInterface
    {
        const asset::IIncludeHandler* m_inclHandler;
        const system::ISystem* m_system;
        const uint32_t m_maxInclCnt;

    public:
        Includer(const asset::IIncludeHandler* _inclhndlr, const system::ISystem* _fs, uint32_t _maxInclCnt) : m_inclHandler(_inclhndlr), m_system(_fs), m_maxInclCnt{_maxInclCnt} {}

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
            const bool reqFromBuiltin = asset::IIncludeHandler::isBuiltinPath(_requesting_source);
            const bool reqBuiltin = asset::IIncludeHandler::isBuiltinPath(_requested_source);
            if (!reqFromBuiltin && !reqBuiltin)
            {
                //While #includ'ing a builtin, one must specify its full path (starting with "nbl/builtin" or "/nbl/builtin").
                //  This rule applies also while a builtin is #includ`ing another builtin.
                //While including a filesystem file it must be either absolute path (or relative to any search dir added to asset::iIncludeHandler; <>-type),
                //  or path relative to executable's working directory (""-type).
                relDir = std::filesystem::path(_requesting_source).parent_path();
            }

            std::filesystem::path name = (_type == shaderc_include_type_relative) ? (relDir.append(_requested_source)) : (_requested_source);
            if (!reqBuiltin)
                name = std::filesystem::absolute(name);

            if (_type == shaderc_include_type_relative)
                res_str = m_inclHandler->getIncludeRelative(_requested_source, relDir.string());
            else //shaderc_include_type_standard
                res_str = m_inclHandler->getIncludeStandard(_requested_source);

            if (!res_str.size()) {
                const char* error_str = "Could not open file";
                res->content_length = strlen(error_str);
                res->content = new char[res->content_length+1u];
                strcpy(const_cast<char*>(res->content), error_str);
                res->source_name_length = 0u;
                res->source_name = "";
            }
            else {
                //employ encloseWithinExtraInclGuards() in order to prevent infinite loop of (not necesarilly direct) self-inclusions while other # directives (incl guards among them) are disabled
                disableAllDirectivesExceptIncludes(res_str);
                res_str = encloseWithinExtraInclGuards( std::move(res_str), m_maxInclCnt, name.string().c_str() );

                res->content_length = res_str.size();
                res->content = new char[res_str.size()+1u];
                strcpy(const_cast<char*>(res->content), res_str.c_str());
                res->source_name_length = name.native().size();
                res->source_name = new char[name.native().size()+1u];
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

core::smart_refctd_ptr<ICPUShader> IGLSLCompiler::resolveIncludeDirectives(std::string&& glslCode, ISpecializedShader::E_SHADER_STAGE _stage, const char* _originFilepath, uint32_t _maxSelfInclusionCnt) const
{
    impl::disableAllDirectivesExceptIncludes(glslCode);//all "#", except those in "#include"/"#version"/"#pragma shader_stage(...)", replaced with `PREPROC_DIRECTIVE_DISABLER`
    shaderc::Compiler comp;
    shaderc::CompileOptions options;
    options.SetTargetSpirv(TARGET_SPIRV_VERSION);
    options.SetIncluder(std::make_unique<impl::Includer>(m_inclHandler.get(), m_system, _maxSelfInclusionCnt+1u));//custom #include handler
    const shaderc_shader_kind stage = _stage==ISpecializedShader::ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(_stage);
    auto res = comp.PreprocessGlsl(glslCode, stage, _originFilepath, options);

    if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
        os::Printer::log(res.GetErrorMessage(), ELL_ERROR);
        return nullptr;
    }

    std::string res_str(res.cbegin(), std::distance(res.cbegin(),res.cend()));
    impl::reenableDirectives(res_str);

    return core::make_smart_refctd_ptr<ICPUShader>(res_str.c_str());
}

core::smart_refctd_ptr<ICPUShader> IGLSLCompiler::resolveIncludeDirectives(system::IFile* _sourcefile, ISpecializedShader::E_SHADER_STAGE _stage, const char* _originFilepath, uint32_t _maxSelfInclusionCnt) const
{
    std::string glsl(_sourcefile->getSize(), '\0');
    system::ISystem::future_t<uint32_t> future;
    m_system->readFile(future, _sourcefile, glsl.data(), 0, _sourcefile->getSize());
    future.get();
    return resolveIncludeDirectives(std::move(glsl), _stage, _originFilepath, _maxSelfInclusionCnt);
}

}}
