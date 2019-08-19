#include "irr/asset/IGLSLCompiler.h"
#include "coreutil.h"
#include "irr/asset/shadercUtils.h"
#include "IFileSystem.h"
#include <sstream>
#include <regex>
#include <iterator>

namespace irr { namespace asset
{

ICPUBuffer* IGLSLCompiler::createSPIRVFromGLSL(const char* _glslCode, E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, std::string* _outAssembly) const
{
    shaderc::Compiler comp;
    shaderc::CompileOptions options;//default options
    const shaderc_shader_kind stage = _stage==ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(_stage);
    const size_t glsl_len = strlen(_glslCode);

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
    asset::ICPUBuffer* spirv = new ICPUBuffer(std::distance(bin_res.cbegin(), bin_res.cend())*sizeof(uint32_t));
    memcpy(spirv->getPointer(), bin_res.cbegin(), spirv->getSize());
    return spirv;
}

namespace impl
{
    //string to be replaced with all "#" except those in "#include"
    static constexpr const char* PREPROC_DIRECTIVE_DISABLER = "_this_is_hash_";
    static std::string disableAllDirectivesExceptIncludes(const char* _glslCode)
    {
        std::regex re("#(?!(include|version|pragma shader_stage))");//all # not followed by "include" nor "version" nor "pragma shader_stage"
        //`#pragma shader_stage(...)` is needed for determining shader stage when `_stage` param of IGLSLCompiler functions is set to ESS_UNKNOWN
        std::stringstream ss;
        std::regex_replace(std::ostreambuf_iterator<char>(ss), _glslCode, _glslCode + strlen(_glslCode), re, PREPROC_DIRECTIVE_DISABLER);
        return ss.str();
    }
    static std::string reenableDirectives(const char* _glslCode)
    {
        std::regex re(PREPROC_DIRECTIVE_DISABLER);
        std::stringstream ss;
        std::regex_replace(std::ostreambuf_iterator<char>(ss), _glslCode, _glslCode + strlen(_glslCode), re, "#");
        return ss.str();
    }

    class Includer : public shaderc::CompileOptions::IncluderInterface
    {
        const asset::IIncludeHandler* m_inclHandler;

    public:
        Includer(const asset::IIncludeHandler* _inclhndlr) : m_inclHandler(_inclhndlr) {}

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
            io::path relDir;
            if (_type == shaderc_include_type_relative) {
                relDir = io::IFileSystem::getFileDir(_requesting_source);
                res_str = m_inclHandler->getIncludeRelative(_requested_source, relDir.c_str());
            }
            else { //shaderc_include_type_standard
                res_str = m_inclHandler->getIncludeStandard(_requested_source);
            }

            if (!res_str.size()) {
                res->content_length = 0u;
                res->content = ""; //error message should be placed here
                res->source_name_length = 0u;
                res->source_name = "";
            }
            else {
                res_str = disableAllDirectivesExceptIncludes(res_str.c_str());

                res->content_length = res_str.size();
                res->content = new char[res_str.size()+1u];
                strcpy(const_cast<char*>(res->content), res_str.c_str());
                io::path name = (_type==shaderc_include_type_relative) ? (relDir + _requested_source) : (_requested_source);
                res->source_name_length = name.size();
                res->source_name = new char[name.size()+1u];
                strcpy(const_cast<char*>(res->source_name), name.c_str());
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

std::string IGLSLCompiler::resolveIncludeDirectives(const char* _glslCode, E_SHADER_STAGE _stage, const char* _originFilepath) const
{
    std::string glslCode = impl::disableAllDirectivesExceptIncludes(_glslCode);//all "#", except those in "#include"/"#version"/"#pragma shader_stage(...)", replaced with `PREPROC_DIRECTIVE_DISABLER`
    shaderc::Compiler comp;
    shaderc::CompileOptions options;//default options
    options.SetIncluder(std::make_unique<impl::Includer>(m_inclHandler));//custom #include handler
    const shaderc_shader_kind stage = (_stage == ESS_UNKNOWN) ? shaderc_glsl_infer_from_source : ESStoShadercEnum(_stage);
    auto res = comp.PreprocessGlsl(glslCode, stage, _originFilepath, options);

    if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
        os::Printer::log(res.GetErrorMessage(), ELL_ERROR);
        return "";
    }

    std::string res_str(res.cbegin(), std::distance(res.cbegin(),res.cend()));
    return impl::reenableDirectives(res_str.c_str());
}

}}