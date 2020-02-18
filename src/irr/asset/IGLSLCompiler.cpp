#include "irr/core/core.h"

#include "irr/asset/IGLSLCompiler.h"
#include "irr/asset/ICPUShader.h"
#include "irr/asset/shadercUtils.h"
#include "IFileSystem.h"
#include "irr/asset/CIncludeHandler.h"
#include "irr/asset/CGLSLScanBuiltinIncludeLoader.h"
#include "irr/asset/CGLSLSkinningBuiltinIncludeLoader.h"
#include "irr/asset/CGLSLBRDFBuiltinIncludeLoader.h"
#include "irr/asset/CGLSLVertexUtilsBuiltinIncludeLoader.h"
#include "irr/asset/CGLSLBumpMappingBuiltinIncludeLoader.h"
#include "irr/asset/CGLSLBrokenDriverWorkaroundsBuiltinIncludeLoader.h"
#include "IReadFile.h"
#include "os.h"
#include <sstream>
#include <regex>
#include <iterator>

namespace irr
{
namespace asset
{

IGLSLCompiler::IGLSLCompiler(io::IFileSystem* _fs) : m_inclHandler(core::make_smart_refctd_ptr<CIncludeHandler>(_fs)), m_fs(_fs)
{
    m_inclHandler->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<asset::CGLSLScanBuiltinIncludeLoader>());
    m_inclHandler->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<asset::CGLSLSkinningBuiltinIncludeLoader>());
    m_inclHandler->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<asset::CGLSLBSDFBuiltinIncludeLoader>());
    m_inclHandler->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<asset::CGLSLVertexUtilsBuiltinIncludeLoader>());
    m_inclHandler->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<asset::CGLSLBumpMappingBuiltinIncludeLoader>());
    m_inclHandler->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<asset::CGLSLBrokenDriverWorkaroundsBuiltinIncludeLoader>());
	// TODO: Add BSDF includes here!
}

core::smart_refctd_ptr<ICPUBuffer> IGLSLCompiler::compileSPIRVFromGLSL(const char* _glslCode, ISpecializedShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, bool _genDebugInfo, std::string* _outAssembly) const
{
    //shaderc requires entry point to be "main" in GLSL
    if (strcmp(_entryPoint, "main") != 0)
        return nullptr;

    shaderc::Compiler comp;
    shaderc::CompileOptions options;//default options
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

core::smart_refctd_ptr<ICPUShader> IGLSLCompiler::createSPIRVFromGLSL(const char* _glslCode, ISpecializedShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, bool _genDebugInfo, std::string* _outAssembly) const
{
    return core::make_smart_refctd_ptr<asset::ICPUShader>(compileSPIRVFromGLSL(_glslCode,_stage,_entryPoint,_compilationId,_genDebugInfo,_outAssembly));
}

core::smart_refctd_ptr<ICPUShader> IGLSLCompiler::createSPIRVFromGLSL(io::IReadFile* _sourcefile, ISpecializedShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, bool _genDebugInfo, std::string* _outAssembly) const
{
    std::string glsl(_sourcefile->getSize(), '\0');
    _sourcefile->read(glsl.data(), glsl.size());
    return createSPIRVFromGLSL(glsl.c_str(), _stage, _entryPoint, _compilationId, _outAssembly);
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
            _glslCode +
            "\n"
            "#endif"
            "\n\n" +
            genUndefs();
    }

    class Includer : public shaderc::CompileOptions::IncluderInterface
    {
        const asset::IIncludeHandler* m_inclHandler;
        const io::IFileSystem* m_fs;
        const uint32_t m_maxInclCnt;

    public:
        Includer(const asset::IIncludeHandler* _inclhndlr, const io::IFileSystem* _fs, uint32_t _maxInclCnt) : m_inclHandler(_inclhndlr), m_fs(_fs), m_maxInclCnt{_maxInclCnt} {}

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
            const bool reqFromBuiltin = asset::IIncludeHandler::isBuiltinPath(_requesting_source);

            if (_type == shaderc_include_type_relative) {
                //While #includ'ing a builtin, one must specify its full path (starting with "irr/builtin" or "/irr/builtin").
                //  This rule applies also while a builtin is #includ`ing another builtin.
                //While including a filesystem file it must be either absolute path (or relative to any search dir added to asset::iIncludeHandler; <>-type),
                //  or path relative to executable's working directory (""-type).
                relDir = reqFromBuiltin ? "" : io::IFileSystem::getFileDir(_requesting_source);
                if (relDir.lastChar()!='/')
                    relDir.append('/');
                res_str = m_inclHandler->getIncludeRelative(_requested_source, relDir.c_str());
            }
            else { //shaderc_include_type_standard
                res_str = m_inclHandler->getIncludeStandard(_requested_source);
            }

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
                res_str = encloseWithinExtraInclGuards( disableAllDirectivesExceptIncludes(res_str.c_str()), m_maxInclCnt, _requested_source );

                res->content_length = res_str.size();
                res->content = new char[res_str.size()+1u];
                strcpy(const_cast<char*>(res->content), res_str.c_str());
                io::path name = (_type==shaderc_include_type_relative) ? (relDir + _requested_source) : (_requested_source);
                if (!asset::IIncludeHandler::isBuiltinPath(name.c_str()))
                    name = m_fs->getAbsolutePath(name);
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

core::smart_refctd_ptr<ICPUShader> IGLSLCompiler::resolveIncludeDirectives(const char* _glslCode, ISpecializedShader::E_SHADER_STAGE _stage, const char* _originFilepath, uint32_t _maxSelfInclusionCnt) const
{
    std::string glslCode = impl::disableAllDirectivesExceptIncludes(_glslCode);//all "#", except those in "#include"/"#version"/"#pragma shader_stage(...)", replaced with `PREPROC_DIRECTIVE_DISABLER`
    shaderc::Compiler comp;
    shaderc::CompileOptions options;//default options
    options.SetIncluder(std::make_unique<impl::Includer>(m_inclHandler.get(), m_fs, _maxSelfInclusionCnt+1u));//custom #include handler
    const shaderc_shader_kind stage = _stage==ISpecializedShader::ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(_stage);
    auto res = comp.PreprocessGlsl(glslCode, stage, _originFilepath, options);

    if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
        os::Printer::log(res.GetErrorMessage(), ELL_ERROR);
        return nullptr;
    }

    std::string res_str(res.cbegin(), std::distance(res.cbegin(),res.cend()));
    return core::make_smart_refctd_ptr<ICPUShader>(impl::reenableDirectives(res_str.c_str()).c_str());
}

core::smart_refctd_ptr<ICPUShader> IGLSLCompiler::resolveIncludeDirectives(io::IReadFile* _sourcefile, ISpecializedShader::E_SHADER_STAGE _stage, const char* _originFilepath, uint32_t _maxSelfInclusionCnt) const
{
    std::string glsl(_sourcefile->getSize(), '\0');
    _sourcefile->read(glsl.data(), glsl.size());
    return resolveIncludeDirectives(glsl.c_str(), _stage, _originFilepath, _maxSelfInclusionCnt);
}

}}