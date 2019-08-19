#include "irr/asset/IGLSLCompiler.h"
#include "coreutil.h"
#include "irr/asset/ICPUShader.h"
#include "irr/asset/shadercUtils.h"

namespace irr { namespace asset
{

ICPUShader* IGLSLCompiler::createShaderFromGLSL(const char* _glslCode, E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, std::string* _outAssembly) const
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
    return new ICPUShader(bin_res.cbegin(), std::distance(bin_res.cbegin(), bin_res.cend())*sizeof(uint32_t));
}

}}