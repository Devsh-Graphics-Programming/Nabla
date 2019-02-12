#include "irr/asset/IGLSLCompiler.h"
#include "coreutil.h"
#include "irr/asset/ICPUShader.h"
#include "irr/asset/shadercUtils.h"

namespace irr { namespace asset
{

ICPUShader* IGLSLCompiler::createShaderFromGLSL(const char* _glslCode, E_SHADER_STAGE _stage, const char* _entryPoint, bool _debug, const char* _compilationId) const
{
    shaderc::Compiler comp;
    shaderc::CompileOptions options;
    if (_debug)
        options.SetGenerateDebugInfo();
    const shaderc_shader_kind stage = _stage==ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(_stage);
    shaderc::SpvCompilationResult res = comp.CompileGlslToSpv(_glslCode, strlen(_glslCode), stage, _compilationId ? _compilationId : "", _entryPoint, options);
    return new ICPUShader(res.cbegin(), std::distance(res.cbegin(), res.cend())*sizeof(uint32_t));
}

}}