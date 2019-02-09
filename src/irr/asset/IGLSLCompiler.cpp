#include "irr/asset/IGLSLCompiler.h"
#include "coreutil.h"
#include "irr/asset/ICPUShader.h"
#include "libshaderc/shaderc/shaderc.hpp"

namespace irr { namespace asset
{

namespace
{
    shaderc_shader_kind ESStoShadercEnum(E_SHADER_STAGE _ss)
    {
        using T = std::underlying_type_t<E_SHADER_STAGE>;

        shaderc_shader_kind convert[6];
        convert[core::numberOfSetBit<T>(ESS_VERTEX)] = shaderc_vertex_shader;
        convert[core::numberOfSetBit<T>(ESS_TESSELATION_CONTROL)] = shaderc_tess_control_shader;
        convert[core::numberOfSetBit<T>(ESS_TESSELATION_EVALUATION)] = shaderc_tess_evaluation_shader;
        convert[core::numberOfSetBit<T>(ESS_GEOMETRY)] = shaderc_geometry_shader;
        convert[core::numberOfSetBit<T>(ESS_FRAGMENT)] = shaderc_fragment_shader;
        convert[core::numberOfSetBit<T>(ESS_COMPUTE)] = shaderc_compute_shader;

        return convert[_ss];
    }
}

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