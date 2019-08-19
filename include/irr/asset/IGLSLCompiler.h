#ifndef __IRR_I_GLSL_COMPILER_H_INCLUDED__
#define __IRR_I_GLSL_COMPILER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/ShaderCommons.h"

namespace irr { namespace asset
{
    class IIncludeHandler;
//! Will be derivative of IShaderGenerator, but we have to establish interface first
class IGLSLCompiler : public core::IReferenceCounted
{
    const IIncludeHandler* m_inclHandler;

public:
    IGLSLCompiler(const IIncludeHandler* _inclhndlr) : m_inclHandler(_inclhndlr) {}

    /**
    If _stage is ESS_UNKNOWN, then compiler will try to deduce shader stage from #pragma annotation, i.e.:
    #pragma shader_stage(vertex),       or
    #pragma shader_stage(tesscontrol),  or
    #pragma shader_stage(tesseval),     or
    #pragma shader_stage(geometry),     or
    #pragma shader_stage(fragment),     or
    #pragma shader_stage(compute)

    Such annotation should be placed right after #version directive.

    This function does NOT process #include directives! Use resolveIncludeDirectives() first.

    @param _outAssembly Optional parameter; if not nullptr, SPIR-V assembly is saved in there.
    @param _compilationId String that will be printed along with possible errors as source identifier.

    @returns SPIR-V bytecode.
    */
    ICPUBuffer* createSPIRVFromGLSL(const char* _glslCode, E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, std::string* _outAssembly = nullptr) const;

    /**
    @param _originFilepath Path to not necesarilly existing file whose directory will be base for relative (""-type) top-level #include's resolution.
        If _originFilepath is non-path-like string, the base directory is assumed to be "." (working directory of your executable).
    */
    std::string resolveIncludeDirectives(const char* _glslCode, E_SHADER_STAGE _stage, const char* _originFilepath) const;
};

}}

#endif//__IRR_I_GLSL_COMPILER_H_INCLUDED__
