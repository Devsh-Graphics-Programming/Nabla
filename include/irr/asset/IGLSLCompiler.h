#ifndef __IRR_I_GLSL_COMPILER_H_INCLUDED__
#define __IRR_I_GLSL_COMPILER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/ShaderCommons.h"

namespace irr { namespace asset
{
class ICPUShader;

//! Will be derivative of IShaderGenerator, but we have to establish interface first
class IGLSLCompiler : public core::IReferenceCounted
{
public:
    /**
    If _stage is ESS_UNKNOWN, then compiler will try to deduce shader stage from #pragma annotation, i.e.:
    #pragma shader_stage(vertex),       or
    #pragma shader_stage(tesscontrol),  or
    #pragma shader_stage(tesseval),     or
    #pragma shader_stage(geometry),     or
    #pragma shader_stage(fragment),     or
    #pragma shader_stage(compute)

    Such annotation should be placed right after #version directive.
    */
    ICPUShader* createShaderFromGLSL(const char* _glslCode, E_SHADER_STAGE _stage, const char* _entryPoint, bool _debug = false, const char* compilationId = nullptr) const;
};

}}

#endif//__IRR_I_GLSL_COMPILER_H_INCLUDED__
