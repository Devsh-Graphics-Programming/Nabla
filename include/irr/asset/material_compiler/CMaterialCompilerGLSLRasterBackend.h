#ifndef __C_MITSUBA_MATERIAL_COMPILER_GLSL_RASTER_BACKEND_H_INCLUDED__
#define __C_MITSUBA_MATERIAL_COMPILER_GLSL_RASTER_BACKEND_H_INCLUDED__

#include <irr/asset/material_compiler/CMaterialCompilerGLSLBackendCommon.h>

namespace irr
{
namespace asset
{
namespace material_compiler
{

class CMaterialCompilerGLSLRasterBackend : public CMaterialCompilerGLSLBackendCommon
{
    using base_t = CMaterialCompilerGLSLBackendCommon;

public:
    result_t compile(SContext* _ctx, IR* _ir);
};

}}}

#endif