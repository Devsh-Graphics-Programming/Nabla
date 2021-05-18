// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_MITSUBA_MATERIAL_COMPILER_GLSL_RASTER_BACKEND_H_INCLUDED__
#define __NBL_ASSET_C_MITSUBA_MATERIAL_COMPILER_GLSL_RASTER_BACKEND_H_INCLUDED__

#include <nbl/asset/material_compiler/CMaterialCompilerGLSLBackendCommon.h>

namespace nbl::asset::material_compiler
{

class CMaterialCompilerGLSLRasterBackend : public CMaterialCompilerGLSLBackendCommon
{
        using base_t = CMaterialCompilerGLSLBackendCommon;

    public:
        result_t compile(SContext* _ctx, IR* _ir);
};

}

#endif