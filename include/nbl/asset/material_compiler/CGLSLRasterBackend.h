// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_C_GLSL_RASTER_BACKEND_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_C_GLSL_RASTER_BACKEND_H_INCLUDED_

#include <nbl/asset/material_compiler/CGLSLBackendCommon.h>

namespace nbl::asset::material_compiler
{

class CGLSLRasterBackend : public CGLSLBackendCommon
{
        using base_t = CGLSLBackendCommon;

    public:
        result_t compile(SContext* _ctx, IR* _ir, E_GENERATOR_STREAM_TYPE _generatorChoiceStream=EGST_PRESENT) override;
};

}

#endif