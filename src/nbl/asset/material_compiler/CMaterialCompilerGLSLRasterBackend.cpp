// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/asset/material_compiler/CMaterialCompilerGLSLRasterBackend.h>

namespace nbl::asset::material_compiler
{
auto CMaterialCompilerGLSLRasterBackend::compile(SContext* _ctx, IR* _ir, E_GENERATOR_STREAM_TYPE _generatorChoiceStream) -> result_t
{
    result_t res = base_t::compile(_ctx, _ir, _generatorChoiceStream);

    res.fragmentShaderSource =
        R"(
#include <nbl/builtin/glsl/material_compiler/rasterization/impl.glsl>
    )";

    return res;
}

}