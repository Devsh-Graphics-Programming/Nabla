// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/asset/material_compiler/CMaterialCompilerGLSLRasterBackend.h>

namespace nbl
{
namespace asset
{
namespace material_compiler
{

auto CMaterialCompilerGLSLRasterBackend::compile(SContext* _ctx, IR* _ir) -> result_t
{
    constexpr bool WITH_GENERATOR_CHOICE = true;
    result_t res = base_t::compile(_ctx, _ir, WITH_GENERATOR_CHOICE);

    res.fragmentShaderSource = 
    R"(

#include <nbl/builtin/glsl/material_compiler/rasterization/impl.glsl>
    )";

    return res;
}

}}}