#include <irr\asset\material_compiler\CMaterialCompilerGLSLRasterBackend.h>

namespace irr
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

#include <irr/builtin/material_compiler/glsl/rasterization/impl.glsl>
    )";

    return res;
}

}}}