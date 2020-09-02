#include <irr\asset\material_compiler\CMaterialCompilerGLSLRasterBackend.h>

namespace irr
{
namespace asset
{
namespace material_compiler
{

auto CMaterialCompilerGLSLRasterBackend::compile(SContext* _ctx, IR* _ir) -> result_t
{
    result_t res = base_t::compile(_ctx, _ir, false);

    res.fragmentShaderSource = 
    R"(

#include <irr/builtin/material_compiler/glsl/rasterization/impl.glsl>
    )";

    return res;
}

}}}