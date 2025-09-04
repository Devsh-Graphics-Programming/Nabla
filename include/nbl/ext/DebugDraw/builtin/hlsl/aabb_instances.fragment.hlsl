#pragma shader_stage(fragment)

#include "common.hlsl"

using namespace nbl::ext::debug_draw;

[shader("pixel")]
float32_t4 main(PSInput input) : SV_TARGET
{
    float32_t4 outColor = input.color;

    return outColor;
}