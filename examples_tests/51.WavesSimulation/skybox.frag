#version 430 core
#include "waves_common.glsl"

layout (set = 0, binding = 0) uniform sampler2D env_map;

layout(location = 0) in vec3 vNormal;

layout(location = 0) out vec4 pixelColor;

const float pi_r = 1 / nbl_glsl_PI;
const float twopi_r = 1 / (2 * nbl_glsl_PI);

void main()
{
    vec3 normal = normalize(vNormal);
    vec2  uv;
    uv.x = abs(0.5 - atan(normal.z, normal.x) * twopi_r);
    uv.y = abs(0.5 + asin(-normal.y) * pi_r);
   
   pixelColor = textureLod(env_map, uv, 0);
}