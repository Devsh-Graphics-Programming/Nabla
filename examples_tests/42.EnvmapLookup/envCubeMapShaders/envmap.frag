#version 430 core

layout(set = 3, binding = 0) uniform sampler2D envMap; 
layout(location = 0) in vec3 localCubePosition;
layout(location = 0) out vec4 pixelColor;

#define irr_glsl_PI 3.14159265359
#define irr_glsl_RECIPROCAL_PI 0.318309886183

vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= irr_glsl_RECIPROCAL_PI*0.5;
    uv += 0.5;
    return uv;
}

void main()
{
	vec2 uv = SampleSphericalMap(normalize(localCubePosition));
    vec3 hdrColor = textureLod(envMap, uv, 0.0).rgb;
  
    // reinhard tone mapping
    vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
  
    pixelColor = vec4(mapped, 1.0);
}	