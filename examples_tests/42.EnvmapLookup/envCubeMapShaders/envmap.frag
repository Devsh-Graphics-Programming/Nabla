#version 430 core

#define INVERSE_OF_PI 0.318309
#define INVERSE_OF_2PI 0.159154

layout(set = 3, binding = 0) uniform sampler2D envMap; 
layout(location = 0) in vec3 localCubePosition;
layout(location = 0) out vec4 pixelColor;

const vec2 inverseAtan = vec2(INVERSE_OF_2PI, INVERSE_OF_PI);
vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= inverseAtan;
    uv += 0.5;
    return uv;
}

void main()
{
	vec2 uv = SampleSphericalMap(normalize(localCubePosition));
    vec3 hdrColor = texture(envMap, uv).rgb;
  
    // reinhard tone mapping
    vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
  
    pixelColor = vec4(mapped, 1.0);
}	