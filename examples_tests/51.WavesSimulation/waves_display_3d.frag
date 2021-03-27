#version 430 core
#include "nbl/builtin/glsl/bxdf/fresnel.glsl"

layout (set = 0, binding = 1) uniform sampler2D normal_map;

layout(location = 0) out vec4 pixelColor;

layout(location = 0) in vec2 texture_pos;
layout(location = 1) in vec3 world_coord;

layout( push_constant ) uniform Block {
	layout(offset = 64) vec3 camera_pos;
} u_pc;

const vec3 water_color = vec3(0.004f, 0.016f, 0.047f);
const vec3 sky_color = vec3(0.1, 0.6, 0.9);
//const vec3 sky_color = vec3(1, 1, 0.9);
const vec3 light_pos = vec3(100, 1000000, 100);     

vec3 fresnel(float cosTheta, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

void main()
{
    vec3 normal = texture(normal_map, texture_pos).rgb;;
    vec3 light = normalize(light_pos - world_coord);
    vec3 view = normalize(u_pc.camera_pos - world_coord);
    
    vec3 f0 = vec3(0.04);
    vec3 f  = nbl_glsl_fresnel_schlick(f0, max(dot(normal, view), 0.0));
    
    vec3 albedo = mix(water_color.rgb, sky_color, f);

    vec3  color = max(dot(normal, light), 0) * albedo.rgb;

    pixelColor = vec4(color, 1);
}