#version 430 core
#include "nbl/builtin/glsl/bxdf/fresnel.glsl"

layout (set = 0, binding = 1) uniform sampler2D normal_map;
layout (set = 0, binding = 2) uniform sampler2D env_map;

layout(location = 0) out vec4 pixelColor;

layout(location = 0) in vec2 texture_pos;
layout(location = 1) in vec3 world_coord;

layout( push_constant ) uniform Block {
	layout(offset = 64) vec3 camera_pos;
} u_pc;

const vec3 water_color = vec3(0, 0.16, 0.43);
const vec3 sky_color = vec3(1, 2.5, 2.5);
const vec3 light_pos = vec3(17000, 100, -17000);     
const float light_intensity = 1;
vec2 getSphericalUV(vec3 r)
{
    return normalize(vec2(acos(r.z), atan(r.y, r.x))); 
}

void main()
{
    vec3 normal = normalize(texture(normal_map, texture_pos).rgb);
    vec3 light = normalize(vec3(100, 1, 100));
    vec3 view = normalize(u_pc.camera_pos - world_coord);
    vec3 r = reflect(-light, normal);

    vec3 f0 = vec3(0.02);
    vec3 f  = nbl_glsl_fresnel_schlick(f0, abs(dot(normal, view)));
    
    vec2 spherical_uv = getSphericalUV(r);
    vec3 sky = texture(env_map, spherical_uv).rgb * light_intensity;
    vec3 albedo = mix(water_color.rgb, sky, f);

    vec3 color = max(dot(normal, light), 0.) * albedo.rgb;
   
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    pixelColor = vec4(color, 1);
}