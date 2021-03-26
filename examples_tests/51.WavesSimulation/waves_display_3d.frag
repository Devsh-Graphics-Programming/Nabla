#version 430 core

layout(location = 0) out vec4 pixelColor;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 world_coord;

layout( push_constant ) uniform Block {
	layout(offset = 64) vec3 camera_pos;
} u_pc;

const vec3 water_color = vec3(0, 0, 0.1);
const vec3 light_color = vec3(0.1, 0.1, 0.1);
const vec3 light_pos = vec3(0, 128, 0);     

vec3 fresnel(float cosTheta, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}
void main()
{
    vec3 normal_normalized = normalize(normal);
    vec3 light = normalize(light_pos - world_coord);
    vec3 view = normalize(u_pc.camera_pos - world_coord);
    
    vec3 f0 = vec3(0.04);
    vec3 f  = fresnel(max(dot(normal_normalized, view), 0.0), f0);
    
    vec3 albedo = mix(water_color.rgb, light_color, f);

    vec3  color = max(dot(normal_normalized, light), 0) * albedo.rgb;
    pixelColor = vec4(color, 1);
}