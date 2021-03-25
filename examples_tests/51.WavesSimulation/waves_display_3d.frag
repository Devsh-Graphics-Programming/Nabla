#version 430 core

layout(location = 0) out vec4 pixelColor;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 world_coord;

layout( push_constant ) uniform Block {
	layout(offset = 64) vec3 camera_pos;
} u_pc;

vec3 fresnel(float cosTheta, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}
const vec4 water_color = vec4(0, 0.1, 0.1, 1);
void main()
{
    vec3 normal_normalized = normalize(normal);
    const vec3 light_pos = vec3(0, 128, 0);     
    vec3 light_vector = (light_pos - world_coord);
    vec3 view_vector = normalize(u_pc.camera_pos - world_coord);
    
    vec3 f0 = vec3(0.04);
    vec3 f  = fresnel(max(dot(normal_normalized, view_vector), 0.0), f0);
    
    vec3 albedo = mix(water_color.rgb, vec3(1, 1, 1), f);

    vec3  color = max(dot(normal_normalized, light_vector), 0) * albedo.rgb;
    pixelColor = vec4(albedo, 1);
}