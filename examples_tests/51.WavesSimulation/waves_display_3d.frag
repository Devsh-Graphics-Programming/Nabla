#version 430 core

layout(location = 0) out vec4 pixelColor;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 world_coord;

const vec3 light_pos = vec3(128, 128, 128); 
const vec4 water_color = vec4(0, 1, 1, 1);
void main()
{
    vec3 light_vector = normalize(light_pos - world_coord);

    pixelColor = water_color * dot(normal, light_vector);
}