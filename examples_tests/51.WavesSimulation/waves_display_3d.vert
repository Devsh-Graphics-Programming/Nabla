#version 450

layout(location = 0) in vec4 world_pos; 
layout(location = 1) in vec2 texture_pos; 
layout (set = 0, binding = 0) uniform sampler2D displacement_map;
layout (set = 0, binding = 1) uniform sampler2D normal_map;

layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} u_pc;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_world_coord;

void main()
{
	uvec2 size = textureSize(displacement_map, 0);
	vec3 displacement = texture(displacement_map, texture_pos).rgb / 2;
	vec4 pos = world_pos + vec4(displacement, 0);
	pos *= size.x;
    gl_Position = u_pc.modelViewProj * pos;

	v_normal = texture(normal_map, texture_pos).rgb;
	v_world_coord = pos.xyz;
}