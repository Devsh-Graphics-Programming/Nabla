#version 450

layout (set = 0, binding = 0) uniform sampler2D displacement_map;

layout(location = 0) in vec4 world_pos; 
layout(location = 1) in vec2 texture_pos; 

layout( push_constant, row_major ) uniform Block {
	mat4 vp;
} u_pc;

layout(location = 0) out vec2 v_tex_coord;
layout(location = 1) out vec3 v_world_coord;

const float scale = 100.; 
void main()
{
	uvec2 size = textureSize(displacement_map, 0);
	vec3 displacement = texture(displacement_map, texture_pos).rgb;
	displacement = displacement.xzy;
	vec4 pos = vec4((world_pos.xyz + displacement) * scale, 1);
    gl_Position = u_pc.vp * pos;

	v_world_coord = pos.xyz;
	v_tex_coord = texture_pos;
}