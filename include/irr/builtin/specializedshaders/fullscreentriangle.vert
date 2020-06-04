#version 430 core

const vec2 pos[3] = vec2[3](vec2(-1.0, 1.0),vec2(-1.0,-3.0),vec2( 3.0, 1.0));
const vec2 tc[3] = vec2[3](vec2( 0.0, 0.0),vec2( 0.0, 2.0),vec2( 2.0, 0.0));

layout(location = 0) out vec2 TexCoord;
 
void main()
{
    gl_Position = vec4(pos[gl_VertexIndex],0.0,1.0);
    TexCoord = tc[gl_VertexIndex];
}