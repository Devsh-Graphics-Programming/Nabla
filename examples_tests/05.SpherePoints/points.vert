#version 330 core
layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0

uniform mat4 MVP;

out vec4 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    gl_Position = MVP*vec4(normalize(vPos.xyz),1.0); //only thing preventing the shader from being core-compliant
    Color = vec4(1.0);
}
