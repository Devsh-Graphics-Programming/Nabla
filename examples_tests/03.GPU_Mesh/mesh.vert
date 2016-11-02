#version 330 core
layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 1) in vec4 vCol; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0

uniform mat4 MVP;

out vec4 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    gl_Position = MVP*vPos; //only thing preventing the shader from being core-compliant
    Color = vCol;
}
