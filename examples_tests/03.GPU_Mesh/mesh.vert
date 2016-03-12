#version 330 compatibility
layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 1) in vec4 vCol; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0

out vec4 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    gl_Position = gl_ModelViewProjectionMatrix*vPos; //only thing preventing the shader from being core-compliant
    Color = vCol;
}
