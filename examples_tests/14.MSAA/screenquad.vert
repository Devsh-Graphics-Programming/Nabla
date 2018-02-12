#version 330 core
layout(location = 0) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 1) in vec2 vTC;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(vPos,1.0); //only thing preventing the shader from being core-compliant
    TexCoord = vTC;
}

