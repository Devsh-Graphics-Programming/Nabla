#version 330 core
layout(location = 0) in vec4 posAndTC;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(posAndTC.xy,0.0,1.0); //only thing preventing the shader from being core-compliant
    TexCoord = posAndTC.zw;
}

