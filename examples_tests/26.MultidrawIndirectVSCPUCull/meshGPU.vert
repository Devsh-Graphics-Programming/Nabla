#version 460 core

#include "commonVertexShader.glsl"

void main()
{
    impl(gl_DrawID);
}

