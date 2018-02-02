#version 330 core
layout(location = 0) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0

void main()
{
    gl_Position = vec4(vPos,1.0);
}

