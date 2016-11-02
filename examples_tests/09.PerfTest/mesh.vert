#version 330 core
uniform mat4 MVP;

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec4 vCol;
layout(location = 2) in vec3 vTranslation;

out vec4 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    vec3 newPos = vPos+vTranslation;
    gl_Position = MVP[0]*newPos.x+MVP[1]*newPos.y+MVP[2]*newPos.z+MVP[3]; //faster than normal GLSL shader matrix mul by 1-4 multiplications on floats
    Color = vec4(vPos.xyz+0.5,1.0);
}
