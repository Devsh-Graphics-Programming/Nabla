#version 430 core
layout(location = 0) in vec3 vNormal;

layout(location = 0) out vec4 pixelColor;

void main()
{
    vec3 normColor = vec3(vNormal.x) * 0.5 + vec3(0.5);
    pixelColor = vec4(normColor, 1.0);
}
