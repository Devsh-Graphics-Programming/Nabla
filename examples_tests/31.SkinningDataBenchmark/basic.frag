#version 460 core

layout(location = 1) in vec3 worldPos;
layout(location = 2) in vec3 vNormal;

layout(location = 0) out vec4 pixelColor;

void main()
{
    const vec3 lightPos = vec3(5.0f, 5.0f, 5.0f);
    const vec3 fragLightVec = normalize(lightPos - worldPos);

    const vec3 lightColor = vec3(1.0);
    const vec3 ambient = vec3(0.05);

    float lightIntensity = clamp(dot(vNormal, fragLightVec), 0.0, 1.0);
    pixelColor = vec4(lightIntensity * lightColor + ambient, 1.0);
}