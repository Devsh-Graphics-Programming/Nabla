#version 430 core

layout(set = 3, binding = 0) uniform sampler2D envMap; 
layout(location = 0) in vec2 fUV;
layout(location = 0) out vec4 pixelColor;

void main()
{
    vec3 hdrColor = texture(envMap, fUV).rgb;
  
    // reinhard tone mapping
    vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
  
    pixelColor = vec4(mapped, 1.0);
}	