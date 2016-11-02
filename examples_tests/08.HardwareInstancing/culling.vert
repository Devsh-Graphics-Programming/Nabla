#version 330 core

layout(location = 0) in vec4 vWorldMatPart0;
layout(location = 1) in vec4 vWorldMatPart1;
layout(location = 2) in vec4 vWorldMatPart2;
layout(location = 3) in vec4 vNormalMatPart0;
layout(location = 4) in vec4 vNormalMatPart1;
layout(location = 5) in uvec2 vNormalMatPart2;

out vec4 gWorldMatPart0;
out vec4 gWorldMatPart1;
out vec4 gWorldMatPart2;

out mat3 gNormalMat;

void main()
{
    gWorldMatPart0 = vWorldMatPart0;
    gWorldMatPart1 = vWorldMatPart1;
    gWorldMatPart2 = vWorldMatPart2;

    gNormalMat = mat3(vNormalMatPart0.xyz,vNormalMatPart0.w,vNormalMatPart1.xy,vNormalMatPart1.zw,uintBitsToFloat(vNormalMatPart2.x));
}
