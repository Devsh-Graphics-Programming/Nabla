#version 330 core

layout(location = 0 ) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 1 ) in vec4 vCol; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2 ) in vec4 vProjViewWorldMatCol3;
layout(location = 3 ) in vec3 vNormal;
layout(location = 4 ) in vec4 vProjViewWorldMatCol0;
layout(location = 5 ) in vec4 vProjViewWorldMatCol1;
layout(location = 6 ) in vec4 vProjViewWorldMatCol2;
layout(location = 7) in vec3 vWorldViewMatCol0;
layout(location = 8) in vec3 vWorldViewMatCol1;
layout(location = 9) in vec3 vWorldViewMatCol2;
layout(location = 10) in vec3 vWorldViewMatCol3;

out vec4 Color; //per vertex output color, will be interpolated across the triangle
out vec3 Normal;
out vec3 lightDir;

void main()
{
    gl_Position = vProjViewWorldMatCol0*vPos.x+vProjViewWorldMatCol1*vPos.y+vProjViewWorldMatCol2*vPos.z+vProjViewWorldMatCol3;
    Color = vec4(1.0,0.0,0.0,1.0);
    Normal = inverse(transpose(mat3(vWorldViewMatCol0,vWorldViewMatCol1,vWorldViewMatCol2)))*vNormal; //have to normalize twice because of normal quantization
    lightDir = -vWorldViewMatCol0*vPos.x-vWorldViewMatCol1*vPos.y-vWorldViewMatCol2*vPos.z-vWorldViewMatCol3;
}
