#version 430 core

layout(location = 0 ) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 1 ) in vec4 vCol; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2 ) in vec4 vProjViewWorldMatCol3;
layout(location = 3 ) in vec3 vNormal;
layout(location = 4 ) in vec4 vProjViewWorldMatCol0;
layout(location = 5 ) in vec4 vProjViewWorldMatCol1;
layout(location = 6 ) in vec4 vProjViewWorldMatCol2;
layout(location = 7) in vec3 vWorldMatCol0;
layout(location = 8) in vec3 vWorldMatCol1;
layout(location = 9) in vec3 vWorldMatCol2;
layout(location = 10) in uvec4 vWorldMatCol3;

out vec4 Color;
out vec3 Normal;

void main()
{
    gl_Position = vProjViewWorldMatCol0*vPos.x+vProjViewWorldMatCol1*vPos.y+vProjViewWorldMatCol2*vPos.z+vProjViewWorldMatCol3;
    Color = vCol*unpackUnorm4x8(vWorldMatCol3.w);
    Normal = inverse(transpose(mat3(vWorldMatCol0,vWorldMatCol1,vWorldMatCol2)))*normalize(vNormal); //have to normalize twice because of normal quantization
}
