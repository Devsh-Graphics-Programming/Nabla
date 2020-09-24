// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
layout(location = 0) uniform mat4 MVP;

layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec3 vNormal;
layout(location = 4) in vec4 vProjViewWorldMatCol0;
layout(location = 5) in vec4 vProjViewWorldMatCol1;
layout(location = 6) in vec4 vProjViewWorldMatCol2;
layout(location = 7) in vec4 vProjViewWorldMatCol3;
layout(location = 8) in vec3 vTInvWorldMatCol0;
layout(location = 9) in vec3 vTInvWorldMatCol1;
layout(location = 10) in vec3 vTInvWorldMatCol2;

out vec2 uv;
out vec3 Normal;

void main()
{
    gl_Position = vProjViewWorldMatCol0*vPos.x+vProjViewWorldMatCol1*vPos.y+vProjViewWorldMatCol2*vPos.z+vProjViewWorldMatCol3;
	uv = vUV;
    Normal = vTInvWorldMatCol0*vNormal.x+vTInvWorldMatCol1*vNormal.y+vTInvWorldMatCol2*vNormal.z;
}
