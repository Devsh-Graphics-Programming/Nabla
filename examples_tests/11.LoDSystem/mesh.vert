// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

struct PerViewPerInstance_t
{
    mat4 modelViewProjectionMatrix;
};
//#include "common.glsl"
layout(set = 1, binding = 0, std430, row_major) restrict readonly buffer PerViewPerInstance
{
    PerViewPerInstance_t data[];
} perViewPerInstance;


layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 15) in uvec2 InstanceGUID_PerViewPerInstance;

layout(location = 0) out vec3 Normal;
layout(location = 1) out flat uint LoD;

void main()
{
	const mat4 mvp = perViewPerInstance.data[InstanceGUID_PerViewPerInstance[1]].modelViewProjectionMatrix;
    const mat3 normalMatrix = mat3(vec3(1.f,0.f,0.f),vec3(0.f,1.f,0.f),vec3(0.f,0.f,1.f)); // TODO: data[InstanceGUID_PerViewPerInstance[0]].worldInverseTransform

    gl_Position = mvp[0]*vPos.x+mvp[1]*vPos.y+mvp[2]*vPos.z+mvp[3];
    Normal = normalize(normalMatrix*vNormal); //have to normalize twice because of normal quantization
    LoD = 2u; // TODO: visualize the LoD somehow
}
