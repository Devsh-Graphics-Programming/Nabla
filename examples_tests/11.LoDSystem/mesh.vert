// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

#include "common.glsl"
layout(set = 1, binding = 0, std430, row_major) restrict readonly buffer PerViewPerInstance
{
    PerViewPerInstance_t data[];
} perViewPerInstance;


layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 15) in uvec2 InstanceGUID_PerViewPerInstance;

layout(location = 0) out vec3 Normal;
layout(location = 1) out flat uint LoD;

#include <nbl/builtin/glsl/utils/transform.glsl>
void main()
{
	const PerViewPerInstance_t pvpi = perViewPerInstance.data[InstanceGUID_PerViewPerInstance[1]];
    const mat3 normalMatrix = mat3(vec3(1.f,0.f,0.f),vec3(0.f,1.f,0.f),vec3(0.f,0.f,1.f)); // global mat without rotation or scaling

    gl_Position = nbl_glsl_pseudoMul4x4with3x1(pvpi.mvp,vPos);
    Normal = normalize(normalMatrix*vNormal); //have to normalize twice because of normal quantization
    LoD = pvpi.lod;
}
