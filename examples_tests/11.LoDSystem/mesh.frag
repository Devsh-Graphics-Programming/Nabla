// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#version 430 core

layout(location = 0) in vec3 Normal;
layout(location = 1) in flat uint LoD;

layout(location = 0) out vec4 pixelColor;

const vec3 kLoDColors[7] = vec3[7](
	vec3(0.f,0.f,1.f),
	vec3(0.f,1.f,1.f),
	vec3(0.f,1.f,0.f),
	vec3(0.f,0.5f,0.f),
	vec3(1.f,0.5f,0.f),
	vec3(1.f,0.f,0.f),
	vec3(0.5f,0.f,0.f)
);

void main()
{
    float ambient = 0.2;
    float diffuse = 0.8;
    float cos_theta_term = max(dot(Normal,normalize(vec3(1.0,1.0,1.0))),0.0);

    float fresnel = 0.0; //not going to implement yet, not important
    float specular = 0.0;///pow(max(dot(halfVector,Normal),0.0),shininess);

    const float sunPower = 3.14156;

    pixelColor = vec4(kLoDColors[LoD]*sunPower*(ambient+mix(diffuse,specular,fresnel)*cos_theta_term/3.14159),1.f);
}
