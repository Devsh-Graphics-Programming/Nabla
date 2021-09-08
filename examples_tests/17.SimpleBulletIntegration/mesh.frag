#version 430 core

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

layout(location = 0) in vec3 Color;
layout(location = 1) in vec3 Normal;

layout(location = 0) out vec4 pixelColor;

void main()
{
	vec3 normal = normalize(Normal);

	float ambient = 0.35;
	float diffuse = 0.8;
	float cos_theta_term = max(dot(normal,vec3(3.0,5.0,-4.0)),0.0);

	float fresnel = 0.0; //not going to implement yet, not important
	float specular = 0.0;///pow(max(dot(halfVector,normal),0.0),shininess);

	const float sunPower = 3.14156*0.3;

	pixelColor = vec4(Color, 1)*sunPower*(ambient+mix(diffuse,specular,fresnel)*cos_theta_term/3.14159);
}