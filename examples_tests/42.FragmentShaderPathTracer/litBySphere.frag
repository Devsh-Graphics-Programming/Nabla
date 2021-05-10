// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
#extension GL_GOOGLE_include_directive : require

#define SPHERE_COUNT 9
#include "common.glsl"


void traceRay_extraShape(inout int objectID, inout float intersectionT, in vec3 origin, in vec3 direction)
{
}

float nbl_glsl_light_deferred_pdf(in Light light, in Ray_t ray)
{
    const Sphere sphere = spheres[ray._mutable.objectID];
    return 1.0/Sphere_getSolidAngle(sphere,ray._immutable.origin);
}

vec3 nbl_glsl_light_generate_and_pdf(out float pdf, out float newRayMaxT, in vec3 origin, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in bool isBSDF, in vec3 xi, in uint objectID)
{
    const Sphere sphere = spheres[objectID];

    vec3 Z = sphere.position-origin;
    const float distanceSQ = dot(Z,Z);
    const float cosThetaMax2 = 1.0-sphere.radius2/distanceSQ;
    if (cosThetaMax2>0.0)
    {
        const float rcpDistance = inversesqrt(distanceSQ);
        Z *= rcpDistance;
    
        const float cosThetaMax = sqrt(cosThetaMax2);
        const float cosTheta = mix(1.0,cosThetaMax,xi.x);

        vec3 L = Z*cosTheta;

        const float cosTheta2 = cosTheta*cosTheta;
        const float sinTheta = sqrt(1.0-cosTheta2);
        float sinPhi,cosPhi;
        nbl_glsl_sincos(2.0*nbl_glsl_PI*xi.y-nbl_glsl_PI,sinPhi,cosPhi);
        mat2x3 XY = nbl_glsl_frisvad(Z);
    
        L += (XY[0]*cosPhi+XY[1]*sinPhi)*sinTheta;
    
        newRayMaxT = (cosTheta-sqrt(cosTheta2-cosThetaMax2))/rcpDistance;
        pdf = 1.0/Sphere_getSolidAngle_impl(cosThetaMax);
        return L;
    }
    pdf = 0.0;
    return vec3(0.0,0.0,0.0);
}

uint getBSDFLightIDAndDetermineNormal(out vec3 normal, in uint objectID, in vec3 intersection)
{
    Sphere sphere = spheres[objectID];
    normal = Sphere_getNormal(sphere,intersection);
    return sphere.bsdfLightIDs;
}