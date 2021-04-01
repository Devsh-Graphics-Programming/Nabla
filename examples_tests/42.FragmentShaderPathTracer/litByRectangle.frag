// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
#extension GL_GOOGLE_include_directive : require

#define SPHERE_COUNT 8
#define POLYGON_METHOD 0 // 0 area sampling, 1 solid angle sampling, 2 approximate projected solid angle sampling
#include "common.glsl"


#define RECTANGLE_COUNT 1
const vec3 edge0 = normalize(vec3(2,0,-1));
const vec3 edge1 = normalize(vec3(2,-5,4));
Rectangle rectangles[RECTANGLE_COUNT] = {
    Rectangle_Rectangle(vec3(-3.8,0.35,1.3),edge0*7.0,edge1*0.1,INVALID_ID_16BIT,0u)
};


void traceRay_extraShape(inout int objectID, inout float intersectionT, in vec3 origin, in vec3 direction)
{
	for (int i=0; i<RECTANGLE_COUNT; i++)
    {
        float t = Rectangle_intersect(rectangles[i],origin,direction);
        bool closerIntersection = t>0.0 && t<intersectionT;

		objectID = closerIntersection ? (i+SPHERE_COUNT):objectID;
        intersectionT = closerIntersection ? t:intersectionT;
    }
}

/// #include <nbl/builtin/glsl/sampling/projected_spherical_rectangle.glsl>
float nbl_glsl_light_deferred_pdf(in Light light, in Ray_t ray)
{
    const Rectangle rect = rectangles[Light_getObjectID(light)];

    const vec3 L = ray._immutable.direction;
#if POLYGON_METHOD==0
    const float dist = ray._mutable.intersectionT;
    return dist*dist/abs(dot(Rectangle_getNormalTimesArea(rect),L));
#else
    const ImmutableRay_t _immutable = ray._immutable;
    const mat3 sphericalVertices = nbl_glsl_shapes_getSphericalTriangle(mat3(tri.vertex0,tri.vertex1,tri.vertex2),_immutable.origin);
    #if POLYGON_METHOD==1
        const float rcpProb = nbl_glsl_shapes_SolidAngleOfTriangle(sphericalVertices);
        // if `rcpProb` is NAN then the triangle's solid angle was close to 0.0 
        return rcpProb>FLT_MIN ? (1.0/rcpProb):FLT_MAX;
    #elif POLYGON_METHOD==2
        const float pdf = nbl_glsl_sampling_probProjectedSphericalTriangleSample(sphericalVertices,_immutable.normalAtOrigin,_immutable.wasBSDFAtOrigin,L);
        // if `pdf` is NAN then the triangle's projected solid angle was close to 0.0, if its close to INF then the triangle was very small
        return pdf<FLT_MAX ? pdf:0.0;
    #endif
#endif
}

vec3 nbl_glsl_light_generate_and_pdf(out float pdf, out float newRayMaxT, in vec3 origin, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in bool isBSDF, in vec3 xi, in uint objectID)
{
    const Rectangle rect = rectangles[objectID];
    
#if POLYGON_METHOD==0
    const vec3 point = rect.offset+rect.edge0*xi.x+rect.edge1*xi.y; // TODO: refactor
    const vec3 L = point-origin;
    
    const float distanceSq = dot(L,L);
    const float rcpDistance = inversesqrt(distanceSq);
    
    pdf = distanceSq/abs(dot(Rectangle_getNormalTimesArea(rect),L));
    newRayMaxT = 1.0/rcpDistance;
    return L*rcpDistance;
#else 
    float rcpPdf;

    const mat3 sphericalVertices = nbl_glsl_shapes_getSphericalTriangle(mat3(tri.vertex0,tri.vertex1,tri.vertex2),origin);
#if POLYGON_METHOD==1
    const vec3 L = nbl_glsl_sampling_generateSphericalTriangleSample(rcpPdf,sphericalVertices,xi.xy);
#elif POLYGON_METHOD==2
    const vec3 L = nbl_glsl_sampling_generateProjectedSphericalTriangleSample(rcpPdf,sphericalVertices,interaction.isotropic.N,isBSDF,xi.xy);
#endif

    // if `rcpProb` is NAN or negative then the triangle's solidAngle or projectedSolidAngle was close to 0.0 
    pdf = rcpPdf>FLT_MIN ? (1.0/rcpPdf):0.0;

    const vec3 N = Triangle_getNormalTimesArea(tri);
    newRayMaxT = dot(N,tri.vertex0-origin)/dot(N,L);
    return L;
#endif
}


uint getBSDFLightIDAndDetermineNormal(out vec3 normal, in uint objectID, in vec3 intersection)
{
    if (objectID<SPHERE_COUNT)
    {
        Sphere sphere = spheres[objectID];
        normal = Sphere_getNormal(sphere,intersection);
        return sphere.bsdfLightIDs;
    }
    else
    {
        Rectangle rect = rectangles[objectID-SPHERE_COUNT];
        normal = normalize(Rectangle_getNormalTimesArea(rect));
        return rect.bsdfLightIDs;
    }
}