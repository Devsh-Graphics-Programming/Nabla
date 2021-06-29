// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
#extension GL_GOOGLE_include_directive : require

#define SPHERE_COUNT 8
#define POLYGON_METHOD 2 // 0 area sampling, 1 solid angle sampling, 2 approximate projected solid angle sampling
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

//
#define TRIANGLE_REFERENCE
#include <nbl/builtin/glsl/sampling/projected_spherical_triangle.glsl>
#include <nbl/builtin/glsl/barycentric/utils.glsl>

//#include <nbl/builtin/glsl/sampling/projected_spherical_rectangle.glsl>
float nbl_glsl_light_deferred_pdf(in Light light, in Ray_t ray)
{
    const Rectangle rect = rectangles[Light_getObjectID(light)];
    
    const ImmutableRay_t _immutable = ray._immutable;
    const vec3 L = _immutable.direction;
#if POLYGON_METHOD==0
    const float dist = ray._mutable.intersectionT;
    return dist*dist/abs(dot(Rectangle_getNormalTimesArea(rect),L));
#else
    #ifdef TRIANGLE_REFERENCE
        const mat3 sphericalVertices[2] = 
        {
            nbl_glsl_shapes_getSphericalTriangle(mat3(rect.offset,rect.offset+rect.edge0,rect.offset+rect.edge1),_immutable.origin),
            nbl_glsl_shapes_getSphericalTriangle(mat3(rect.offset+rect.edge1,rect.offset+rect.edge0,rect.offset+rect.edge0+rect.edge1),_immutable.origin)
        };
        float solidAngle[2];
        vec3 cos_vertices[2],sin_vertices[2];
        float cos_a[2],cos_c[2],csc_b[2],csc_c[2];
        for (uint i=0u; i<2u; i++)
            solidAngle[i] = nbl_glsl_shapes_SolidAngleOfTriangle(sphericalVertices[i],cos_vertices[i],sin_vertices[i],cos_a[i],cos_c[i],csc_b[i],csc_c[i]);
        const float rectSolidAngle = solidAngle[0]+solidAngle[1];
        #if POLYGON_METHOD==1
            return 1.f/rectSolidAngle;
        #elif POLYGON_METHOD==2
            // TODO: figure out what breaks for a directly visible light under MIS
            if (rectSolidAngle>FLT_MIN)
            {
                const vec2 bary = nbl_glsl_barycentric_reconstructBarycentrics(L*ray._mutable.intersectionT+_immutable.origin-rect.offset,mat2x3(rect.edge0,rect.edge1));
                const uint i = bary.x>=0.f&&bary.y>=0.f&&(bary.x+bary.y)<=1.f ? 0u:1u;

                float pdf = nbl_glsl_sampling_probProjectedSphericalTriangleSample(solidAngle[i],cos_vertices[i],sin_vertices[i],cos_a[i],cos_c[i],csc_b[i],csc_c[i],sphericalVertices[i],_immutable.normalAtOrigin,_immutable.wasBSDFAtOrigin,L);
                pdf *= solidAngle[i]/rectSolidAngle;
                return pdf;
            }
            else
                return FLT_INF;
        #endif
    #else
        #if POLYGON_METHOD==1
        #elif POLYGON_METHOD==2
        #endif
    #endif
#endif
}

vec3 nbl_glsl_light_generate_and_pdf(out float pdf, out float newRayMaxT, in vec3 origin, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in bool isBSDF, in vec3 xi, in uint objectID)
{
    const Rectangle rect = rectangles[objectID];
    const vec3 N = Rectangle_getNormalTimesArea(rect);

    const vec3 origin2origin = rect.offset-origin;
#if POLYGON_METHOD==0
    vec3 L = origin2origin+rect.edge0*xi.x+rect.edge1*xi.y; // TODO: refactor
    
    const float distanceSq = dot(L,L);
    const float rcpDistance = inversesqrt(distanceSq);
    L *= rcpDistance;
    
    pdf = distanceSq/abs(dot(N,L));
    newRayMaxT = 1.0/rcpDistance;
    return L;
#else 
    #ifdef TRIANGLE_REFERENCE
        const mat3 sphericalVertices[2] = 
        {
            nbl_glsl_shapes_getSphericalTriangle(mat3(rect.offset,rect.offset+rect.edge0,rect.offset+rect.edge1),origin),
            nbl_glsl_shapes_getSphericalTriangle(mat3(rect.offset+rect.edge1,rect.offset+rect.edge0,rect.offset+rect.edge0+rect.edge1),origin)
        };
        float solidAngle[2];
        vec3 cos_vertices[2],sin_vertices[2];
        float cos_a[2],cos_c[2],csc_b[2],csc_c[2];
        for (uint i=0u; i<2u; i++)
            solidAngle[i] = nbl_glsl_shapes_SolidAngleOfTriangle(sphericalVertices[i],cos_vertices[i],sin_vertices[i],cos_a[i],cos_c[i],csc_b[i],csc_c[i]);
        vec3 L = vec3(0.f,0.f,0.f);
        const float rectangleSolidAngle = solidAngle[0]+solidAngle[1];
        if (rectangleSolidAngle>FLT_MIN)
        {
            float rcpTriangleChoiceProb;
            const uint i = nbl_glsl_partitionRandVariable(solidAngle[0]/rectangleSolidAngle,xi.z,rcpTriangleChoiceProb) ? 1u:0u;
        #if POLYGON_METHOD==1
            L = nbl_glsl_sampling_generateSphericalTriangleSample(solidAngle[i],cos_vertices[i],sin_vertices[i],cos_a[i],cos_c[i],csc_b[i],csc_c[i],sphericalVertices[i],xi.xy);
            pdf = 1.f/rectangleSolidAngle;
        #elif POLYGON_METHOD==2
            float rcpPdf;
            L = nbl_glsl_sampling_generateProjectedSphericalTriangleSample(rcpPdf,solidAngle[i],cos_vertices[i],sin_vertices[i],cos_a[i],cos_c[i],csc_b[i],csc_c[i],sphericalVertices[i],interaction.isotropic.N,isBSDF,xi.xy);
            pdf = 1.f/(rcpPdf*rcpTriangleChoiceProb);
        #endif
        }
        else
            pdf = FLT_INF;
    #else
        #if POLYGON_METHOD==1
            #error ""
        #elif POLYGON_METHOD==2
            #error ""
        #endif
    #endif
    newRayMaxT = dot(N,origin2origin)/dot(N,L);
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