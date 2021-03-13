// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
#extension GL_GOOGLE_include_directive : require

#define SPHERE_COUNT 8
#define POLYGON_METHOD 0 // 0 area sampling, 1 solid angle sampling, 2 approximate projected solid angle sampling
#include "common.glsl"

#define TRIANGLE_COUNT 1
Triangle triangles[TRIANGLE_COUNT] = {
    Triangle_Triangle(mat3(vec3(-1.8,0.35,0.3),vec3(-1.2,0.35,0.0),vec3(-1.5,0.8,-0.3)),INVALID_ID_16BIT,0u)
};

void traceRay_extraShape(inout int objectID, inout float intersectionT, in vec3 origin, in vec3 direction)
{
	for (int i=0; i<TRIANGLE_COUNT; i++)
    {
        float t = Triangle_intersect(triangles[i],origin,direction);
        bool closerIntersection = t>0.0 && t<intersectionT;

		objectID = closerIntersection ? (i+SPHERE_COUNT):objectID;
        intersectionT = closerIntersection ? t:intersectionT;
    }
}


#include <nbl/builtin/glsl/sampling/projected_spherical_triangle.glsl>
float nbl_glsl_light_deferred_pdf(in Light light, in Ray_t ray)
{
    const Triangle tri = triangles[Light_getObjectID(light)];

    const vec3 L = ray._immutable.direction;
#if POLYGON_METHOD==0
    const float dist = ray._mutable.intersectionT;
    return dist*dist/abs(dot(Triangle_getNormalTimesArea(tri),L));
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
    const Triangle tri = triangles[objectID];
    
#if POLYGON_METHOD==0
    const mat2x3 edges = mat2x3(tri.vertex1-tri.vertex0,tri.vertex2-tri.vertex0);
    const float sqrtU = sqrt(xi.x);
    vec3 point = tri.vertex0+edges[0]*(1.0-sqrtU)+edges[1]*sqrtU*xi.y;
    const vec3 L = point-origin;
    
    const float distanceSq = dot(L,L);
    const float rcpDistance = inversesqrt(distanceSq);
    
    pdf = distanceSq/abs(dot(Triangle_getNormalTimesArea_impl(edges),L));
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


bool closestHitProgram(in uint depth, in uint _sample, inout Ray_t ray, inout nbl_glsl_xoroshiro64star_state_t scramble_state)
{
    const MutableRay_t _mutable = ray._mutable;

    const uint objectID = _mutable.objectID;
    Sphere sphere = spheres[_mutable.objectID];

    // interaction stuffs
    const ImmutableRay_t _immutable = ray._immutable;
    const vec3 intersection = _immutable.origin+_immutable.direction*_mutable.intersectionT;
    uint bsdfLightIDs;
    nbl_glsl_AnisotropicViewSurfaceInteraction interaction;
    {
        nbl_glsl_IsotropicViewSurfaceInteraction isotropic;

        isotropic.V.dir = -_immutable.direction;
        //isotropic.V.dPosdScreen = screw that
        if (objectID<SPHERE_COUNT)
        {
            Sphere sphere = spheres[objectID];
            isotropic.N = Sphere_getNormal(sphere,intersection);
            bsdfLightIDs = sphere.bsdfLightIDs;
        }
        else
        {
            Triangle tri = triangles[objectID-SPHERE_COUNT];
            isotropic.N = normalize(Triangle_getNormalTimesArea(tri));
            bsdfLightIDs = tri.bsdfLightIDs;
        }
        isotropic.NdotV = dot(isotropic.V.dir,isotropic.N);
        isotropic.NdotV_squared = isotropic.NdotV*isotropic.NdotV;

        interaction = nbl_glsl_calcAnisotropicInteraction(isotropic);
    }

    //
    vec3 throughput = ray._payload.throughput;

    // add emissive and finish MIS
    const uint lightID = bitfieldExtract(bsdfLightIDs,16,16);
    if (lightID!=INVALID_ID_16BIT) // has emissive
    {
        float lightPdf;
        ray._payload.accumulation += nbl_glsl_light_deferred_eval_and_prob(lightPdf,lights[lightID],ray)*throughput/(1.0+lightPdf*lightPdf*ray._payload.otherTechniqueHeuristic);
    }

    // check if we even have a BSDF at all
    uint bsdfID = bitfieldExtract(bsdfLightIDs,0,16);
    if (bsdfID!=INVALID_ID_16BIT)
    {
        BSDFNode bsdf = bsdfs[bsdfID];
        #ifdef KILL_DIFFUSE_SPECULAR_PATHS
        if (BSDFNode_isNotDiffuse(bsdf))
        {
            if (ray._payload.hasDiffuse)
                return true;
        }
        else
            ray._payload.hasDiffuse = true;
        #endif
        
        const bool isBSDF = BSDFNode_isBSDF(bsdf);
        //rand
        mat2x3 epsilon = rand3d(depth,_sample,scramble_state);
        
        // thresholds
        const float bsdfPdfThreshold = 0.0001;
        const float lumaContributionThreshold = getLuma(nbl_glsl_eotf_sRGB(vec3(1.0)/255.0)); // OETF smallest perceptible value
        const vec3 throughputCIE_Y = transpose(nbl_glsl_sRGBtoXYZ)[1]*throughput;
        const float monochromeEta = dot(throughputCIE_Y,BSDFNode_getEta(bsdf)[0])/(throughputCIE_Y.r+throughputCIE_Y.g+throughputCIE_Y.b);
           
        // do NEE
        const float neeSkipProbability = BSDFNode_getNEESkipProb(bsdf);
        float rcpChoiceProb;
        if (nbl_glsl_partitionRandVariable(neeSkipProbability,epsilon[0].z,rcpChoiceProb))
        {
            vec3 neeContrib; float lightPdf, t;
            nbl_glsl_LightSample nee_sample = nbl_glsl_light_generate_and_remainder_and_pdf(
                neeContrib,lightPdf,t,
                intersection,interaction,
                isBSDF,epsilon[0],depth
            );
            // We don't allow non watertight transmitters in this renderer
            bool validPath = nee_sample.NdotL>0.0;
            // but if we allowed non-watertight transmitters (single water surface), it would make sense just to apply this line by itself
            nbl_glsl_AnisotropicMicrofacetCache _cache;
            validPath = validPath && nbl_glsl_calcAnisotropicMicrofacetCache(_cache,interaction,nee_sample,monochromeEta);
            if (validPath)
            {
                float bsdfPdf;
                neeContrib *= nbl_glsl_bsdf_cos_remainder_and_pdf(bsdfPdf,nee_sample,interaction,bsdf,monochromeEta,_cache)*throughput;
                const float oc = bsdfPdf*rcpChoiceProb;
                neeContrib /= 1.0/oc+oc/(lightPdf*lightPdf); // MIS weight
                if (bsdfPdf<FLT_MAX && getLuma(neeContrib)>lumaContributionThreshold && traceRay(t,intersection+nee_sample.L*t*getStartTolerance(depth),nee_sample.L)==-1)
                    ray._payload.accumulation += neeContrib;
            }
        }
        
        // sample BSDF
        float bsdfPdf; vec3 bsdfSampleL;
        {
            nbl_glsl_AnisotropicMicrofacetCache _cache;
            nbl_glsl_LightSample bsdf_sample = nbl_glsl_bsdf_cos_generate(interaction,epsilon[1],bsdf,monochromeEta,_cache);
            // the value of the bsdf divided by the probability of the sample being generated
            throughput *= nbl_glsl_bsdf_cos_remainder_and_pdf(bsdfPdf,bsdf_sample,interaction,bsdf,monochromeEta,_cache);
            //
            bsdfSampleL = bsdf_sample.L;
        }

        // additional threshold
        const float lumaThroughputThreshold = lumaContributionThreshold;
        if (bsdfPdf>bsdfPdfThreshold && getLuma(throughput)>lumaThroughputThreshold)
        {
            ray._payload.throughput = throughput;
            ray._payload.otherTechniqueHeuristic = (1.0-neeSkipProbability)/bsdfPdf; // numerically stable, don't touch
            ray._payload.otherTechniqueHeuristic *= ray._payload.otherTechniqueHeuristic;
                    
            // trace new ray
            ray._immutable.origin = intersection+bsdfSampleL*(1.0/*kSceneSize*/)*getStartTolerance(depth);
            ray._immutable.direction = bsdfSampleL;
            #if POLYGON_METHOD==2
            ray._immutable.normalAtOrigin = interaction.isotropic.N;
            ray._immutable.wasBSDFAtOrigin = isBSDF;
            #endif
            return true;
        }
    }
    return false;
}