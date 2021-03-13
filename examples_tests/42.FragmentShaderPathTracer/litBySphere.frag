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


bool closestHitProgram(in uint depth, in uint _sample, inout Ray_t ray, inout nbl_glsl_xoroshiro64star_state_t scramble_state)
{
    const MutableRay_t _mutable = ray._mutable;

    Sphere sphere = spheres[_mutable.objectID];

    // interaction stuffs
    const vec3 intersection = ray._immutable.origin+ray._immutable.direction*_mutable.intersectionT;
    nbl_glsl_AnisotropicViewSurfaceInteraction interaction;
    {
        nbl_glsl_IsotropicViewSurfaceInteraction isotropic;

        isotropic.V.dir = -ray._immutable.direction;
        //isotropic.V.dPosdScreen = screw that
        isotropic.N = Sphere_getNormal(sphere,intersection);
        isotropic.NdotV = dot(isotropic.V.dir,isotropic.N);
        isotropic.NdotV_squared = isotropic.NdotV*isotropic.NdotV;

        interaction = nbl_glsl_calcAnisotropicInteraction(isotropic);
    }
    
    //
    const uint bsdfLightIDs = sphere.bsdfLightIDs;

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
            nbl_glsl_LightSample nee_sample = nbl_glsl_light_generate_and_remainder_and_pdf(neeContrib,lightPdf,t,intersection,interaction,false,epsilon[0],depth);
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
            return true;
        }
    }
    return false;
}