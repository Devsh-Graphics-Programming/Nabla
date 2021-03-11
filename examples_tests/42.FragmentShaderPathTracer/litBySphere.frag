// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
#extension GL_GOOGLE_include_directive : require

#define SPHERE_COUNT 9
#include "common.glsl"



void traceRay(in bool anyHit, in ImmutableRay_t _immutable, inout MutableRay_t _mutable)
{
	int objectID = -1;
    float intersectionT = _immutable.maxT;
	for (int i=0; i<SPHERE_COUNT; i++)
    {
        float t = Sphere_intersect(spheres[i],_immutable.origin,_immutable.direction);
        bool closerIntersection = t>0.0 && t<intersectionT;

		objectID = closerIntersection ? i:objectID;
        intersectionT = closerIntersection ? t:intersectionT;
        
        // allowing early out results in a performance regression, WTF!?
        //if (anyHit && closerIntersection && anyHitProgram(_immutable))
           //break;
    }
    _mutable.objectID = objectID;
    _mutable.intersectionT = intersectionT;
}

#if 0
// the interaction here is the interaction at the illuminator-end of the ray, not the receiver
vec3 nbl_glsl_light_deferred_eval_and_prob(out float pdf, in Sphere sphere, in vec3 origin, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in Light light)
{
    // we don't have to worry about solid angle of the light w.r.t. surface of the light because this function only ever gets called from closestHit routine, so such ray cannot be produced
    pdf = scene_getLightChoicePdf(light)/Sphere_getSolidAngle(sphere,origin);
    return Light_getRadiance(light);
}

nbl_glsl_LightSample nbl_glsl_light_generate_and_remainder_and_pdf(out vec3 remainder, out float pdf, out float newRayMaxT, in vec3 origin, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u, in uint depth)
{
    // normally we'd pick from set of lights, using `u.z`
    const Light light = lights[0];
    const float choicePdf = scene_getLightChoicePdf(light);

    const Sphere sphere = spheres[Light_getObjectID(light)];

    vec3 Z = sphere.position-origin;
    const float distanceSQ = dot(Z,Z);
    const float cosThetaMax2 = 1.0-sphere.radius2/distanceSQ;
    const float rcpDistance = inversesqrt(distanceSQ);

    const bool possibilityOfLightVisibility = cosThetaMax2>0.0;
    Z *= rcpDistance;
    
    // following only have valid values if `possibilityOfLightVisibility` is true
    const float cosThetaMax = sqrt(cosThetaMax2);
    const float cosTheta = mix(1.0,cosThetaMax,u.x);

    vec3 L = Z*cosTheta;

    const float cosTheta2 = cosTheta*cosTheta;
    const float sinTheta = sqrt(1.0-cosTheta2);
    float sinPhi,cosPhi;
    nbl_glsl_sincos(2.0*nbl_glsl_PI*u.y-nbl_glsl_PI,sinPhi,cosPhi);
    mat2x3 XY = nbl_glsl_frisvad(Z);
    
    L += (XY[0]*cosPhi+XY[1]*sinPhi)*sinTheta;
    
    const float rcpPdf = Sphere_getSolidAngle_impl(cosThetaMax)/choicePdf;
    remainder = Light_getRadiance(light)*(possibilityOfLightVisibility ? rcpPdf:0.0); // this ternary operator kills invalid rays
    pdf = 1.0/rcpPdf;
    
    newRayMaxT = (cosTheta-sqrt(cosTheta2-cosThetaMax2))/rcpDistance*getEndTolerance(depth);
    
    return nbl_glsl_createLightSample(L,interaction);
}
#endif

bool closestHitProgram(in uint depth, in uint _sample, inout Ray_t ray, inout nbl_glsl_xoroshiro64star_state_t scramble_state)
{
    const MutableRay_t _mutable = ray._mutable;

    Sphere sphere = spheres[_mutable.objectID];
    const uint bsdfLightIDs = sphere.bsdfLightIDs;

    vec3 throughput = ray._payload.throughput;
    
    // add emissive
    const uint lightID = bitfieldExtract(bsdfLightIDs,16,16);
    if (lightID!=INVALID_ID_16BIT) // has emissive
        ray._payload.accumulation += throughput*Light_getRadiance(lights[lightID]);

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

        // intersection stuffs
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
        

    

        float maxT = FLT_MAX;
        nbl_glsl_AnisotropicMicrofacetCache _cache;

        // do I need this?
        const vec3 throughputCIE_Y = transpose(nbl_glsl_sRGBtoXYZ)[1]*throughput;
        const float monochromeEta = dot(throughputCIE_Y,BSDFNode_getEta(bsdf)[0])/(throughputCIE_Y.r+throughputCIE_Y.g+throughputCIE_Y.b);
            
        // sample BSDF
        float bsdfPdf; vec3 bsdfSampleL;
        {
            nbl_glsl_LightSample _sample = nbl_glsl_bsdf_cos_generate(interaction,epsilon[0],bsdf,monochromeEta,_cache);
            // the value of the bsdf divided by the probability of the sample being generated
            throughput *= nbl_glsl_bsdf_cos_remainder_and_pdf(bsdfPdf,_sample,interaction,bsdf,monochromeEta,_cache);
            //
            bsdfSampleL = _sample.L;
        }

        const float bsdfPdfThreshold = 0.0001;
        // OETF smallest perceptible value
        const float lumaThroughputThreshold = getLuma(nbl_glsl_eotf_sRGB(vec3(1.0)/255.0));
        if (bsdfPdf>bsdfPdfThreshold && getLuma(throughput)>lumaThroughputThreshold)
        {
            ray._payload.throughput = throughput;
                    
            // trace new ray
            ray._immutable.origin = intersection+bsdfSampleL*(1.0/*kSceneSize*/)*getStartTolerance(depth);
            ray._immutable.maxT = maxT;
            ray._immutable.direction = bsdfSampleL;
            return false;
        }
    }
    return true;
}
#if 0
bool closestHitProgram(in uint depth, in uint _sample, inout Ray_t ray, inout nbl_glsl_xoroshiro64star_state_t scramble_state)
{
    const MutableRay_t _mutable = ray._mutable;

    Sphere sphere = spheres[_mutable.objectID];
    vec3 intersection = ray._immutable.origin+ray._immutable.direction*_mutable.intersectionT;
    
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

    const uint bsdfLightIDs = sphere.bsdfLightIDs;
    const uint lightID = bitfieldExtract(bsdfLightIDs,16,16);

    vec3 throughput = ray._payload.throughput;
    
    // finish MIS
    if (lightID!=INVALID_ID_16BIT) // has emissive
    {
        float lightPdf;
        vec3 lightVal = nbl_glsl_light_deferred_eval_and_prob(lightPdf,sphere,ray._immutable.origin,interaction,lights[lightID]);
        ray._payload.accumulation += throughput*lightVal/(1.0+lightPdf*lightPdf*ray._payload.otherTechniqueHeuristic);
    }

    // check if we even have a BSDF at all
    uint bsdfID = bitfieldExtract(bsdfLightIDs,0,16);
    if (bsdfID!=INVALID_ID_16BIT)
    {
        // common preload
        BSDFNode bsdf = bsdfs[bsdfID];
        uint opType = BSDFNode_getType(bsdf);

        #ifdef KILL_DIFFUSE_SPECULAR_PATHS
        if (BSDFNode_isNotDiffuse(bsdf))
        {
            if (ray._payload.hasDiffuse)
                return true;
        }
        else
            ray._payload.hasDiffuse = true;
        #endif


        const float bsdfGeneratorProbability = BSDFNode_getMISWeight(bsdf);
        mat2x3 epsilon = rand3d(depth,_sample,scramble_state);
    
        float rcpChoiceProb;
        const bool doNEE = nbl_glsl_partitionRandVariable(bsdfGeneratorProbability,epsilon[0].z,rcpChoiceProb);
    

        float maxT;
        // the probability of generating a sample w.r.t. the light generator only possible and used when it was generated with it!
        float lightPdf;
        nbl_glsl_LightSample _sample;
        nbl_glsl_AnisotropicMicrofacetCache _cache;
        if (doNEE)
        {
            vec3 lightRemainder;
            _sample = nbl_glsl_light_generate_and_remainder_and_pdf(
                lightRemainder,lightPdf,maxT,
                intersection,interaction,epsilon[0],
                depth
            );
            throughput *= lightRemainder;
        }

        bool validPath = true;
        const vec3 throughputCIE_Y = transpose(nbl_glsl_sRGBtoXYZ)[1]*throughput;
        const float monochromeEta = dot(throughputCIE_Y,BSDFNode_getEta(bsdf)[0])/(throughputCIE_Y.r+throughputCIE_Y.g+throughputCIE_Y.b);
        if (doNEE)
        {
            // if we allowed non-watertight transmitters (single water surface), it would make sense just to apply this line.
            validPath = nbl_glsl_calcAnisotropicMicrofacetCache(_cache,interaction,_sample,monochromeEta);
            // but we don't allow non watertight transmitters in this renderer
            validPath = validPath && _sample.NdotL>0.0;
        }
        else
        {
            maxT = FLT_MAX;
            _sample = nbl_glsl_bsdf_cos_generate(interaction,epsilon[0],bsdf,monochromeEta,_cache);
        }
            
        // do a cool trick and always compute the bsdf parts this way! (no divergence)
        float bsdfPdf;
        // the value of the bsdf divided by the probability of the sample being generated
        if (validPath)
            throughput *= nbl_glsl_bsdf_cos_remainder_and_pdf(bsdfPdf,_sample,interaction,bsdf,monochromeEta,_cache);
        else
            throughput = vec3(0.0);

        // OETF smallest perceptible value
        const float bsdfPdfThreshold = getLuma(nbl_glsl_eotf_sRGB(vec3(1.0)/255.0));
        const float lumaThroughputThreshold = bsdfPdfThreshold;
        if (bsdfPdf>bsdfPdfThreshold && (!doNEE || bsdfPdf<FLT_MAX) && getLuma(throughput)>lumaThroughputThreshold)
        {
            ray._payload.throughput = throughput*rcpChoiceProb;

            float heuristicFactor = rcpChoiceProb-1.0; // weightNonGenerator/weightGenerator
            heuristicFactor /= doNEE ? lightPdf:bsdfPdf; // weightNonGenerator/(weightGenerator*probGenerated)
            heuristicFactor *= heuristicFactor; // (weightNonGenerator/(weightGenerator*probGenerated))^2
            if (doNEE)
                heuristicFactor = 1.0/(1.0/bsdfPdf+heuristicFactor*bsdfPdf); // numerically stable, don't touch
            ray._payload.otherTechniqueHeuristic = heuristicFactor;
                    
            // trace new ray
            ray._immutable.origin = intersection+_sample.L*(doNEE ? maxT:1.0/*kSceneSize*/)*getStartTolerance(depth);
            ray._immutable.maxT = maxT;
            ray._immutable.direction = _sample.L;
            ray._immutable.anyHit = doNEE;
            return false;
        }
    }
    return true;
}
#endif