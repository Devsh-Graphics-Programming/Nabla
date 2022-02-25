// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// basic settings
#define MAX_DEPTH 15
#define SAMPLES 32

// firefly and variance reduction techniques
//#define KILL_DIFFUSE_SPECULAR_PATHS
//#define VISUALIZE_HIGH_VARIANCE

#define INVALID_ID_16BIT 0xffffu
struct Sphere
{
    vec3 position;
    float radius2;
    uint bsdfLightIDs;
}; 

layout(set=0, binding=0, rgba16f) uniform image2D outImage;

layout(set = 2, binding = 0) uniform sampler2D envMap; 
layout(set = 2, binding = 1) uniform usamplerBuffer sampleSequence;
layout(set = 2, binding = 2) uniform usampler2D scramblebuf;
layout(set = 2, binding = 3) uniform accelerationStructureEXT topLevelAS;
layout(set = 2, binding = 4) readonly restrict buffer InputBuffer
{
	Sphere spheres[];
};

#ifndef _NBL_GLSL_WORKGROUP_SIZE_
#define _NBL_GLSL_WORKGROUP_SIZE_ 16
layout(local_size_x=_NBL_GLSL_WORKGROUP_SIZE_, local_size_y=_NBL_GLSL_WORKGROUP_SIZE_, local_size_z=1) in;
#endif

ivec2 getCoordinates() {
    return ivec2(gl_GlobalInvocationID.xy);
}

vec2 getTexCoords() {
    ivec2 imageSize = imageSize(outImage);
    ivec2 iCoords = getCoordinates();
    return vec2(float(iCoords.x) / imageSize.x, 1.0 - float(iCoords.y) / imageSize.y);
}


#include <nbl/builtin/glsl/limits/numeric.glsl>
#include <nbl/builtin/glsl/math/constants.glsl>
#include <nbl/builtin/glsl/utils/common.glsl>

#include <nbl/builtin/glsl/sampling/box_muller_transform.glsl>

layout(set = 1, binding = 0, row_major, std140) uniform UBO
{
	nbl_glsl_SBasicViewParameters params;
} cameraData;

Sphere Sphere_Sphere(in vec3 position, in float radius, in uint bsdfID, in uint lightID)
{
    Sphere sphere;
    sphere.position = position;
    sphere.radius2 = radius*radius;
    sphere.bsdfLightIDs = bitfieldInsert(bsdfID,lightID,16,16);
    return sphere;
}

// return intersection distance if found, FLT_NAN otherwise
float Sphere_intersect(in Sphere sphere, in vec3 origin, in vec3 direction)
{
    vec3 relOrigin = origin-sphere.position;
    float relOriginLen2 = dot(relOrigin,relOrigin);
    const float radius2 = sphere.radius2;

    float dirDotRelOrigin = dot(direction,relOrigin);
    float det = radius2-relOriginLen2+dirDotRelOrigin*dirDotRelOrigin;

    // do some speculative math here
    float detsqrt = sqrt(det);
    return -dirDotRelOrigin+(relOriginLen2>radius2 ? (-detsqrt):detsqrt);
}

vec3 Sphere_getNormal(in Sphere sphere, in vec3 position)
{
    const float radiusRcp = inversesqrt(sphere.radius2);
    return (position-sphere.position)*radiusRcp;
}

float Sphere_getSolidAngle_impl(in float cosThetaMax)
{
    return 2.0*nbl_glsl_PI*(1.0-cosThetaMax);
}
float Sphere_getSolidAngle(in Sphere sphere, in vec3 origin)
{
    float cosThetaMax = sqrt(1.0-sphere.radius2/nbl_glsl_lengthSq(sphere.position-origin));
    return Sphere_getSolidAngle_impl(cosThetaMax);
}

struct Triangle
{
    vec3 vertex0;
    uint bsdfLightIDs;
    vec3 vertex1;
    uint padding0;
    vec3 vertex2;
    uint padding1;
};

Triangle Triangle_Triangle(in mat3 vertices, in uint bsdfID, in uint lightID)
{
    Triangle tri;
    tri.vertex0 = vertices[0];
    tri.vertex1 = vertices[1];
    tri.vertex2 = vertices[2];
    //
    tri.bsdfLightIDs = bitfieldInsert(bsdfID, lightID, 16, 16);
    return tri;
}

// return intersection distance if found, FLT_NAN otherwise
float Triangle_intersect(in Triangle tri, in vec3 origin, in vec3 direction)
{
    const vec3 edges[2] = vec3[2](tri.vertex1-tri.vertex0,tri.vertex2-tri.vertex0);

    const vec3 h = cross(direction,edges[1]);
    const float a = dot(edges[0],h);

    const vec3 relOrigin = origin-tri.vertex0;

    const float u = dot(relOrigin,h)/a;

    const vec3 q = cross(relOrigin,edges[0]);
    const float v = dot(direction,q)/a;

    const float t = dot(edges[1],q)/a;

    return t>0.f&&u>=0.f&&v>=0.f&&(u+v)<=1.f ? t:nbl_glsl_FLT_NAN;
}

vec3 Triangle_getNormalTimesArea_impl(in mat2x3 edges)
{
    return cross(edges[0],edges[1])*0.5;
}
vec3 Triangle_getNormalTimesArea(in Triangle tri)
{
    return Triangle_getNormalTimesArea_impl(mat2x3(tri.vertex1-tri.vertex0,tri.vertex2-tri.vertex0));
}



struct Rectangle
{
    vec3 offset;
    uint bsdfLightIDs;
    vec3 edge0;
    uint padding0;
    vec3 edge1;
    uint padding1;
};

Rectangle Rectangle_Rectangle(in vec3 offset, in vec3 edge0, in vec3 edge1, in uint bsdfID, in uint lightID)
{
    Rectangle rect;
    rect.offset = offset;
    rect.edge0 = edge0;
    rect.edge1 = edge1;
    //
    rect.bsdfLightIDs = bitfieldInsert(bsdfID, lightID, 16, 16);
    return rect;
}

// return intersection distance if found, FLT_NAN otherwise
float Rectangle_intersect(in Rectangle rect, in vec3 origin, in vec3 direction)
{
    const vec3 h = cross(direction,rect.edge1);
    const float a = dot(rect.edge0,h);

    const vec3 relOrigin = origin-rect.offset;

    const float u = dot(relOrigin,h)/a;

    const vec3 q = cross(relOrigin,rect.edge0);
    const float v = dot(direction,q)/a;

    const float t = dot(rect.edge1,q)/a;

    const bool intersection = t>0.f&&u>=0.f&&v>=0.f&&u<=1.f&&v<=1.f;
    return intersection ? t:nbl_glsl_FLT_NAN;
}

vec3 Rectangle_getNormalTimesArea(in Rectangle rect)
{
    return cross(rect.edge0,rect.edge1);
}



#define DIFFUSE_OP 0u
#define CONDUCTOR_OP 1u
#define DIELECTRIC_OP 2u
#define OP_BITS_OFFSET 0
#define OP_BITS_SIZE 2
struct BSDFNode
{ 
    uvec4 data[2];
};

uint BSDFNode_getType(in BSDFNode node)
{
    return bitfieldExtract(node.data[0].w,OP_BITS_OFFSET,OP_BITS_SIZE);
}
bool BSDFNode_isBSDF(in BSDFNode node)
{
    return BSDFNode_getType(node)==DIELECTRIC_OP;
}
bool BSDFNode_isNotDiffuse(in BSDFNode node)
{
    return BSDFNode_getType(node)!=DIFFUSE_OP;
}
float BSDFNode_getRoughness(in BSDFNode node)
{
    return uintBitsToFloat(node.data[1].w);
}
vec3 BSDFNode_getRealEta(in BSDFNode node)
{
    return uintBitsToFloat(node.data[0].rgb);
}
vec3 BSDFNode_getImaginaryEta(in BSDFNode node)
{
    return uintBitsToFloat(node.data[1].rgb);
}
mat2x3 BSDFNode_getEta(in BSDFNode node)
{
    return mat2x3(BSDFNode_getRealEta(node),BSDFNode_getImaginaryEta(node));
}
#include <nbl/builtin/glsl/bxdf/fresnel.glsl>
vec3 BSDFNode_getReflectance(in BSDFNode node, in float VdotH)
{
    const vec3 albedoOrRealIoR = uintBitsToFloat(node.data[0].rgb);
    if (BSDFNode_isNotDiffuse(node))
        return nbl_glsl_fresnel_conductor(albedoOrRealIoR, BSDFNode_getImaginaryEta(node), VdotH);
    else
        return albedoOrRealIoR;
}

float BSDFNode_getNEEProb(in BSDFNode bsdf)
{
    const float alpha = BSDFNode_isNotDiffuse(bsdf) ? BSDFNode_getRoughness(bsdf):1.0;
    return min(8.0*alpha,1.0);
}

#include <nbl/builtin/glsl/colorspace/EOTF.glsl>
#include <nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl>
float getLuma(in vec3 col)
{
    return dot(transpose(nbl_glsl_scRGBtoXYZ)[1],col);
}

#define BSDF_COUNT 7
BSDFNode bsdfs[BSDF_COUNT] = {
    {{uvec4(floatBitsToUint(vec3(0.8,0.8,0.8)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(0.8,0.4,0.4)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(0.4,0.8,0.4)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.02,1.3)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,1.0,2.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.3,1.02)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,2.0,1.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.3,1.02)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,2.0,1.0,0.15))}},
    {{uvec4(floatBitsToUint(vec3(1.4,1.45,1.5)),DIELECTRIC_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0625))}}
};


struct Light
{
    vec3 radiance;
    uint objectID;
};

vec3 Light_getRadiance(in Light light)
{
    return light.radiance;
}
uint Light_getObjectID(in Light light)
{
    return light.objectID;
}


#define LIGHT_COUNT 1
float scene_getLightChoicePdf(in Light light)
{
    return 1.0/float(LIGHT_COUNT);
}


#define LIGHT_COUNT 1
Light lights[LIGHT_COUNT] =
{
    {
        vec3(30.0,25.0,15.0),
#ifdef POLYGON_METHOD
        0u
#else
        8u
#endif
    }
};



#define ANY_HIT_FLAG (-2147483648)
#define DEPTH_BITS_COUNT 8
#define DEPTH_BITS_OFFSET (31-DEPTH_BITS_COUNT)
struct ImmutableRay_t
{
    vec3 origin;
    vec3 direction;
#if POLYGON_METHOD==2
    vec3 normalAtOrigin;
    bool wasBSDFAtOrigin;
#endif
};
struct MutableRay_t
{
    float intersectionT;
    uint objectID;
    /* irrelevant here
    uint triangleID;
    vec2 barycentrics;
    */
};
struct Payload_t
{
    vec3 accumulation;
    float otherTechniqueHeuristic;
    vec3 throughput;
    #ifdef KILL_DIFFUSE_SPECULAR_PATHS
    bool hasDiffuse;
    #endif
};

struct Ray_t
{
    ImmutableRay_t _immutable;
    MutableRay_t _mutable;
    Payload_t _payload;
};


#define INTERSECTION_ERROR_BOUND_LOG2 (-8.0)
float getTolerance_common(in uint depth)
{
    float depthRcp = 1.0/float(depth);
    return INTERSECTION_ERROR_BOUND_LOG2;// *depthRcp*depthRcp;
}
float getStartTolerance(in uint depth)
{
    return exp2(getTolerance_common(depth));
}
float getEndTolerance(in uint depth)
{
    return 1.0-exp2(getTolerance_common(depth)+1.0);
}


vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= nbl_glsl_RECIPROCAL_PI*0.5;
    uv += 0.5; 
    return uv;
}

void missProgram(in ImmutableRay_t _immutable, inout Payload_t _payload)
{
    vec3 finalContribution = _payload.throughput; 
    // #define USE_ENVMAP
#ifdef USE_ENVMAP
	vec2 uv = SampleSphericalMap(_immutable.direction);
    finalContribution *= textureLod(envMap, uv, 0.0).rgb;
#else
    const vec3 kConstantEnvLightRadiance = vec3(0.15, 0.21, 0.3);
    finalContribution *= kConstantEnvLightRadiance;
#endif
    _payload.accumulation += finalContribution;
}

#include <nbl/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/beckmann.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <nbl/builtin/glsl/bxdf/bsdf/diffuse/lambert.glsl>
#include <nbl/builtin/glsl/bxdf/bsdf/specular/dielectric.glsl>
#include <nbl/builtin/glsl/bxdf/bsdf/specular/beckmann.glsl>
#include <nbl/builtin/glsl/bxdf/bsdf/specular/ggx.glsl>
nbl_glsl_LightSample nbl_glsl_bsdf_cos_generate(in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u, in BSDFNode bsdf, in float monochromeEta, out nbl_glsl_AnisotropicMicrofacetCache _cache)
{
    const float a = BSDFNode_getRoughness(bsdf);
    const mat2x3 ior = BSDFNode_getEta(bsdf);
    
    // fresnel stuff for dielectrics
    float orientedEta, rcpOrientedEta;
    const bool viewerInsideMedium = nbl_glsl_getOrientedEtas(orientedEta,rcpOrientedEta,interaction.isotropic.NdotV,monochromeEta);

    nbl_glsl_LightSample smpl;
    nbl_glsl_AnisotropicMicrofacetCache dummy;
    switch (BSDFNode_getType(bsdf))
    {
        case DIFFUSE_OP:
            smpl = nbl_glsl_oren_nayar_cos_generate(interaction,u.xy,a*a);
            break;
        case CONDUCTOR_OP:
            smpl = nbl_glsl_ggx_cos_generate(interaction,u.xy,a,a,_cache);
            break;
        default:
            smpl = nbl_glsl_ggx_dielectric_cos_generate(interaction,u,a,a,monochromeEta,_cache);
            break;
    }
    return smpl;
}

vec3 nbl_glsl_bsdf_cos_remainder_and_pdf(out float pdf, in nbl_glsl_LightSample _sample, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in BSDFNode bsdf, in float monochromeEta, in nbl_glsl_AnisotropicMicrofacetCache _cache)
{
    // are V and L on opposite sides of the surface?
    const bool transmitted = nbl_glsl_isTransmissionPath(interaction.isotropic.NdotV,_sample.NdotL);

    // is the BSDF or BRDF, if it is then we make the dot products `abs` before `max(,0.0)`
    const bool transmissive = BSDFNode_isBSDF(bsdf);
    const float clampedNdotL = nbl_glsl_conditionalAbsOrMax(transmissive,_sample.NdotL,0.0);
    const float clampedNdotV = nbl_glsl_conditionalAbsOrMax(transmissive,interaction.isotropic.NdotV,0.0);

    vec3 remainder;

    const float minimumProjVectorLen = 0.00000001;
    if (clampedNdotV>minimumProjVectorLen && clampedNdotL>minimumProjVectorLen)
    {
        // fresnel stuff for conductors (but reflectance also doubles as albedo)
        const mat2x3 ior = BSDFNode_getEta(bsdf);
        const vec3 reflectance = BSDFNode_getReflectance(bsdf,_cache.isotropic.VdotH);

        // fresnel stuff for dielectrics
        float orientedEta, rcpOrientedEta;
        const bool viewerInsideMedium = nbl_glsl_getOrientedEtas(orientedEta,rcpOrientedEta,interaction.isotropic.NdotV,monochromeEta);

        //
        const float VdotL = dot(interaction.isotropic.V.dir,_sample.L);

        //
        const float a = max(BSDFNode_getRoughness(bsdf),0.0001); // TODO: @Crisspl 0-roughness still doesn't work! Also Beckmann has a weird dark rim instead as fresnel!?
        const float a2 = a*a;

        // TODO: refactor into Material Compiler-esque thing
        switch (BSDFNode_getType(bsdf))
        {
            case DIFFUSE_OP:
                remainder = reflectance*nbl_glsl_oren_nayar_cos_remainder_and_pdf_wo_clamps(pdf,a*a,VdotL,clampedNdotL,clampedNdotV);
                break;
            case CONDUCTOR_OP:
                remainder = nbl_glsl_ggx_cos_remainder_and_pdf_wo_clamps(pdf,nbl_glsl_ggx_trowbridge_reitz(a2,_cache.isotropic.NdotH2),clampedNdotL,_sample.NdotL2,clampedNdotV,interaction.isotropic.NdotV_squared,reflectance,a2);
                break;
            default:
                remainder = vec3(nbl_glsl_ggx_dielectric_cos_remainder_and_pdf(pdf, _sample, interaction.isotropic, _cache.isotropic, monochromeEta, a*a));
                break;
        }
    }
    else
        remainder = vec3(0.0);
    return remainder;
}

layout (constant_id = 0) const int MAX_DEPTH_LOG2 = 4;
layout (constant_id = 1) const int MAX_SAMPLES_LOG2 = 10;


#include <nbl/builtin/glsl/random/xoroshiro.glsl>

mat2x3 rand3d(in uint protoDimension, in uint _sample, inout nbl_glsl_xoroshiro64star_state_t scramble_state)
{
    mat2x3 retval;
    uint address = bitfieldInsert(protoDimension,_sample,MAX_DEPTH_LOG2,MAX_SAMPLES_LOG2);
    for (int i=0; i<2u; i++)
    {
	    uvec3 seqVal = texelFetch(sampleSequence,int(address)+i).xyz;
	    seqVal ^= uvec3(nbl_glsl_xoroshiro64star(scramble_state),nbl_glsl_xoroshiro64star(scramble_state),nbl_glsl_xoroshiro64star(scramble_state));
        retval[i] = vec3(seqVal)*uintBitsToFloat(0x2f800004u);
    }
    return retval;
}


void traceRay_extraShape(inout int objectID, inout float intersectionT, in vec3 origin, in vec3 direction);
int traceRay(inout float intersectionT, in vec3 origin, in vec3 direction)
{
	int objectID = -1;
    
#define USE_RAY_QUERY
#ifdef USE_RAY_QUERY
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsNoneEXT, 0xFF, origin, 0.0, direction, 1000.0);
    
    // Start traversal: return false if traversal is complete
    while(rayQueryProceedEXT(rayQuery))
    {
        if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionAABBEXT)
        {
            int id = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);
            float t = Sphere_intersect(spheres[id],origin,direction);
            bool reportIntersection = (t != nbl_glsl_FLT_NAN && t > 0 && t < intersectionT);
            if(reportIntersection)
            {
                intersectionT = t;
                objectID = id;
                rayQueryGenerateIntersectionEXT(rayQuery, t);
            }
        }
    }
#else
	for (int i=0; i<SPHERE_COUNT; i++)
    {
        float t = Sphere_intersect(spheres[i],origin,direction);
        bool closerIntersection = t>0.0 && t<intersectionT;

        intersectionT = closerIntersection ? t : intersectionT;
		objectID = closerIntersection ? i:objectID;
    }
#endif

    traceRay_extraShape(objectID,intersectionT,origin,direction);
    return objectID;
}

//
float nbl_glsl_light_deferred_pdf(in Light light, in Ray_t ray);
vec3 nbl_glsl_light_deferred_eval_and_prob(out float pdf, in Light light, in Ray_t ray)
{
    // we don't have to worry about solid angle of the light w.r.t. surface of the light because this function only ever gets called from closestHit routine, so such ray cannot be produced (because lights have no BSDFs here)
    pdf = scene_getLightChoicePdf(light);
    pdf *= nbl_glsl_light_deferred_pdf(light,ray);
    return Light_getRadiance(light);
}

vec3 nbl_glsl_light_generate_and_pdf(out float pdf, out float newRayMaxT, in vec3 origin, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in bool isBSDF, in vec3 xi, in uint objectID);
nbl_glsl_LightSample nbl_glsl_light_generate_and_remainder_and_pdf(out vec3 remainder, out float pdf, out float newRayMaxT, in vec3 origin, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in bool isBSDF, in vec3 xi, in uint depth)
{
    // normally we'd pick from set of lights, using `xi.z`
    const Light light = lights[0];
    
    vec3 L = nbl_glsl_light_generate_and_pdf(pdf,newRayMaxT,origin,interaction,isBSDF,xi,Light_getObjectID(light));

    newRayMaxT *= getEndTolerance(depth);
    pdf *= scene_getLightChoicePdf(light);
    remainder = Light_getRadiance(light)/pdf;
    return nbl_glsl_createLightSample(L,interaction);
}

uint getBSDFLightIDAndDetermineNormal(out vec3 normal, in uint objectID, in vec3 intersection);
bool closestHitProgram(in uint depth, in uint _sample, inout Ray_t ray, inout nbl_glsl_xoroshiro64star_state_t scramble_state)
{
    const MutableRay_t _mutable = ray._mutable;
    const uint objectID = _mutable.objectID;

    // interaction stuffs
    const ImmutableRay_t _immutable = ray._immutable;
    const vec3 intersection = _immutable.origin+_immutable.direction*_mutable.intersectionT;

    uint bsdfLightIDs;
    nbl_glsl_AnisotropicViewSurfaceInteraction interaction;
    {
        nbl_glsl_IsotropicViewSurfaceInteraction isotropic;
        bsdfLightIDs = getBSDFLightIDAndDetermineNormal(isotropic.N,objectID,intersection);

        isotropic.V.dir = -_immutable.direction;
        isotropic.NdotV = dot(isotropic.V.dir,isotropic.N);
        isotropic.NdotV_squared = isotropic.NdotV*isotropic.NdotV;

        interaction = nbl_glsl_calcAnisotropicInteraction(isotropic);
    }

    //
    vec3 throughput = ray._payload.throughput;

    // add emissive and finish MIS
    const uint lightID = bitfieldExtract(bsdfLightIDs,16,16);
    if (lightID != INVALID_ID_16BIT) // has emissive
    {
        float lightPdf;
        ray._payload.accumulation += nbl_glsl_light_deferred_eval_and_prob(lightPdf,lights[lightID],ray)*throughput/(1.0+lightPdf*lightPdf*ray._payload.otherTechniqueHeuristic);
    }

    // check if we even have a BSDF at all
    uint bsdfID = bitfieldExtract(bsdfLightIDs, 0, 16);
    if (bsdfID != INVALID_ID_16BIT)
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
        const float neeProbability = BSDFNode_getNEEProb(bsdf);
        float rcpChoiceProb;
        if (!nbl_glsl_partitionRandVariable(neeProbability,epsilon[0].z,rcpChoiceProb))
        {
            vec3 neeContrib; float lightPdf, t;
            nbl_glsl_LightSample nee_sample = nbl_glsl_light_generate_and_remainder_and_pdf(
                neeContrib, lightPdf, t,
                intersection, interaction,
                isBSDF, epsilon[0], depth
            );
            // We don't allow non watertight transmitters in this renderer
            bool validPath = nee_sample.NdotL>0.0;
            // but if we allowed non-watertight transmitters (single water surface), it would make sense just to apply this line by itself
            nbl_glsl_AnisotropicMicrofacetCache _cache;
            validPath = validPath && nbl_glsl_calcAnisotropicMicrofacetCache(_cache, interaction, nee_sample, monochromeEta);
            if (validPath)
            {
                float bsdfPdf;
                neeContrib *= nbl_glsl_bsdf_cos_remainder_and_pdf(bsdfPdf,nee_sample,interaction,bsdf,monochromeEta,_cache)*throughput;
                const float oc = bsdfPdf*rcpChoiceProb;
                neeContrib /= 1.0/oc+oc/(lightPdf*lightPdf); // MIS weight
                if (bsdfPdf<nbl_glsl_FLT_MAX && getLuma(neeContrib)>lumaContributionThreshold && traceRay(t,intersection+nee_sample.L*t*getStartTolerance(depth),nee_sample.L)==-1)
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
            ray._payload.otherTechniqueHeuristic = neeProbability/bsdfPdf; // numerically stable, don't touch
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

void main()
{
    const ivec2 coords = getCoordinates();
    const vec2 texCoord = getTexCoords();

    if (false == (all(lessThanEqual(ivec2(0),coords)) && all(greaterThan(imageSize(outImage),coords)))) {
        return;
    }

    if (((MAX_DEPTH-1)>>MAX_DEPTH_LOG2)>0 || ((SAMPLES-1)>>MAX_SAMPLES_LOG2)>0)
    {
        vec4 pixelCol = vec4(1.0,0.0,0.0,1.0);
        imageStore(outImage, coords, pixelCol);
        return;
    }

	nbl_glsl_xoroshiro64star_state_t scramble_start_state = texelFetch(scramblebuf,coords,0).rg;
    const vec2 pixOffsetParam = vec2(1.0)/vec2(textureSize(scramblebuf,0));


    const mat4 invMVP = inverse(cameraData.params.MVP);
    
    vec4 NDC = vec4(texCoord*vec2(2.0,-2.0)+vec2(-1.0,1.0),0.0,1.0);
    vec3 camPos;
    {
        vec4 tmp = invMVP*NDC;
        camPos = tmp.xyz/tmp.w;
        NDC.z = 1.0;
    }

    vec3 color = vec3(0.0);
    float meanLumaSquared = 0.0;
    // TODO: if we collapse the nested for loop, then all GPUs will get `MAX_DEPTH` factor speedup, not just NV with separate PC
    for (int i=0; i<SAMPLES; i++)
    {
        nbl_glsl_xoroshiro64star_state_t scramble_state = scramble_start_state;

        Ray_t ray;
        // raygen
        {
            ray._immutable.origin = camPos;

            vec4 tmp = NDC;
            // apply stochastic reconstruction filter
            const float gaussianFilterCutoff = 2.5;
            const float truncation = exp(-0.5*gaussianFilterCutoff*gaussianFilterCutoff);
            vec2 remappedRand = rand3d(0u,i,scramble_state)[0].xy;
            remappedRand.x *= 1.0-truncation;
            remappedRand.x += truncation;
            tmp.xy += pixOffsetParam*nbl_glsl_BoxMullerTransform(remappedRand,1.5);
            // for depth of field we could do another stochastic point-pick
            tmp = invMVP*tmp;
            ray._immutable.direction = normalize(tmp.xyz/tmp.w-camPos);

            #if POLYGON_METHOD==2
                ray._immutable.normalAtOrigin = vec3(0.0,0.0,0.0);
                ray._immutable.wasBSDFAtOrigin = false;
            #endif

            ray._payload.accumulation = vec3(0.0);
            ray._payload.otherTechniqueHeuristic = 0.0; // needed for direct eye-light paths
            ray._payload.throughput = vec3(1.0);
            #ifdef KILL_DIFFUSE_SPECULAR_PATHS
            ray._payload.hasDiffuse = false;
            #endif
        }

        // bounces
        {
            bool hit = true; bool rayAlive = true;
            for (int d=1; d<=MAX_DEPTH && hit && rayAlive; d+=2)
            {
                ray._mutable.intersectionT = nbl_glsl_FLT_MAX;
                ray._mutable.objectID = traceRay(ray._mutable.intersectionT,ray._immutable.origin,ray._immutable.direction);
                hit = ray._mutable.objectID!=-1;
                if (hit)
                    rayAlive = closestHitProgram(3u, i, ray, scramble_state);
            }
            // was last trace a miss?
            if (!hit)
                missProgram(ray._immutable,ray._payload);
        }

        vec3 accumulation = ray._payload.accumulation;

        float rcpSampleSize = 1.0/float(i+1);
        color += (accumulation-color)*rcpSampleSize;
        
        #ifdef VISUALIZE_HIGH_VARIANCE
            float luma = getLuma(accumulation);
            meanLumaSquared += (luma*luma-meanLumaSquared)*rcpSampleSize;
        #endif
    }

    #ifdef VISUALIZE_HIGH_VARIANCE
        float variance = getLuma(color);
        variance *= variance;
        variance = meanLumaSquared-variance;
        if (variance>5.0)
            color = vec3(1.0,0.0,0.0);
    #endif

    vec4 pixelCol = vec4(color, 1.0);
    imageStore(outImage, coords, pixelCol);
}
/** TODO: Improving Rendering

Now:
- Always MIS (path correlated reuse)
- Test MIS alpha (roughness) scheme

Many Lights:
- Path Guiding
- Light Importance Lists/Classification
- Spatio-Temporal Reservoir Sampling

Indirect Light:
- Bidirectional Path Tracing
- Uniform Path Sampling / Vertex Connection and Merging / Path Space Regularization

Animations:
- A-SVGF / BMFR
**/