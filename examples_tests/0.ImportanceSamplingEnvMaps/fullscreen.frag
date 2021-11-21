#version 430 core

#extension GL_ARB_derivative_control : enable

layout(location = 0) in vec2 TexCoord;

layout (location = 0) out vec4 pixelColor;

// basic settings
#define MAX_DEPTH 15
#define SAMPLES 32
// #define IMPORTANCE_SAMPLING

layout(set = 3, binding = 0) uniform sampler2D envMap; 
layout(set = 3, binding = 1) uniform usamplerBuffer sampleSequence;
layout(set = 3, binding = 2) uniform usampler2D scramblebuf;
layout(set = 3, binding = 3) uniform sampler2D phiPdfLUT;
layout(set = 3, binding = 4) uniform sampler1D thetaLUT;

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/limits/numeric.glsl>

layout(set = 1, binding = 0, row_major, std140) uniform UBO
{
	nbl_glsl_SBasicViewParameters params;
} cameraData;

layout (push_constant) uniform PushConstants
{
    layout (offset = 0) float envmapNormalizationFactor;
} pc;

layout (constant_id = 0) const int MAX_DEPTH_LOG2 = 0;
layout (constant_id = 1) const int MAX_SAMPLES_LOG2 = 0;

#include <nbl/builtin/glsl/random/xoroshiro.glsl>
#include <nbl/builtin/glsl/sampling/box_muller_transform.glsl>
#include <nbl/builtin/glsl/limits/numeric.glsl>
#include <nbl/builtin/glsl/bxdf/common.glsl>
#include <nbl/builtin/glsl/colorspace/EOTF.glsl>
#include <nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl>
#include <nbl/builtin/glsl/math/functions.glsl>

#define INVALID_ID_16BIT 0xffffu

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

float getLuma(in vec3 col)
{
    return dot(transpose(nbl_glsl_scRGBtoXYZ)[1],col);
}

vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), acos(v.y));
    uv.x *= nbl_glsl_RECIPROCAL_PI*0.5;
    uv.x += 0.5; 
    uv.y *= nbl_glsl_RECIPROCAL_PI;
    return uv;
}

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

struct Sphere
{
    vec3 position;
    float radius2;
    uint bsdfLightIDs;
}; 

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

struct Light
{
    vec3 radiance;
    uint objectID;
};

vec3 Light_getRadiance(in Light light, in vec3 L)
{
    vec2 uv = SampleSphericalMap(L);
    return textureLod(envMap, uv, 0.0).rgb;
}

uint Light_getObjectID(in Light light)
{
    return light.objectID;
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
    return bitfieldExtract(node.data[0].w, OP_BITS_OFFSET, OP_BITS_SIZE);
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

float BSDFNode_getNEEProb(in BSDFNode bsdf)
{
    const float alpha = BSDFNode_isNotDiffuse(bsdf) ? BSDFNode_getRoughness(bsdf):1.0;
    return min(8.0*alpha,1.0);
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

// Create yo Scene

#define SPHERE_COUNT 8

Sphere spheres[SPHERE_COUNT] =
{
    Sphere_Sphere(vec3(0.0,-100.5,-1.0), 100.0, 0u, INVALID_ID_16BIT),
    Sphere_Sphere(vec3(2.0,0.0,-1.0),0.5,1u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(0.0,0.0,-1.0),0.5,2u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(-2.0,0.0,-1.0),0.5,3u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(2.0,0.0,1.0),0.5,4u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(0.0,0.0,1.0),0.5,4u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(-2.0,0.0,1.0),0.5,5u,INVALID_ID_16BIT),
    Sphere_Sphere( vec3(0.5,1.0,0.5), 0.5, 6u, INVALID_ID_16BIT)
};

#define LIGHT_COUNT 1
Light lights[LIGHT_COUNT] = { { vec3(30.0,25.0,15.0), 8u } }; // This doesn't matter if theres only a single env light in the scene

#define BSDF_COUNT 7
BSDFNode bsdfs[BSDF_COUNT] =
{
    {{uvec4(floatBitsToUint(vec3(0.8,0.8,0.8)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(0.8,0.4,0.4)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(0.4,0.8,0.4)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.02,1.3)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,1.0,2.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.3,1.02)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,2.0,1.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.3,1.02)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,2.0,1.0,0.15))}},
    {{uvec4(floatBitsToUint(vec3(1.4,1.45,1.5)),DIELECTRIC_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0625))}}
};

float scene_getLightChoicePdf(in Light light)
{
    return 1.0/float(LIGHT_COUNT);
}

uint getBSDFLightIDAndDetermineNormal(out vec3 normal, in uint objectID, in vec3 intersection)
{
    Sphere sphere = spheres[objectID];
    normal = Sphere_getNormal(sphere,intersection);
    return sphere.bsdfLightIDs;
}

float nbl_glsl_light_deferred_pdf(in Light light, in Ray_t ray)
{
#ifdef IMPORTANCE_SAMPLING
    const vec3 direction = ray._immutable.direction;
    const vec2 uv = SampleSphericalMap(direction);
    return (getLuma(textureLod(envMap, uv, 0.0).rgb))*(pc.envmapNormalizationFactor);
#else
    return 0.f;
#endif
}

vec3 nbl_glsl_light_deferred_eval_and_prob(out float pdf, in Light light, in Ray_t ray)
{
    // we don't have to worry about solid angle of the light w.r.t. surface of the light because this function only ever gets called from closestHit routine, so such ray cannot be produced (because lights have no BSDFs here)
    pdf = scene_getLightChoicePdf(light);
    pdf *= nbl_glsl_light_deferred_pdf(light, ray);
    return Light_getRadiance(light, normalize(ray._immutable.direction));
}

#ifdef IMPORTANCE_SAMPLING
vec3 nbl_glsl_light_generate_and_pdf(out float pdf, out float newRayMaxT, in vec3 origin, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in bool isBSDF, in vec3 xi, in uint objectID)
{   
    const vec2 phiPdf = texture(phiPdfLUT, xi.xy).xy;
    const float theta = texture(thetaLUT, xi.y).x;

    float sinPhi, cosPhi;
    nbl_glsl_sincos(phiPdf.x - nbl_glsl_PI, sinPhi, cosPhi);

    float sinTheta, cosTheta;
    nbl_glsl_sincos(theta, sinTheta, cosTheta);

    vec3 L = vec3(sinTheta*cosPhi, cosTheta, sinTheta*sinPhi);

    pdf = phiPdf.y;
    newRayMaxT = nbl_glsl_FLT_MAX;
    return L;
}

nbl_glsl_LightSample nbl_glsl_light_generate_and_remainder_and_pdf(out vec3 remainder, out float pdf, out float newRayMaxT, in vec3 origin, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in bool isBSDF, in vec3 xi, in uint depth)
{
    // normally we'd pick from set of lights, using `xi.z`
    const Light light = lights[0]; // unused for env light
    
    vec3 L = nbl_glsl_light_generate_and_pdf(pdf,newRayMaxT,origin,interaction,isBSDF,xi,Light_getObjectID(light));

    newRayMaxT *= getEndTolerance(depth);
    pdf *= scene_getLightChoicePdf(light);
    remainder = Light_getRadiance(light, L)/pdf;
    return nbl_glsl_createLightSample(L,interaction);
}
#endif

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

int traceRay(inout float intersectionT, in vec3 origin, in vec3 direction)
{
    const bool anyHit = intersectionT != nbl_glsl_FLT_MAX;

	int objectID = -1;
	for (int i=0; i<SPHERE_COUNT; i++)
    {
        float t = Sphere_intersect(spheres[i],origin,direction);
        bool closerIntersection = t>0.0 && t<intersectionT;

        intersectionT = closerIntersection ? t : intersectionT;
		objectID = closerIntersection ? i:objectID;
        
        // allowing early out results in a performance regression, WTF!?
        //if (anyHit && closerIntersection)
           //break;
    }

    return objectID;
}

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

    vec3 throughput = ray._payload.throughput;

    // add emissive and finish MIS
    const uint lightID = bitfieldExtract(bsdfLightIDs,16,16);
    if (lightID != INVALID_ID_16BIT) // has emissive
    {
        float lightPdf;
#ifdef IMPORTANCE_SAMPLING
        ray._payload.accumulation += nbl_glsl_light_deferred_eval_and_prob(lightPdf, lights[lightID], ray) * throughput/(1.0 + lightPdf*lightPdf*ray._payload.otherTechniqueHeuristic);
#else
        ray._payload.accumulation += nbl_glsl_light_deferred_eval_and_prob(lightPdf, lights[lightID], ray) * throughput;
#endif
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
#ifdef IMPORTANCE_SAMPLING
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
#endif

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
#ifdef IMPORTANCE_SAMPLING
            ray._payload.otherTechniqueHeuristic = neeProbability/bsdfPdf; // numerically stable, don't touch
            ray._payload.otherTechniqueHeuristic *= ray._payload.otherTechniqueHeuristic;
#endif
                    
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


void missProgram(inout Ray_t ray)
{
    float lightPdf;
    vec3 finalContribution = nbl_glsl_light_deferred_eval_and_prob(lightPdf, lights[0], ray);
    finalContribution *= ray._payload.throughput/(1.0+lightPdf*lightPdf*ray._payload.otherTechniqueHeuristic);

    ray._payload.accumulation += finalContribution;
}

void main()
{
    // Keep it around for devsh's testing experiments
#if 0
    {
        vec3 center = imageLoad(sphericalCoordLUT, ivec2(gl_FragCoord.xy)+ivec2(0,0)).xyz;
        float up = imageLoad(sphericalCoordLUT, ivec2(gl_FragCoord.xy)+ivec2(0,1)).y;
        float right = imageLoad(sphericalCoordLUT, ivec2(gl_FragCoord.xy)+ivec2(1,0)).x;


        float dThetadXi2 = (up-center.y)*1024.f;
        float dPhiXi1 = (right-center.x)*2048.f;
        float error = abs(1.f-abs(center.z*(dThetadXi2*dPhiXi1)));

        pixelColor = vec4(vec3(error)*0.5f, 1.0); // 0.5 so that mid-gray shows up

        return;
    }
#endif

	if (((MAX_DEPTH-1)>>MAX_DEPTH_LOG2)>0 || ((SAMPLES-1)>>MAX_SAMPLES_LOG2)>0)
    {
        pixelColor = vec4(1.0,0.0,0.0,1.0);
        return;
    }

	nbl_glsl_xoroshiro64star_state_t scramble_start_state = textureLod(scramblebuf, TexCoord, 0).rg;
    const vec2 pixOffsetParam = vec2(1.0)/vec2(textureSize(scramblebuf,0));

    const mat4 invMVP = inverse(cameraData.params.MVP);

    vec4 NDC = vec4(TexCoord*vec2(2.0,-2.0) + vec2(-1.0,1.0), 0.0, 1.0);
    vec3 camPos;
    {
        vec4 tmp = invMVP*NDC;
        camPos = tmp.xyz/tmp.w;
        NDC.z = 1.0;
    }

	vec3 color = vec3(0.0);
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
            vec2 remappedRand = rand3d(0u, i, scramble_state)[0].xy;
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
                missProgram(ray);
        }

        vec3 accumulation = ray._payload.accumulation;

        float rcpSampleSize = 1.0/float(i+1);
        color += (accumulation-color)*rcpSampleSize;
	}

	pixelColor = vec4(color, 1.0);
}