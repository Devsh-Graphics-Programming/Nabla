#version 430 core
#extension GL_GOOGLE_include_directive : require

layout(set = 3, binding = 0) uniform sampler2D envMap; 
layout(set = 3, binding = 1) uniform usamplerBuffer sampleSequence;
layout(set = 3, binding = 2) uniform usampler2D scramblebuf;

layout(location = 0) in vec2 TexCoord;

layout(location = 0) out vec4 pixelColor;



// TODO: @Crisspl change the location of the `irr_glsl_SBasicViewParameters`, vertex header is not the right place for it
#include <irr/builtin/glsl/utils/vertex.glsl>

layout(set = 1, binding = 0, row_major, std140) uniform UBO
{
	irr_glsl_SBasicViewParameters params;
} cameraData;

#define INVALID_ID_16BIT 0xffffu
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

float Sphere_getSolidAngle_impl(in float cosThetaMax)
{
    return 2.0*irr_glsl_PI*(1.0-cosThetaMax);
}
float Sphere_getSolidAngle(in Sphere sphere, in vec3 origin)
{
    float cosThetaMax = sqrt(1.0-sphere.radius2/irr_glsl_lengthSq(sphere.position-origin));
    return Sphere_getSolidAngle_impl(cosThetaMax);
}

#define SPHERE_COUNT 9
Sphere spheres[SPHERE_COUNT] = {
    Sphere_Sphere(vec3(0.0,-100.5,-1.0),100.0,0u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(2.0,0.0,-1.0),0.5,1u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(0.0,0.0,-1.0),0.5,2u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(-2.0,0.0,-1.0),0.5,3u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(2.0,0.0,1.0),0.5,4u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(0.0,0.0,1.0),0.5,4u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(-2.0,0.0,1.0),0.5,5u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(0.5,1.0,0.5),0.5,6u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(-1.5,1.5,0.0),0.3,7u,0u)
};


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
bool BSDFNode_isNotDiffuse(in BSDFNode node)
{
    return BSDFNode_getType(node)!=DIFFUSE_OP;
}
float BSDFNode_getRoughness(in BSDFNode node)
{
    return uintBitsToFloat(node.data[1].w);
}
vec3 BSDFNode_getReflectance(in BSDFNode node)
{
    return uintBitsToFloat(node.data[0].rgb);
}
vec3 BSDFNode_getRealEta(in BSDFNode node)
{
    return uintBitsToFloat(node.data[0].rgb);
}
vec3 BSDFNode_getImaginaryEta(in BSDFNode node)
{
    return uintBitsToFloat(node.data[1].rgb);
}

float BSDFNode_getMISWeight(in BSDFNode bsdf)
{
    const float alpha = BSDFNode_getRoughness(bsdf);
    const bool notDiffuse = BSDFNode_isNotDiffuse(bsdf);
    const float DIFFUSE_MIS_WEIGHT = 0.5;
    return notDiffuse ? mix(uintBitsToFloat(0x3f800001u),DIFFUSE_MIS_WEIGHT,alpha):DIFFUSE_MIS_WEIGHT; // TODO: test alpha*alpha
}

#include <irr/builtin/glsl/colorspace/EOTF.glsl>
#include <irr/builtin/glsl/colorspace/encodeCIEXYZ.glsl>

#define BSDF_COUNT 8
BSDFNode bsdfs[BSDF_COUNT] = {
    {{uvec4(floatBitsToUint(vec3(0.8,0.8,0.8)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(0.8,0.4,0.4)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(0.4,0.8,0.4)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.02,1.3)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,1.0,2.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.3,1.02)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,2.0,1.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.3,1.02)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,2.0,1.0,0.15))}},
    {{uvec4(floatBitsToUint(vec3(1.5,1.5,1.5)),DIELECTRIC_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(0.8,0.6,0.2)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}}
};


struct Light
{
    vec3 radiance;
    uint objectID;
};

#define LIGHT_COUNT 1
Light lights[LIGHT_COUNT] = {
    {vec3(30.0,25.0,15.0),8u}
};
vec3 Light_getRadiance(in Light light)
{
    return light.radiance;
}
uint Light_getObjectID(in Light light)
{
    return light.objectID;
}


float scene_getLightChoicePdf(in Light light)
{
    return 1.0/float(LIGHT_COUNT);
}


#define ANY_HIT_FLAG (-2147483648)
#define DEPTH_BITS_COUNT 8
#define DEPTH_BITS_OFFSET (31-DEPTH_BITS_COUNT)
struct ImmutableRay_t
{
    vec3 origin;
    float maxT;
    vec3 direction;
    int typeDepthSampleIx;
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
    vec3 throughput;
    float otherTechniqueHeuristic;
};
struct Ray_t
{
    ImmutableRay_t _immutable;
    MutableRay_t _mutable;
    Payload_t _payload;
};
#define MAX_STACK_SIZE 1
int stackPtr = 0;
Ray_t rayStack[MAX_STACK_SIZE];


bool anyHitProgram(in ImmutableRay_t _immutable)
{
    return true;
}

#define INTERSECTION_ERROR_BOUND 0.00001
bool traceRay(in ImmutableRay_t _immutable)
{
    const bool anyHit = bitfieldExtract(_immutable.typeDepthSampleIx,31,1)!=0;

	int objectID = -1;
    float intersectionT = _immutable.maxT;
	for (int i=0; i<SPHERE_COUNT; i++)
    {
        vec3 origin = _immutable.origin-spheres[i].position;
        float originLen2 = dot(origin,origin);

        float dirDotOrigin = dot(_immutable.direction,origin);
        float det = spheres[i].radius2-originLen2+dirDotOrigin*dirDotOrigin;

        // do some speculative math here
        float t = -dirDotOrigin-sqrt(det);
        bool closerIntersection = t>0.0 && t<intersectionT;

		objectID = closerIntersection ? i:objectID;
        intersectionT = closerIntersection ? t:intersectionT;
        
        // allowing early out results in a performance regression, WTF!?
        //if (anyHit && closerIntersection && anyHitProgram(_immutable))
           //break;
    }
    rayStack[stackPtr]._mutable.objectID = objectID;
    rayStack[stackPtr]._mutable.intersectionT = intersectionT;
    // hit
    return anyHit;
}


#include <irr/builtin/glsl/math/constants.glsl>
vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= irr_glsl_RECIPROCAL_PI*0.5;
    uv += 0.5; 
    return uv;
}

void missProgram() 
{
    vec3 finalContribution = rayStack[stackPtr]._payload.throughput; 
    //#define USE_ENVMAP
    // true miss
    if (rayStack[stackPtr]._immutable.maxT>=FLT_MAX)
    {
        #ifdef USE_ENVMAP
	        vec2 uv = SampleSphericalMap(rayStack[stackPtr]._immutable.direction);
            finalContribution *= textureLod(envMap, uv, 0.0).rgb;
        #else
            const vec3 kConstantEnvLightRadiance = vec3(0.15,0.21,0.3);
            finalContribution *= kConstantEnvLightRadiance;
        #endif
    }
    else
    {
        finalContribution *= rayStack[stackPtr]._payload.otherTechniqueHeuristic;
    }
    rayStack[stackPtr]._payload.accumulation += finalContribution;
}


#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>
irr_glsl_BSDFSample irr_glsl_bsdf_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u, in BSDFNode bsdf)
{
    irr_glsl_BSDFSample smpl;
    switch (BSDFNode_getType(bsdf))
    {
        case DIFFUSE_OP:
            smpl = irr_glsl_lambertian_cos_generate(interaction,u.xy);
            break;
        case CONDUCTOR_OP:
            smpl = irr_glsl_reflection_cos_generate(interaction);
            break;
        default: // TODO: for dielectric
            smpl = irr_glsl_reflection_cos_generate(interaction);
            break;
    }
    return smpl;
}
vec3 irr_glsl_bsdf_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample _sample, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in BSDFNode bsdf)
{
    vec3 remainder;
    switch (BSDFNode_getType(bsdf))
    {
        case DIFFUSE_OP:
            remainder = vec3(irr_glsl_lambertian_cos_remainder_and_pdf(pdf,_sample,interaction));
            remainder *= BSDFNode_getReflectance(bsdf);
            break;
        case CONDUCTOR_OP:
            //remainder = irr_glsl_reflection_cos_remainder_and_pdf(pdf,_sample);
            pdf = _sample.NdotH==1.0 ? (1.0/0.0):0.0;
            remainder = normalize(BSDFNode_getReflectance(bsdf));
            break;
        default: // TODO: for dielectric
            pdf = _sample.NdotH==1.0 ? (1.0/0.0):0.0;
            remainder = normalize(BSDFNode_getReflectance(bsdf));
            break;
    }
    return remainder;
}

// the interaction here is the interaction at the illuminator-end of the ray, not the receiver
vec3 irr_glsl_light_deferred_eval_and_prob(out float pdf, in Sphere sphere, in vec3 origin, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in Light light)
{
    // we don't have to worry about solid angle of the light w.r.t. surface of the light because this function only ever gets called from closestHit routine, so such ray cannot be produced
    pdf = scene_getLightChoicePdf(light)/Sphere_getSolidAngle(sphere,origin);
    return Light_getRadiance(light);
}

#define GeneratorSample irr_glsl_BSDFSample
#define irr_glsl_LightSample irr_glsl_BSDFSample
irr_glsl_LightSample irr_glsl_createLightSample(in vec3 L, in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BSDFSample s;
    s.L = L;

    // Fix this @Crisspl after making the API consistent
    s.LdotT = dot(interaction.T,L);
    s.LdotB = dot(interaction.B,L);
    s.LdotN = dot(interaction.isotropic.N,L);
   
    float VdotL = dot(interaction.isotropic.V.dir,L);
    float LplusV_rcpLen = inversesqrt(2.0+2.0*VdotL);

    s.TdotH = (interaction.TdotV+s.LdotT)*LplusV_rcpLen;
    s.BdotH = (interaction.BdotV+s.LdotB)*LplusV_rcpLen;
    s.NdotH = (interaction.isotropic.NdotV+s.LdotN)*LplusV_rcpLen;

    s.VdotH = LplusV_rcpLen+LplusV_rcpLen*VdotL;
    
    return s;
}
// TODO: move this to header and optimize @Crisspl
vec2 irr_glsl_sincos(in float theta)
{ 
    float sinTheta = sin(theta);
    return vec2(sinTheta,cos(theta));
}
irr_glsl_LightSample irr_glsl_light_generate_and_remainder_and_pdf(out vec3 remainder, out float pdf, out float newRayMaxT, in vec3 origin, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u)
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
    const vec2 sincosPhi = irr_glsl_sincos(2.0*irr_glsl_PI*u.y); // TODO: @Crisspl random number may have to be in the -PI,PI range instead
    mat2x3 XY = irr_glsl_frisvad(Z);
    
    L += (XY[0]*sincosPhi.x+XY[1]*sincosPhi.y)*sinTheta;
    
    const float rcpPdf = Sphere_getSolidAngle_impl(cosThetaMax)/choicePdf;
    remainder = Light_getRadiance(light)*(possibilityOfLightVisibility ? rcpPdf:0.0); // this ternary operator kills invalid rays
    pdf = 1.0/rcpPdf;
    
    newRayMaxT = (cosTheta-sqrt(cosTheta2-cosThetaMax2))/rcpDistance*(1.0-INTERSECTION_ERROR_BOUND*2.0);
    
    return irr_glsl_createLightSample(L,interaction);
}


layout (constant_id = 0) const int MAX_DEPTH_LOG2 = 0;
layout (constant_id = 1) const int MAX_SAMPLES_LOG2 = 0;
#define MAX_DEPTH 8
#define SAMPLES 32

// TODO: upgrade the xorshift variant to one that uses an addition
uint rand_xorshift(inout uint rng_state)
{
    // Xorshift algorithm from George Marsaglia's paper
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}
vec3 rand3d(in uint protoDimension, in uint _sample, inout uint scramble_state)
{
    uint address = bitfieldInsert(protoDimension,_sample,MAX_DEPTH_LOG2,MAX_SAMPLES_LOG2);
	uvec3 seqVal = texelFetch(sampleSequence,int(address)).xyz;
	seqVal ^= uvec3(rand_xorshift(scramble_state),rand_xorshift(scramble_state),rand_xorshift(scramble_state));
    return vec3(seqVal)*uintBitsToFloat(0x2f800004u);
}

void closestHitProgram(in ImmutableRay_t _immutable, inout uint scramble_state)
{
    const MutableRay_t mutable = rayStack[stackPtr]._mutable;

    Sphere sphere = spheres[mutable.objectID];
    vec3 intersection = _immutable.origin+_immutable.direction*mutable.intersectionT;
    
    irr_glsl_AnisotropicViewSurfaceInteraction interaction;
    {
        irr_glsl_IsotropicViewSurfaceInteraction isotropic;

        isotropic.V.dir = -_immutable.direction;
        //isotropic.V.dPosdScreen = screw that
        isotropic.N = (intersection-sphere.position)*inversesqrt(sphere.radius2);
        isotropic.NdotV = dot(isotropic.V.dir,isotropic.N);
        isotropic.NdotV_squared = isotropic.NdotV*isotropic.NdotV;

        interaction = irr_glsl_calcAnisotropicInteraction(isotropic);
    }

    const uint bsdfLightIDs = sphere.bsdfLightIDs;
    const uint lightID = bitfieldExtract(bsdfLightIDs,16,16);

    vec3 throughput = rayStack[stackPtr]._payload.throughput;
    
    // finish MIS
    if (lightID!=INVALID_ID_16BIT) // has emissive
    {
        float lightPdf;
        vec3 lightVal = irr_glsl_light_deferred_eval_and_prob(lightPdf,sphere,_immutable.origin,interaction,lights[lightID]);
        rayStack[stackPtr]._payload.accumulation += throughput*lightVal/(1.0+lightPdf*lightPdf*rayStack[stackPtr]._payload.otherTechniqueHeuristic);
    }
    
    const int sampleIx = bitfieldExtract(_immutable.typeDepthSampleIx,0,DEPTH_BITS_OFFSET);
    const int depth = bitfieldExtract(_immutable.typeDepthSampleIx,DEPTH_BITS_OFFSET,DEPTH_BITS_COUNT);

    // check if we even have a BSDF at all
    uint bsdfID = bitfieldExtract(bsdfLightIDs,0,16);
    if (depth<MAX_DEPTH && bsdfID!=INVALID_ID_16BIT)
    {
        // common preload
        BSDFNode bsdf = bsdfs[bsdfID];
        uint opType = BSDFNode_getType(bsdf);

        float bsdfGeneratorProbability = BSDFNode_getMISWeight(bsdf);
        
        // this could run in a loop if we'd allow splitting/amplification (if we modified the payload managment)
        {
            vec3 epsilon = rand3d(depth,sampleIx,scramble_state);


            const bool pickBSDF = epsilon.z<bsdfGeneratorProbability;
            const float choiceProb = pickBSDF ? bsdfGeneratorProbability:(1.0-bsdfGeneratorProbability); // TODO: proper computation
            epsilon.z -= pickBSDF ? 0.0:bsdfGeneratorProbability; // TODO: do it properly
            const float rcpChoiceProb = 1.0/choiceProb; // should be impossible to get a div by 0 here
            epsilon.z *= rcpChoiceProb;
            

            float maxT;
            // the probability of generating a sample w.r.t. the light generator only possible and used when it was generated with it!
            float lightPdf;
            GeneratorSample _sample;
            if (pickBSDF)
            {
                 maxT = FLT_MAX;
                _sample = irr_glsl_bsdf_cos_generate(interaction,epsilon,bsdf);
            }
            else
            {
                vec3 lightRemainder;
                _sample = irr_glsl_light_generate_and_remainder_and_pdf(
                    lightRemainder,lightPdf,maxT,
                    intersection,interaction,epsilon
                );
                throughput *= lightRemainder;
            }
            
            // do a cool trick and always compute the bsdf parts this way! (no divergence)
            float bsdfPdf;
            // the value of the bsdf divided by the probability of the sample being generated
            throughput *= irr_glsl_bsdf_cos_remainder_and_pdf(bsdfPdf,_sample,interaction,bsdf);

            const vec3 lumaCoeffs = transpose(irr_glsl_scRGBtoXYZ)[1];
            if (bsdfPdf>FLT_MIN && dot(throughput,lumaCoeffs)>FLT_MIN) // TODO: proper thresholds for probability and color contribution
            {
                rayStack[stackPtr]._payload.throughput = throughput*rcpChoiceProb;

                float heuristicFactor = rcpChoiceProb-1.0; // weightNonGenerator/weightGenerator
                heuristicFactor /= pickBSDF ? bsdfPdf:lightPdf; // weightNonGenerator/(weightGenerator*probGenerated)
                heuristicFactor *= heuristicFactor; // (weightNonGenerator/(weightGenerator*probGenerated))^2
                if (!pickBSDF)
                    heuristicFactor = 1.0/(1.0/bsdfPdf+heuristicFactor*bsdfPdf); // numerically stable, don't touch
                rayStack[stackPtr]._payload.otherTechniqueHeuristic = heuristicFactor;
                    
                // trace new ray
                rayStack[stackPtr]._immutable.origin = intersection+_sample.L*(pickBSDF ? 1.0/*kSceneSize*/:maxT)*INTERSECTION_ERROR_BOUND;
                rayStack[stackPtr]._immutable.maxT = maxT;
                rayStack[stackPtr]._immutable.direction = _sample.L;
                rayStack[stackPtr]._immutable.typeDepthSampleIx = bitfieldInsert(sampleIx,depth+1,DEPTH_BITS_OFFSET,DEPTH_BITS_COUNT)|(pickBSDF ? 0:ANY_HIT_FLAG);
                stackPtr++;
            }
        }
    }
}

void main()
{
    if (((MAX_DEPTH-1)>>MAX_DEPTH_LOG2)>0 || ((SAMPLES-1)>>MAX_SAMPLES_LOG2)>0)
    {
        pixelColor = vec4(1.0,0.0,0.0,1.0);
        return;
    }

	uint scramble = textureLod(scramblebuf,TexCoord,0).r;
    const vec2 pixOffsetParam = vec2(2.0)/vec2(textureSize(scramblebuf,0)); // depending on denoiser used, we could use Lanczos or Gaussian filter instead of Box


    const mat4 invMVP = inverse(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(cameraData.params.MVP));
    
    vec4 NDC = vec4(TexCoord*vec2(2.0,-2.0)+vec2(-1.0,1.0),0.0,1.0);
    vec3 camPos;
    {
        vec4 tmp = invMVP*NDC;
        camPos = tmp.xyz/tmp.w;
        NDC.z = 1.0;
    }

    vec3 color = vec3(0.0);
    for (int i=0; i<SAMPLES; i++)
    {
        uint scramble_state = scramble;

        stackPtr = 0;
        // raygen
        {
            rayStack[stackPtr]._immutable.origin = camPos;
            rayStack[stackPtr]._immutable.maxT = FLT_MAX;

            vec4 tmp = NDC;
            tmp.xy += pixOffsetParam*(rand3d(0u,i,scramble_state).xy-vec2(0.5));
            //tmp.xy += pixOffsetParam*irr_glsl_concentricMapping(rand3d(0u,i,scramble_state).xy)*25.0;
            tmp = invMVP*tmp;
            rayStack[stackPtr]._immutable.direction = normalize(tmp.xyz/tmp.w-camPos);
            
            rayStack[stackPtr]._immutable.typeDepthSampleIx = bitfieldInsert(i,1,DEPTH_BITS_OFFSET,DEPTH_BITS_COUNT);


            rayStack[stackPtr]._payload.accumulation = vec3(0.0);
            rayStack[stackPtr]._payload.otherTechniqueHeuristic = 0.0; // needed for direct eye-light paths
            rayStack[stackPtr]._payload.throughput = vec3(1.0);
        }

        // trace
        while (stackPtr!=-1)
        {
            ImmutableRay_t _immutable = rayStack[stackPtr]._immutable;
            bool anyHitType = traceRay(_immutable);
                
            if (rayStack[stackPtr]._mutable.intersectionT>=_immutable.maxT)
            {
                missProgram();
            }
            else if (!anyHitType)
            {
                closestHitProgram(_immutable,scramble_state);
            }
            stackPtr--;
        }

        color += rayStack[0]._payload.accumulation;
    }
  
    pixelColor = vec4(color/float(SAMPLES), 1.0);

/** TODO: Improving Rendering

Quality:
- Geometry Specular AA (Curvature adjusted roughness)
- Density Based Outlier Rejection
- Gaussian Reconstruction Filter (off for AI denoising)
- Thinlens Model DoF

When proper scheduling is available:
- Russian Roulette
- Divergence Optimization

When finally texturing:
- Covariance Rendering
- CLEAR/LEAN/Toksvig for simult roughness + bumpmap filtering

Many Lights:
- Path Guiding
- Light Importance Lists/Classification
**/
}	