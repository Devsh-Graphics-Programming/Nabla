#version 430 core

layout(set = 3, binding = 0) uniform sampler2D envMap; 

layout(location = 0) in vec2 TexCoord;

layout(location = 0) out vec4 pixelColor;


// TODO: change the location of the `irr_glsl_SBasicViewParameters`, vertex header is not the right place for it
#include <irr/builtin/glsl/utils/vertex.glsl>

layout(set = 1, binding = 0, row_major, std140) uniform UBO
{
	irr_glsl_SBasicViewParameters params;
} cameraData;

#define INVALID_ID_16BIT 0xffff0000u
struct Sphere
{
    vec3 position;
    float radius2;
    uint bsdfLightIDs;
};

float sqr(in float x)
{
    return x*x;
}

#define SPHERE_COUNT 9
Sphere spheres[SPHERE_COUNT] = {
    {{0.0,-100.5,-1.0},sqr(100.0),0u|INVALID_ID_16BIT},
    {{2.0,0.0,-1.0},sqr(0.5),1u|INVALID_ID_16BIT},
    {{0.0,0.0,-1.0},sqr(0.5),2u|INVALID_ID_16BIT},
    {{-2.0,0.0,-1.0},sqr(0.5),3u|INVALID_ID_16BIT},
    {{2.0,0.0,1.0},sqr(0.5),4u|INVALID_ID_16BIT},
    {{0.0,0.0,1.0},sqr(0.5),4u|INVALID_ID_16BIT},
    {{-2.0,0.0,1.0},sqr(0.5),5u|INVALID_ID_16BIT},
    {{0.5,1.0,0.5},sqr(0.5),6u|INVALID_ID_16BIT},
    {{-1.5,1.5,0.0},sqr(0.3),7u|0u}
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

#include <irr/builtin/glsl/colorspace/EOTF.glsl>

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
};

#define LIGHT_COUNT 1
Light lights[LIGHT_COUNT] = {
    {vec3(30.0,25.0,15.0)}
};

float scene_getLightChoicePdf(in Light light)
{
    return 1.0/float(LIGHT_COUNT);
}


#define ANY_HIT_FLAG (-1)
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

bool anyHitProgram(in ImmutableRay_t _immutable)
{
    return true;
}

bool traceRay(in ImmutableRay_t _immutable, inout MutableRay_t _mutable)
{
    bool anyHit = false;//(_immutable.typeDepthSampleIx&ANY_HIT_FLAG)!=0;
    _mutable.intersectionT = _immutable.maxT;
    for (int i=0; i<SPHERE_COUNT; i++)
    {
        vec3 origin = _immutable.origin-spheres[i].position;
        float originLen2 = dot(origin,origin);

        float dirDotOrigin = dot(_immutable.direction,origin);
        float det = spheres[i].radius2-originLen2+dirDotOrigin*dirDotOrigin;

        // do some speculative math here
        float t = -dirDotOrigin-sqrt(det);
        bool closerIntersection = !isnan(t) && t>0.0 && t<_mutable.intersectionT;
        _mutable.intersectionT = closerIntersection ? t:_mutable.intersectionT;

        if (anyHit && closerIntersection && anyHitProgram(_immutable))
            break;

        _mutable.objectID = closerIntersection ? i:_mutable.objectID;
    }
    // hit
    return anyHit;
}


struct Ray_t
{
    ImmutableRay_t _immutable;
    MutableRay_t _mutable;
    Payload_t _payload;
};

#include <irr/builtin/glsl/math/constants.glsl>
vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= irr_glsl_RECIPROCAL_PI*0.5;
    uv += 0.5;
    return uv;
}

void missProgram(in ImmutableRay_t _immutable, inout Payload_t payload)
{
    if (_immutable.maxT<FLT_MAX)
        payload.accumulation += payload.throughput;
    else
    {
    #define USE_ENVMAP

    #ifdef USE_ENVMAP
	    vec2 uv = SampleSphericalMap(_immutable.direction);
        payload.accumulation += payload.throughput*textureLod(envMap, uv, 0.0).rgb;
    #else
        const vec3 kConstantEnvLightRadiance = vec3(0.15,0.21,0.3);
        payload.accumulation += kConstantEnvLightRadiance;
    #endif
    }
}


#define MAX_STACK_SIZE 1
int stackPtr = 0;
Ray_t rayStack[MAX_STACK_SIZE];

#include <irr/builtin/glsl/bxdf/common.glsl>
float impl_sphereSolidAngle(in Sphere sphere, in vec3 origin, in irr_glsl_IsotropicViewSurfaceInteraction interaction)
{
    float cosThetaMax = sqrt(1.0-sphere.radius2/irr_glsl_lengthSq(sphere.position-origin));
    return 2.0*irr_glsl_PI*(isnan(cosThetaMax) ? 2.0:(1.0-cosThetaMax));
}

// the interaction here is the interaction at the illuminator-end of the ray, not the receiver
vec3 irr_glsl_light_deferred_eval_and_prob(out float lightPdf, in Sphere sphere, in vec3 origin, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in Light light)
{
    lightPdf = scene_getLightChoicePdf(light)/impl_sphereSolidAngle(sphere,origin,interaction);
    return light.radiance;
}

#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#define irr_glsl_LightSample irr_glsl_BSDFSample
irr_glsl_LightSample irr_glsl_light_generate(in Sphere sphere, in vec3 origin, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in vec2 u)
{
    irr_glsl_LightSample retval;
    retval.L = normalize(sphere.position-origin);
    retval.LdotN = dot(retval.L,interaction.N);
    retval.VdotH = dot(retval.L,normalize(retval.L+interaction.V.dir));
    return retval;
}
/*
irr_glsl_LightSample irr_glsl_createLightSample(in vec3 H, in vec3 V, in float VdotH, in mat3 m)
{
    irr_glsl_BSDFSample s;

    vec3 L = irr_glsl_reflect(V, H, VdotH);
    s.L = m * L; // m must be an orthonormal matrix
    s.LdotT = L.x;
    s.LdotB = L.y;
    s.LdotN = L.z;
    s.TdotH = H.x;
    s.BdotH = H.y;
    s.NdotH = H.z;
    s.VdotH = VdotH;

    return s;
}
*/

// TODO
float BSDFNode_getMISWeight(in BSDFNode bsdf)
{
    return 1.0;
}

#define MAX_DEPTH 2
void closestHitProgram(in ImmutableRay_t _immutable, in MutableRay_t _mutable, inout Payload_t _payload)
{
    Sphere sphere = spheres[_mutable.objectID];
    vec3 intersection = _immutable.origin+_immutable.direction*_mutable.intersectionT;
    
    irr_glsl_IsotropicViewSurfaceInteraction interaction;
    interaction.V.dir = -_immutable.direction;
    //interaction.V.dPosdScreen = screw that
    interaction.N = (intersection-sphere.position)*inversesqrt(sphere.radius2);
    interaction.NdotV = dot(interaction.V.dir,interaction.N);
    interaction.NdotV_squared = interaction.NdotV*interaction.NdotV;

    uint bsdfLightIDs = sphere.bsdfLightIDs;
    uint lightID = bitfieldExtract(bsdfLightIDs,16,16);

    _payload.accumulation += vec3(1.0,0.0,0.0);

#if 0

    // finish MIS
    if (lightID!=INVALID_ID_16BIT) // has emissive
    {
        float lightPdf;
        vec3 lightVal = irr_glsl_light_deferred_eval_and_prob(lightPdf,sphere,_immutable.origin,interaction,lights[lightID]);
        _payload.accumulation += _payload.throughput*lightVal/(1.0+lightPdf*lightPdf*_payload.otherTechniqueHeuristic);
    }

    int sampleIx = bitfieldExtract(_immutable.typeDepthSampleIx,0,DEPTH_BITS_OFFSET);
    int depth = bitfieldExtract(_immutable.typeDepthSampleIx,DEPTH_BITS_OFFSET,DEPTH_BITS_COUNT);

    // do we even have a BSDF at all
    uint bsdfID = bitfieldExtract(bsdfLightIDs,0,16);
    if (depth<MAX_DEPTH && bsdfID!=INVALID_ID_16BIT)
    {
        // common preload
        BSDFNode bsdf = bsdfs[bsdfID];
        uint opType = BSDFNode_getType(bsdf);

        float bsdfGeneratorProbability = BSDFNode_getMISWeight(bsdf);
        
        // this could run in a loop if we'd allow splitting/amplification (if we modified the payload managment)
        {
            //vec2 epsilon = rand2d(depth,sampleIx);

            _payload.accumulation += BSDFNode_getReflectance(bsdf);

            //stackPtr++;
            //rayStack[stackPtr].origin = interaction.pos;
        }
    }
#endif
}

#define SAMPLES 16
void main()
{
    mat4 invMVP = inverse(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(cameraData.params.MVP));
    vec4 NDC = vec4(TexCoord*2.0-vec2(1.0),0.0,1.0);
    NDC.y = -NDC.y;
    vec4 tmp = invMVP*NDC;
    vec3 camPos = tmp.xyz/tmp.w;

    vec3 color = vec3(0.0);
    for (int i=0; i<SAMPLES; i++)
    {
        stackPtr = 0;

        // raygen
        {
            rayStack[stackPtr]._immutable.origin = camPos;
            rayStack[stackPtr]._immutable.maxT = FLT_MAX;
            NDC.z = 1.0;
            tmp = invMVP*NDC;
            rayStack[stackPtr]._immutable.direction = normalize(tmp.xyz/tmp.w-camPos);
            rayStack[stackPtr]._immutable.typeDepthSampleIx = i|(1<<DEPTH_BITS_OFFSET);

            rayStack[stackPtr]._payload.accumulation = vec3(0.0);
            rayStack[stackPtr]._payload.otherTechniqueHeuristic = 0.0; // TODO: remove
            rayStack[stackPtr]._payload.throughput = vec3(1.0);
            stackPtr++;
        }

        // trace
        //while ((stackPtr--)!=0)
        stackPtr--;
        {
            ImmutableRay_t _immutable = rayStack[stackPtr]._immutable;
            MutableRay_t _mutable = rayStack[stackPtr]._mutable;
            bool anyHitType = traceRay(_immutable,_mutable);
                
            // TODO: better scheduling for less divergence in some version of this demo
            if (_mutable.intersectionT>=_immutable.maxT)
                missProgram(_immutable,rayStack[stackPtr]._payload);
            else if (!anyHitType)
                closestHitProgram(_immutable,_mutable,rayStack[stackPtr]._payload);
        }

        color += rayStack[stackPtr]._payload.accumulation;
    }
  
    pixelColor = vec4(color/float(SAMPLES), 1.0);
}	