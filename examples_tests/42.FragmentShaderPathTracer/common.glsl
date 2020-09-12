// basic settings
#define MAX_DEPTH 8
#define SAMPLES 128

// firefly and variance reduction techniques
//#define KILL_DIFFUSE_SPECULAR_PATHS
//#define VISUALIZE_HIGH_VARIANCE

layout(set = 3, binding = 0) uniform sampler2D envMap; 
layout(set = 3, binding = 1) uniform usamplerBuffer sampleSequence;
layout(set = 3, binding = 2) uniform usampler2D scramblebuf;

layout(location = 0) in vec2 TexCoord;

layout(location = 0) out vec4 pixelColor;


#include <irr/builtin/glsl/limits/numeric.glsl>
#include <irr/builtin/glsl/math/constants.glsl>
#include <irr/builtin/glsl/utils/common.glsl>

//! @Crisspl move this to `irr/builtin/glsl/sampling.glsl` (along with the circle transform)
vec2 irr_glsl_BoxMullerTransform(in vec2 xi, in float stddev)
{
    float sinPhi, cosPhi;
    irr_glsl_sincos(2.0 * irr_glsl_PI * xi.y - irr_glsl_PI, sinPhi, cosPhi);
    return vec2(cosPhi, sinPhi) * sqrt(-2.0 * log(xi.x)) * stddev;
}

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
    return 2.0*irr_glsl_PI*(1.0-cosThetaMax);
}
float Sphere_getSolidAngle(in Sphere sphere, in vec3 origin)
{
    float cosThetaMax = sqrt(1.0-sphere.radius2/irr_glsl_lengthSq(sphere.position-origin));
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

    return t>0.f&&u>=0.f&&v>=0.f&&(u+v)<=1.f ? t:irr_glsl_FLT_NAN;
}

vec3 Triangle_getNormalTimesArea_impl(in mat2x3 edges)
{
    return cross(edges[0],edges[1])*0.5;
}
vec3 Triangle_getNormalTimesArea(in Triangle tri)
{
    return Triangle_getNormalTimesArea_impl(mat2x3(tri.vertex1-tri.vertex0,tri.vertex2-tri.vertex0));
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
mat2x3 BSDFNode_getEta(in BSDFNode node)
{
    return mat2x3(BSDFNode_getRealEta(node),BSDFNode_getImaginaryEta(node));
}

float BSDFNode_getMISWeight(in BSDFNode bsdf)
{
    const float alpha = BSDFNode_getRoughness(bsdf);
    const bool notDiffuse = BSDFNode_isNotDiffuse(bsdf);
    const float DIFFUSE_MIS_WEIGHT = 0.5;
    return notDiffuse ? mix(1.0,DIFFUSE_MIS_WEIGHT,alpha):DIFFUSE_MIS_WEIGHT; // TODO: test alpha*alpha
}

#include <irr/builtin/glsl/colorspace/EOTF.glsl>
#include <irr/builtin/glsl/colorspace/encodeCIEXYZ.glsl>
float getLuma(in vec3 col)
{
    return dot(transpose(irr_glsl_scRGBtoXYZ)[1],col);
}

#define BSDF_COUNT 7
BSDFNode bsdfs[BSDF_COUNT] = {
    {{uvec4(floatBitsToUint(vec3(0.8,0.8,0.8)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(0.8,0.4,0.4)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(0.4,0.8,0.4)),DIFFUSE_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.02,1.3)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,1.0,2.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.3,1.02)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,2.0,1.0,0.0))}},
    {{uvec4(floatBitsToUint(vec3(1.02,1.3,1.02)),CONDUCTOR_OP),floatBitsToUint(vec4(1.0,2.0,1.0,0.15))}},
    {{uvec4(floatBitsToUint(vec3(1.4,1.45,1.5)),DIELECTRIC_OP),floatBitsToUint(vec4(0.0,0.0,0.0,0.0))}}
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



#define ANY_HIT_FLAG (-2147483648)
#define DEPTH_BITS_COUNT 8
#define DEPTH_BITS_OFFSET (31-DEPTH_BITS_COUNT)
struct ImmutableRay_t
{
    vec3 origin;
    float maxT;
    vec3 direction;
    int typeDepthSampleIx;
#ifdef TRIANGLE_METHOD
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
#define MAX_STACK_SIZE 1
int stackPtr = 0;
Ray_t rayStack[MAX_STACK_SIZE];


bool anyHitProgram(in ImmutableRay_t _immutable)
{
    return true;
}


#define INTERSECTION_ERROR_BOUND_LOG2 (-13.0)
float getTolerance_common(in int depth)
{
    float depthRcp = 1.0/float(depth);
    return INTERSECTION_ERROR_BOUND_LOG2;// *depthRcp*depthRcp;
}
float getStartTolerance(in int depth)
{
    return exp2(getTolerance_common(depth));
}
float getEndTolerance(in int depth)
{
    return 1.0-exp2(getTolerance_common(depth)+1.0);
}


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
        const vec3 kConstantEnvLightRadiance = vec3(0.15, 0.21, 0.3);
            finalContribution *= kConstantEnvLightRadiance;
        #endif
    }
    else
    {
        finalContribution *= rayStack[stackPtr]._payload.otherTechniqueHeuristic;
    }
    rayStack[stackPtr]._payload.accumulation += finalContribution;
}

#include <irr/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <irr/builtin/glsl/bxdf/bsdf/specular/dielectric.glsl>
irr_glsl_BSDFSample irr_glsl_bsdf_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u, in BSDFNode bsdf, in vec3 luminosityContributionHint)
{
    const float a = BSDFNode_getRoughness(bsdf);
    const mat2x3 ior = BSDFNode_getEta(bsdf);

    irr_glsl_BSDFSample smpl;
    switch (BSDFNode_getType(bsdf))
    {
        case DIFFUSE_OP:
            smpl = irr_glsl_oren_nayar_cos_generate(interaction,u.xy,a*a);
            break;
        case CONDUCTOR_OP:
            smpl = irr_glsl_ggx_cos_generate(interaction,u.xy,a,a);
            break;
        default:
            smpl = irr_glsl_thin_smooth_dielectric_cos_generate(interaction,u,ior[0],luminosityContributionHint);
            break;
    }
    return smpl;
}

vec3 irr_glsl_bsdf_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample _sample, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in BSDFNode bsdf, in vec3 luminosityContributionHint)
{
    const vec3 reflectance = BSDFNode_getReflectance(bsdf);
    const float a = max(BSDFNode_getRoughness(bsdf),0.01);
    mat2x3 ior = BSDFNode_getEta(bsdf);
    
    // if V and L are on different sides of the surface normal, then their dot product sign bits will differ, hence XOR will yield 1 at last bit
    const bool transmitted = ((floatBitsToUint(interaction.isotropic.NdotV)^floatBitsToUint(_sample.NdotL))&0x80000000u)!=0u;

    const bool transmissive = BSDFNode_isBSDF(bsdf);
    _sample.NdotL = irr_glsl_conditionalAbsOrMax(transmissive,_sample.NdotL,0.0);
    interaction.isotropic.NdotV = irr_glsl_conditionalAbsOrMax(transmissive,interaction.isotropic.NdotV,0.0);

    vec3 remainder;
    switch (BSDFNode_getType(bsdf))
    {
        case DIFFUSE_OP:
            remainder = reflectance*irr_glsl_oren_nayar_cos_remainder_and_pdf_wo_clamps(pdf,_sample,interaction.isotropic,a*a);
            break;
        case CONDUCTOR_OP:
            remainder = irr_glsl_ggx_cos_remainder_and_pdf(pdf,_sample,interaction.isotropic,ior,a*a);
            break;
        default:
            remainder = irr_glsl_thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(pdf,transmitted,interaction.isotropic.NdotV,ior[0],luminosityContributionHint);
            break;
    }
    return remainder;
}


#define GeneratorSample irr_glsl_BSDFSample
#define irr_glsl_LightSample irr_glsl_BSDFSample
irr_glsl_LightSample irr_glsl_createLightSample(in vec3 L, in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BSDFSample s;
    s.L = L;

    s.TdotL = dot(interaction.T,L);
    s.BdotL = dot(interaction.B,L);
    s.NdotL = dot(interaction.isotropic.N,L);
   
    float VdotL = dot(interaction.isotropic.V.dir,L);
    float LplusV_rcpLen = inversesqrt(2.0+2.0*VdotL);

    s.TdotH = (interaction.TdotV+s.TdotL)*LplusV_rcpLen;
    s.BdotH = (interaction.BdotV+s.BdotL)*LplusV_rcpLen;
    s.NdotH = (interaction.isotropic.NdotV+s.NdotL)*LplusV_rcpLen;

    s.VdotH = LplusV_rcpLen+LplusV_rcpLen*VdotL;
    
    return s;
}

layout (constant_id = 0) const int MAX_DEPTH_LOG2 = 0;
layout (constant_id = 1) const int MAX_SAMPLES_LOG2 = 0;


// TODO: @Przemog move to a GLSL math header, unless there is a native GLSL function for this
uint irr_glsl_rotl(in uint x, in uint k)
{
	return (x<<k) | (x>>(32u-k));
}


// TODO: @Przemog move to a GLSL built-in "random" header
#define irr_glsl_xoroshiro64star_state_t uvec2
#define irr_glsl_xoroshiro64starstar_state_t uvec2
void irr_glsl_xoroshiro64_state_advance(inout uvec2 state)
{
	state[1] ^= state[0];
	state[0] = irr_glsl_rotl(state[0], 26u) ^ state[1] ^ (state[1]<<9u); // a, b
	state[1] = irr_glsl_rotl(state[1], 13u); // c
}

uint irr_glsl_xoroshiro64star(inout irr_glsl_xoroshiro64starstar_state_t state)
{
	const uint result = state[0]*0x9E3779BBu;

    irr_glsl_xoroshiro64_state_advance(state);

	return result;
}
uint irr_glsl_xoroshiro64starstar(inout irr_glsl_xoroshiro64starstar_state_t state)
{
	const uint result = irr_glsl_rotl(state[0]*0x9E3779BBu,5u)*5u;
    
    irr_glsl_xoroshiro64_state_advance(state);

	return result;
}
// dont move anything below this line

vec3 rand3d(in uint protoDimension, in uint _sample, inout irr_glsl_xoroshiro64star_state_t scramble_state)
{
    uint address = bitfieldInsert(protoDimension,_sample,MAX_DEPTH_LOG2,MAX_SAMPLES_LOG2);
	uvec3 seqVal = texelFetch(sampleSequence,int(address)).xyz;
	seqVal ^= uvec3(irr_glsl_xoroshiro64star(scramble_state),irr_glsl_xoroshiro64star(scramble_state),irr_glsl_xoroshiro64star(scramble_state));
    return vec3(seqVal)*uintBitsToFloat(0x2f800004u);
}

bool traceRay(in ImmutableRay_t _immutable);
void closestHitProgram(in ImmutableRay_t _immutable, inout irr_glsl_xoroshiro64star_state_t scramble_state);

void main()
{
    if (((MAX_DEPTH-1)>>MAX_DEPTH_LOG2)>0 || ((SAMPLES-1)>>MAX_SAMPLES_LOG2)>0)
    {
        pixelColor = vec4(1.0,0.0,0.0,1.0);
        return;
    }

	irr_glsl_xoroshiro64star_state_t scramble_start_state = textureLod(scramblebuf,TexCoord,0).rg;
    const vec2 pixOffsetParam = vec2(1.0)/vec2(textureSize(scramblebuf,0));


    const mat4 invMVP = inverse(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(cameraData.params.MVP));
    
    vec4 NDC = vec4(TexCoord*vec2(2.0,-2.0)+vec2(-1.0,1.0),0.0,1.0);
    vec3 camPos;
    {
        vec4 tmp = invMVP*NDC;
        camPos = tmp.xyz/tmp.w;
        NDC.z = 1.0;
    }

    vec3 color = vec3(0.0);
    float meanLumaSquared = 0.0;
    for (int i=0; i<SAMPLES; i++)
    {
        irr_glsl_xoroshiro64star_state_t scramble_state = scramble_start_state;

        stackPtr = 0;
        // raygen
        {
            rayStack[stackPtr]._immutable.origin = camPos;
            rayStack[stackPtr]._immutable.maxT = FLT_MAX;

            vec4 tmp = NDC;
            // apply stochastic reconstruction filter
            const float gaussianFilterCutoff = 2.5;
            const float truncation = exp(-0.5*gaussianFilterCutoff*gaussianFilterCutoff);
            vec2 remappedRand = rand3d(0u,i,scramble_state).xy;
            remappedRand.x *= 1.0-truncation;
            remappedRand.x += truncation;
            tmp.xy += pixOffsetParam*irr_glsl_BoxMullerTransform(remappedRand,1.5);
            // for depth of field we could do another stochastic point-pick
            tmp = invMVP*tmp;
            rayStack[stackPtr]._immutable.direction = normalize(tmp.xyz/tmp.w-camPos);
            
            rayStack[stackPtr]._immutable.typeDepthSampleIx = bitfieldInsert(i,1,DEPTH_BITS_OFFSET,DEPTH_BITS_COUNT);

            #ifdef TRIANGLE_METHOD
                rayStack[stackPtr]._immutable.normalAtOrigin = vec3(0.0,0.0,0.0);
                rayStack[stackPtr]._immutable.wasBSDFAtOrigin = false;
            #endif

            rayStack[stackPtr]._payload.accumulation = vec3(0.0);
            rayStack[stackPtr]._payload.otherTechniqueHeuristic = 0.0; // needed for direct eye-light paths
            rayStack[stackPtr]._payload.throughput = vec3(1.0);
            #ifdef KILL_DIFFUSE_SPECULAR_PATHS
            rayStack[stackPtr]._payload.hasDiffuse = false;
            #endif
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

        vec3 accumulation = rayStack[0]._payload.accumulation;

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

    pixelColor = vec4(color, 1.0);
}
/** TODO: Improving Rendering

Now:
- Always MIS (path correlated reuse)
- Proper Universal&Robust Materials
- Test MIS alpha (roughness) scheme

Quality:
-* Reweighting Noise Removal
-* Covariance Rendering
-* Geometry Specular AA (Curvature adjusted roughness)

When proper scheduling is available:
- Russian Roulette
- Divergence Optimization
- Adaptive Sampling

Offline firefly removal:
- Density Based Outlier Rejection (requires fast k-nearest neighbours on the GPU, at which point we've pretty much got photonmapping ready)

When finally texturing:
- Covariance Rendering
- CLEAR/LEAN/Toksvig for simult roughness + bumpmap filtering

Many Lights:
- Path Guiding
- Light Importance Lists/Classification

Indirect Light:
- Bidirectional Path Tracing
- Uniform Path Sampling / Vertex Connection and Merging / Path Space Regularization

Animations:
- A-SVGF / BMFR
**/