#version 430 core
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

#define SPHERE_COUNT 8
Sphere spheres[SPHERE_COUNT] = {
    Sphere_Sphere(vec3(0.0,-100.5,-1.0),100.0,0u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(2.0,0.0,-1.0),0.5,1u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(0.0,0.0,-1.0),0.5,2u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(-2.0,0.0,-1.0),0.5,3u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(2.0,0.0,1.0),0.5,4u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(0.0,0.0,1.0),0.5,4u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(-2.0,0.0,1.0),0.5,5u,INVALID_ID_16BIT),
    Sphere_Sphere(vec3(0.5,1.0,0.5),0.5,6u,INVALID_ID_16BIT)
};
#define TRIANGLE_COUNT 1
Triangle triangles[TRIANGLE_COUNT] = {
    Triangle_Triangle(mat3(vec3(-1.8,1.2,0.3),vec3(-1.5,1.2,0.0),vec3(-1.5,1.8,0.0)),INVALID_ID_16BIT,0u)
};


#define LIGHT_COUNT 1
Light lights[LIGHT_COUNT] = {
    {vec3(30.0,25.0,15.0),8u}
};


bool traceRay(in ImmutableRay_t _immutable)
{
    const bool anyHit = bitfieldExtract(_immutable.typeDepthSampleIx,31,1)!=0;

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
	for (int i=0; i<SPHERE_COUNT; i++)
    {
        float t = Triangle_intersect(triangles[i],_immutable.origin,_immutable.direction);
        bool closerIntersection = t>0.0 && t<intersectionT;

		objectID = closerIntersection ? (i+SPHERE_COUNT):objectID;
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


#if 0
// the interaction here is the interaction at the illuminator-end of the ray, not the receiver
vec3 irr_glsl_light_deferred_eval_and_prob(out float pdf, in Triangle tri, in vec3 origin, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in Light light)
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
irr_glsl_LightSample irr_glsl_light_generate_and_remainder_and_pdf(out vec3 remainder, out float pdf, out float newRayMaxT, in vec3 origin, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u, in int depth)
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
    irr_glsl_sincos(2.0*irr_glsl_PI*u.y-irr_glsl_PI,sinPhi,cosPhi);
    mat2x3 XY = irr_glsl_frisvad(Z);
    
    L += (XY[0]*cosPhi+XY[1]*sinPhi)*sinTheta;
    
    const float rcpPdf = Sphere_getSolidAngle_impl(cosThetaMax)/choicePdf;
    remainder = Light_getRadiance(light)*(possibilityOfLightVisibility ? rcpPdf:0.0); // this ternary operator kills invalid rays
    pdf = 1.0/rcpPdf;
    
    newRayMaxT = (cosTheta-sqrt(cosTheta2-cosThetaMax2))/rcpDistance*getEndTolerance(depth);
    
    return irr_glsl_createLightSample(L,interaction);
}
#endif

void closestHitProgram(in ImmutableRay_t _immutable, inout irr_glsl_xoroshiro64star_state_t scramble_state)
{
    const MutableRay_t mutable = rayStack[stackPtr]._mutable;

    vec3 intersection = _immutable.origin+_immutable.direction*mutable.intersectionT;
    const uint objectID = [mutable.objectID;
    
    irr_glsl_AnisotropicViewSurfaceInteraction interaction;
    {
        irr_glsl_IsotropicViewSurfaceInteraction isotropic;

        isotropic.V.dir = -_immutable.direction;
        //isotropic.V.dPosdScreen = screw that
        const float radiusRcp = inversesqrt(sphere.radius2);
        isotropic.N = (intersection-sphere.position)*radiusRcp;
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

        #ifdef KILL_DIFFUSE_SPECULAR_PATHS
        if (BSDFNode_isNotDiffuse(bsdf))
        {
            if (rayStack[stackPtr]._payload.hasDiffuse)
                return;
        }
        else
            rayStack[stackPtr]._payload.hasDiffuse = true;
        #endif


        const float bsdfGeneratorProbability = BSDFNode_getMISWeight(bsdf);    
        vec3 epsilon = rand3d(depth,sampleIx,scramble_state);
    
        float rcpChoiceProb;
        const bool doNEE = irr_glsl_partitionRandVariable(bsdfGeneratorProbability,epsilon.z,rcpChoiceProb);
    

        float maxT;
        // the probability of generating a sample w.r.t. the light generator only possible and used when it was generated with it!
        float lightPdf;
        GeneratorSample _sample;
        if (doNEE)
        {
            vec3 lightRemainder;
            _sample = irr_glsl_light_generate_and_remainder_and_pdf(
                lightRemainder,lightPdf,maxT,
                intersection,interaction,epsilon,
                depth
            );
            throughput *= lightRemainder;
        }
        else
        {
            maxT = FLT_MAX;
            _sample = irr_glsl_bsdf_cos_generate(interaction,epsilon,bsdf);
        }
            
        // do a cool trick and always compute the bsdf parts this way! (no divergence)
        float bsdfPdf;
        // the value of the bsdf divided by the probability of the sample being generated
        throughput *= irr_glsl_bsdf_cos_remainder_and_pdf(bsdfPdf,_sample,interaction,bsdf);

        // OETF smallest perceptible value
        const float bsdfPdfThreshold = getLuma(irr_glsl_eotf_sRGB(vec3(1.0)/255.0));
        const float lumaThroughputThreshold = bsdfPdfThreshold;
        if (bsdfPdf>bsdfPdfThreshold && getLuma(throughput)>lumaThroughputThreshold)
        {
            rayStack[stackPtr]._payload.throughput = throughput*rcpChoiceProb;

            float heuristicFactor = rcpChoiceProb-1.0; // weightNonGenerator/weightGenerator
            heuristicFactor /= doNEE ? lightPdf:bsdfPdf; // weightNonGenerator/(weightGenerator*probGenerated)
            heuristicFactor *= heuristicFactor; // (weightNonGenerator/(weightGenerator*probGenerated))^2
            if (doNEE)
                heuristicFactor = 1.0/(1.0/bsdfPdf+heuristicFactor*bsdfPdf); // numerically stable, don't touch
            rayStack[stackPtr]._payload.otherTechniqueHeuristic = heuristicFactor;
                    
            // trace new ray
            rayStack[stackPtr]._immutable.origin = intersection+_sample.L*(doNEE ? maxT:1.0/*kSceneSize*/)*getStartTolerance(depth);
            rayStack[stackPtr]._immutable.maxT = maxT;
            rayStack[stackPtr]._immutable.direction = _sample.L;
            rayStack[stackPtr]._immutable.typeDepthSampleIx = bitfieldInsert(sampleIx,depth+1,DEPTH_BITS_OFFSET,DEPTH_BITS_COUNT)|(doNEE ? ANY_HIT_FLAG:0);
            stackPtr++;
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

/** TODO: Improving Rendering

Now:
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
}	