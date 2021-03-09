#version 430 core
#extension GL_GOOGLE_include_directive : require

#define SPHERE_COUNT 8
#define RECTANGLE_METHOD 0 // 0 area sampling, 1 solid angle sampling, 2 approximate projected solid angle sampling
#include "common.glsl"

#define RECTANGLE_COUNT 1
Rectangle rectangles[RECTANGLE_COUNT] = {
    Rectangle_Rectangle(vec3(-1.8,0.35,0.3),vec3(0.6,0.0,-0.3),vec3(0.3,0.45,-0.6),INVALID_ID_16BIT,0u)
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
	for (int i=0; i<RECTANGLE_COUNT; i++)
    {
        float t = Rectangle_intersect(rectangles[i],_immutable.origin,_immutable.direction);
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


/// #include <nbl/builtin/glsl/sampling/projected_spherical_rectangle.glsl>


// the interaction here is the interaction at the illuminator-end of the ray, not the receiver
vec3 nbl_glsl_light_deferred_eval_and_prob(
    out float pdf, in Light light, in vec3 L
#if RECTANGLE_METHOD==0
    ,in float intersectionT
#else
    ,in vec3 origin
#if RECTANGLE_METHOD==2
    ,in vec3 normalAtOrigin, in bool wasBSDFAtOrigin
#endif
#endif
)
{
    // we don't have to worry about solid angle of the light w.r.t. surface of the light because this function only ever gets called from closestHit routine, so such ray cannot be produced
    pdf = scene_getLightChoicePdf(light);

    Rectangle rect = rectangles[Light_getObjectID(light)];
#if RECTANGLE_METHOD==0
    pdf *= intersectionT*intersectionT/abs(dot(Rectangle_getNormalTimesArea(rect),L));
#else
    const mat3 sphericalVertices = nbl_glsl_shapes_getSphericalTriangle(mat3(tri.vertex0,tri.vertex1,tri.vertex2),origin);
    Triangle tmpTri = Triangle_Triangle(mat3(tri.vertex0,tri.vertex1,tri.vertex2),0u,0u);
    #if RECTANGLE_METHOD==1
        float rcpProb = nbl_glsl_shapes_SolidAngleOfTriangle(sphericalVertices);
        // if `rcpProb` is NAN then the triangle's solid angle was close to 0.0 
        pdf = rcpProb>FLT_MIN ? (pdf/rcpProb):FLT_MAX;
    #elif RECTANGLE_METHOD==2
        pdf *= nbl_glsl_sampling_probProjectedSphericalTriangleSample(sphericalVertices,normalAtOrigin,wasBSDFAtOrigin,L);
        // if `pdf` is NAN then the triangle's projected solid angle was close to 0.0, if its close to INF then the triangle was very small
        pdf = pdf<FLT_MAX ? pdf:0.0;
    #endif
#endif
    return Light_getRadiance(light);
}


nbl_glsl_LightSample nbl_glsl_light_generate_and_remainder_and_pdf(out vec3 remainder, out float pdf, out float newRayMaxT, in vec3 origin, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in bool isBSDF, in vec3 u, in int depth)
{
    // normally we'd pick from set of lights, using `u.z`
    const Light light = lights[0];
    const float choicePdf = scene_getLightChoicePdf(light);

    const Rectangle rect = rectangles[Light_getObjectID(light)];
    
#if RECTANGLE_METHOD==0
    const vec3 point = rect.offset+rect.edge0*u.x+rect.edge1*u.y;
    vec3 L = point-origin;
    
    const float distanceSq = dot(L,L);
    const float rcpDistance = inversesqrt(distanceSq);
    L *= rcpDistance;

    const float dist = 1.0/rcpDistance;
    
    const float rcpPdf = abs(dot(Rectangle_getNormalTimesArea(rect),L))/(distanceSq*choicePdf);
#else 
    float rcpPdf;

    const mat3 sphericalVertices = nbl_glsl_shapes_getSphericalTriangle(mat3(tri.vertex0,tri.vertex1,tri.vertex2),origin);
#if RECTANGLE_METHOD==1
    const vec3 L = nbl_glsl_sampling_generateSphericalTriangleSample(rcpPdf,sphericalVertices,u.xy);
#elif RECTANGLE_METHOD==2
    const vec3 L = nbl_glsl_sampling_generateProjectedSphericalTriangleSample(rcpPdf,sphericalVertices,interaction.isotropic.N,isBSDF,u.xy);
#endif
    // if `rcpProb` is NAN or negative then the triangle's solidAngle or projectedSolidAngle was close to 0.0 
    rcpPdf = rcpPdf>FLT_MIN ? rcpPdf:0.0;

    const vec3 N = Triangle_getNormalTimesArea(tri);
    const float dist = dot(N,tri.vertex0-origin)/dot(N,L);
#endif

    remainder = Light_getRadiance(light)*rcpPdf;
    pdf = 1.0/rcpPdf;

    newRayMaxT = getEndTolerance(depth)*dist;
    
    return nbl_glsl_createLightSample(L,interaction);
}

void closestHitProgram(in ImmutableRay_t _immutable, inout nbl_glsl_xoroshiro64star_state_t scramble_state)
{
    const MutableRay_t mutable = rayStack[stackPtr]._mutable;

    vec3 intersection = _immutable.origin+_immutable.direction*mutable.intersectionT;
    const uint objectID = mutable.objectID;
    
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
            Rectangle rect = rectangles[objectID-SPHERE_COUNT];
            isotropic.N = normalize(Rectangle_getNormalTimesArea(rect));
            bsdfLightIDs = rect.bsdfLightIDs;
        }
        isotropic.NdotV = dot(isotropic.V.dir,isotropic.N);
        isotropic.NdotV_squared = isotropic.NdotV*isotropic.NdotV;

        interaction = nbl_glsl_calcAnisotropicInteraction(isotropic);
    }

    const uint lightID = bitfieldExtract(bsdfLightIDs,16,16);

    vec3 throughput = rayStack[stackPtr]._payload.throughput;
    // finish MIS
    if (lightID!=INVALID_ID_16BIT) // has emissive
    {
        float lightPdf;
        vec3 lightVal = nbl_glsl_light_deferred_eval_and_prob(
            lightPdf,lights[lightID],_immutable.direction
        #if RECTANGLE_METHOD==0
            ,mutable.intersectionT
        #else
            ,_immutable.origin
        #if RECTANGLE_METHOD==2
            ,_immutable.normalAtOrigin,_immutable.wasBSDFAtOrigin
        #endif
        #endif
        );
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
        mat2x3 epsilon = rand3d(depth,sampleIx,scramble_state);
    
        float rcpChoiceProb;
        const bool doNEE = nbl_glsl_partitionRandVariable(bsdfGeneratorProbability,epsilon[0].z,rcpChoiceProb);
    

        float maxT;
        // the probability of generating a sample w.r.t. the light generator only possible and used when it was generated with it!
        float lightPdf;
        nbl_glsl_LightSample _sample;
        nbl_glsl_AnisotropicMicrofacetCache _cache;
        const bool isBSDF = BSDFNode_isBSDF(bsdf);
        if (doNEE)
        {
            vec3 lightRemainder;
            _sample = nbl_glsl_light_generate_and_remainder_and_pdf(
                lightRemainder,lightPdf,maxT,
                intersection,interaction,
                isBSDF,epsilon[0],depth
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
            rayStack[stackPtr]._immutable.typeDepthSampleIx = bitfieldInsert(sampleIx,depth+2,DEPTH_BITS_OFFSET,DEPTH_BITS_COUNT)|(doNEE ? ANY_HIT_FLAG:0);
            rayStack[stackPtr]._immutable.normalAtOrigin = interaction.isotropic.N;
            rayStack[stackPtr]._immutable.wasBSDFAtOrigin = isBSDF;
            stackPtr++;
        }
    }
}