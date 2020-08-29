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
    Triangle_Triangle(mat3(vec3(-1.8,0.35,0.3),vec3(-1.2,0.35,0.0),vec3(-1.5,0.8,-0.3)),INVALID_ID_16BIT,0u)
    //Triangle_Triangle(mat3(vec3(-4,0.7,-4),vec3(0.0,0.7,0.0),vec3(-4.0,0.8,4.0)),INVALID_ID_16BIT,0u)
};


#define LIGHT_COUNT 1
Light lights[LIGHT_COUNT] = {
    {vec3(30.0,25.0,15.0),0u}
    //{vec3(30.0,25.0,15.0)*0.01,0u}
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
	for (int i=0; i<TRIANGLE_COUNT; i++)
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


// @Crisspl move this to `irr/builtin/glsl/math/quaternions.glsl`
vec3 irr_glsl_slerp_impl_impl(in vec3 start, in vec3 preScaledWaypoint, float cosAngleFromStart)
{
    vec3 planeNormal = cross(start,preScaledWaypoint);
    
    cosAngleFromStart *= 0.5;
    const float sinAngle = sqrt(0.5-cosAngleFromStart);
    const float cosAngle = sqrt(0.5+cosAngleFromStart);
    
    planeNormal *= sinAngle;
    const vec3 precompPart = cross(planeNormal,start)*2.0;

    return start+precompPart*cosAngle+cross(planeNormal,precompPart);
}

// @Crisspl move this to `irr/builtin/glsl/shapes/triangle.glsl`

// returns solid angle of a spherical triangle
// WARNING: can and will return NAN if one or three of the triangle edges are near zero length
// this function is beyond optimized.
float irr_glsl_SolidAngleOfTriangle(in vec3 A, in vec3 B, in vec3 C, out vec3 cos_vertices, out vec3 sin_vertices, out float cosC, out float cscB)
{    
    // The sides are denoted by lower-case letters a, b, and c. On the unit sphere their lengths are numerically equal to the radian measure of the angles that the great circle arcs subtend at the centre. The sides of proper spherical triangles are (by convention) less than PI
    const vec3 cos_sides = vec3(dot(B,C),dot(C,A),dot(A,B));
    const vec3 csc_sides = inversesqrt(vec3(1.0)-cos_sides*cos_sides);

    // these variables might eventually get optimized out
    cosC = cos_sides[2];
    cscB = csc_sides[1];
    
    // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or, equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
    cos_vertices = (cos_sides-cos_sides.yzx*cos_sides.zxy)*csc_sides.yzx*csc_sides.zxy; // using Spherical Law of Cosines
    sin_vertices = sqrt(vec3(1.0)-cos_vertices*cos_vertices);

    // sorry about the naming of `something` I just can't seem to be able to give good name to the variables that is consistent with semantics
	const bool something0 = cos_vertices[0]<-cos_vertices[1];
    const float cosSumAB = cos_vertices[0]*cos_vertices[1]-sin_vertices[0]*sin_vertices[1];
	const bool something1 = cosSumAB<(-cos_vertices[2]);
	const bool something2 = cosSumAB<cos_vertices[2];
	// apply triple angle formula
	const float absArccosSumABC = acos(cosSumAB*cos_vertices[2]-(cos_vertices[0]*sin_vertices[1]+sin_vertices[0]*cos_vertices[1])*sin_vertices[2]);
	return ((something0 ? something2:something1) ? (-absArccosSumABC):absArccosSumABC)+(something0||something1 ? irr_glsl_PI:(-irr_glsl_PI));
}
// returns solid angle of a triangle given by its world-space vertices and world-space viewing position
float irr_glsl_SolidAngleOfTriangle(in mat3 vertices, in vec3 origin)
{
    // the `normalize` cannot be optimized out
    const vec3 A = normalize(vertices[0]-origin);
    const vec3 B = normalize(vertices[1]-origin);
    const vec3 C = normalize(vertices[2]-origin);

    vec3 dummy0,dummy1;
    float dummy2,dummy3;
    return irr_glsl_SolidAngleOfTriangle(A,B,C,dummy0,dummy1,dummy2,dummy3);
}
// WARNING: can and will return NAN if one or three of the triangle edges are near zero length
// this function could use some more optimizing
vec3 irr_glsl_sampling_generateSphericalTriangleSample(out float rcpPdf, in mat3 vertices, in vec3 origin, in vec2 u)
{
    // the `normalize` cannot be optimized out
    const vec3 A = normalize(vertices[0]-origin);
    const vec3 B = normalize(vertices[1]-origin);
    const vec3 C = normalize(vertices[2]-origin);

    // for angles between view-to-vertex vectors
    float cosC,cscB;
    // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or, equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
    vec3 cos_vertices,sin_vertices;
    // get solid angle, which is also the reciprocal of the probability
    rcpPdf = irr_glsl_SolidAngleOfTriangle(A,B,C,cos_vertices,sin_vertices,cosC,cscB);

    // this part literally cannot be optimized further
    float negSinSubSolidAngle,negCosSubSolidAngle;
    irr_glsl_sincos(rcpPdf*u.x-irr_glsl_PI,negSinSubSolidAngle,negCosSubSolidAngle);

    // TODO: we could optimize everything up and including to the first slerp, because precision here is just godawful
	const float p = negCosSubSolidAngle*sin_vertices[0]-negSinSubSolidAngle*cos_vertices[0];
	const float q = -negSinSubSolidAngle*sin_vertices[0]-negCosSubSolidAngle*cos_vertices[0];

	float u_ = q - cos_vertices[0];
	float v_ = p + sin_vertices[0]*cosC;

	const float cosAngleAlongAC = clamp(((v_*q - u_*p)*cos_vertices[0] - v_) / ((v_*p + u_*q)*sin_vertices[0]), -1.0, 1.0); // TODO: get rid of this clamp (by improving the precision here)

	vec3 C_s = irr_glsl_slerp_impl_impl(A, C*cscB, cosAngleAlongAC);

    const float cosBC_s = dot(C_s,B);
	const float cosAngleAlongBC_s = cosBC_s*u.y - u.y + 1.0;

	return irr_glsl_slerp_impl_impl(B, C_s*inversesqrt(1.0-cosBC_s*cosBC_s), cosAngleAlongBC_s);
}
// End-of @Crisspl move this to `irr/builtin/glsl/shapes/triangle.glsl`

// the interaction here is the interaction at the illuminator-end of the ray, not the receiver
vec3 irr_glsl_light_deferred_eval_and_prob(out float pdf, in vec3 origin, in float intersectionT, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in Light light)
{
    // we don't have to worry about solid angle of the light w.r.t. surface of the light because this function only ever gets called from closestHit routine, so such ray cannot be produced
    pdf = scene_getLightChoicePdf(light);

    Triangle tri = triangles[Light_getObjectID(light)];
#if TRIANGLE_METHOD==0
    pdf *= intersectionT*intersectionT/abs(dot(Triangle_getNormalTimesArea(tri),interaction.isotropic.V.dir));
#elif TRIANGLE_METHOD==1
    float rcpProb = irr_glsl_SolidAngleOfTriangle(mat3(tri.vertex0,tri.vertex1,tri.vertex2),origin);
    pdf /= isnan(rcpProb) ? 0.0:rcpProb;
#elif TRIANGLE_METHOD==2
    pdf /= Triangle_getApproxProjSolidAngle(tri,origin,interaction.isotropic.V.dir);
#endif
    return Light_getRadiance(light);
}


irr_glsl_LightSample irr_glsl_light_generate_and_remainder_and_pdf(out vec3 remainder, out float pdf, out float newRayMaxT, in vec3 origin, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u, in int depth)
{
    // normally we'd pick from set of lights, using `u.z`
    const Light light = lights[0];
    const float choicePdf = scene_getLightChoicePdf(light);

    const Triangle tri = triangles[Light_getObjectID(light)];
    
#if TRIANGLE_METHOD==0
    const mat2x3 edges = mat2x3(tri.vertex1-tri.vertex0,tri.vertex2-tri.vertex0);
    const float sqrtU = sqrt(u.x);
    vec3 point = tri.vertex0+edges[0]*(1.0-sqrtU)+edges[1]*sqrtU*u.y;
    vec3 L = point-origin;
    
    const float distanceSq = dot(L,L);
    const float rcpDistance = inversesqrt(distanceSq);
    L *= rcpDistance;

    const float dist = 1.0/rcpDistance;
    
    const float rcpPdf = abs(dot(Triangle_getNormalTimesArea_impl(edges),L))/(distanceSq*choicePdf);
#elif TRIANGLE_METHOD==1
    float rcpPdf;
    const vec3 L = irr_glsl_sampling_generateSphericalTriangleSample(rcpPdf,mat3(tri.vertex0,tri.vertex1,tri.vertex2),origin,u.xy);
    rcpPdf = isnan(rcpPdf) ? 0.0:rcpPdf;

    const vec3 N = Triangle_getNormalTimesArea(tri);
    const float dist = dot(N,tri.vertex0-origin)/dot(N,L);
#elif TRIANGLE_METHOD==2
    const float rcpPdf = Triangle_getApproxProjSolidAngle(tri,origin,interaction.isotropic.V.dir);
#endif

    remainder = Light_getRadiance(light)*rcpPdf;
    pdf = 1.0/rcpPdf;

    newRayMaxT = getEndTolerance(depth)*dist;
    
    return irr_glsl_createLightSample(L,interaction);
}

void closestHitProgram(in ImmutableRay_t _immutable, inout irr_glsl_xoroshiro64star_state_t scramble_state)
{
    const MutableRay_t mutable = rayStack[stackPtr]._mutable;

    vec3 intersection = _immutable.origin+_immutable.direction*mutable.intersectionT;
    const uint objectID = mutable.objectID;
    
    uint bsdfLightIDs;
    irr_glsl_AnisotropicViewSurfaceInteraction interaction;
    {
        irr_glsl_IsotropicViewSurfaceInteraction isotropic;

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

        interaction = irr_glsl_calcAnisotropicInteraction(isotropic);
    }

    const uint lightID = bitfieldExtract(bsdfLightIDs,16,16);

    vec3 throughput = rayStack[stackPtr]._payload.throughput;
    // finish MIS
    if (lightID!=INVALID_ID_16BIT) // has emissive
    {
        float lightPdf;
        vec3 lightVal = irr_glsl_light_deferred_eval_and_prob(lightPdf,_immutable.origin,mutable.intersectionT,interaction,lights[lightID]);
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