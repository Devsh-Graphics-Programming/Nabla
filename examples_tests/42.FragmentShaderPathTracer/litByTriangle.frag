// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
#extension GL_GOOGLE_include_directive : require

#define TRIANGLE_METHOD 2 // 0 area sampling, 1 solid angle sampling, 2 approximate projected solid angle sampling
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
};


#define LIGHT_COUNT 1
Light lights[LIGHT_COUNT] = {
    {vec3(30.0,25.0,15.0),0u}
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

// @Crisspl move this to `irr/builtin/glsl/math.glsl`
// returns `acos(acos(A)+acos(B)+acos(C))-PI` but requires `sinA,sinB,sinC` are all positive
float irr_glsl_getArccosSumofABC_minus_PI(in float cosA, in float cosB, in float cosC, in float sinA, in float sinB, in float sinC)
{
    // sorry about the naming of `something` I just can't seem to be able to give good name to the variables that is consistent with semantics
	const bool something0 = cosA<(-cosB);
    const float cosSumAB = cosA*cosB-sinA*sinB;
	const bool something1 = cosSumAB<(-cosC);
	const bool something2 = cosSumAB<cosC;
	// apply triple angle formula
	const float absArccosSumABC = acos(cosSumAB*cosC-(cosA*sinB+sinA*cosB)*sinC);
	return ((something0 ? something2:something1) ? (-absArccosSumABC):absArccosSumABC)+(something0||something1 ? irr_glsl_PI:(-irr_glsl_PI));
}


// @Crisspl move this to `irr/builtin/glsl/sampling/bilinear.glsl`

float irr_glsl_sampling_generateLinearSample(in vec2 linearCoeffs, in float u)
{
    const float rcpDiff = 1.0/(linearCoeffs[0]-linearCoeffs[1]);
    const vec2 squaredCoeffs = linearCoeffs*linearCoeffs;
    return abs(rcpDiff)<FLT_MAX ? (linearCoeffs[0]-sqrt(mix(squaredCoeffs[0],squaredCoeffs[1],u)))*rcpDiff:u;
}

// The square's vertex values are defined in Z-order, so indices 0,1,2,3 (xyzw) correspond to (0,0),(1,0),(0,1),(1,1)
vec2 irr_glsl_sampling_generateBilinearSample(out float rcpPdf, in vec4 bilinearCoeffs, vec2 u)
{
    const vec2 twiceAreasUnderXCurve = vec2(bilinearCoeffs[0]+bilinearCoeffs[1],bilinearCoeffs[2]+bilinearCoeffs[3]);
    u.y = irr_glsl_sampling_generateLinearSample(twiceAreasUnderXCurve,u.y);

    const vec2 ySliceEndPoints = vec2(mix(bilinearCoeffs[0],bilinearCoeffs[2],u.y),mix(bilinearCoeffs[1],bilinearCoeffs[3],u.y));
    u.x = irr_glsl_sampling_generateLinearSample(ySliceEndPoints,u.x);

    rcpPdf = (twiceAreasUnderXCurve[0]+twiceAreasUnderXCurve[1])/(4.0*mix(ySliceEndPoints[0],ySliceEndPoints[1],u.x));

    return u;
}
float irr_glsl_sampling_probBilinearSample(in vec4 bilinearCoeffs, vec2 u)
{
    return 4.0*mix(mix(bilinearCoeffs[0],bilinearCoeffs[1],u.x),mix(bilinearCoeffs[2],bilinearCoeffs[3],u.x),u.y)/(bilinearCoeffs[0]+bilinearCoeffs[1]+bilinearCoeffs[2]+bilinearCoeffs[3]);
}


// @Crisspl move this to `irr/builtin/glsl/shapes/triangle.glsl`

//
mat3 irr_glsl_shapes_getSphericalTriangle(in mat3 vertices, in vec3 origin)
{
    // the `normalize` cannot be optimized out
    return mat3(normalize(vertices[0]-origin),normalize(vertices[1]-origin),normalize(vertices[2]-origin));
}

// returns solid angle of a spherical triangle
// WARNING: can and will return NAN if one or three of the triangle edges are near zero length
// this function is beyond optimized.
float irr_glsl_shapes_SolidAngleOfTriangle(in mat3 sphericalVertices, out vec3 cos_vertices, out vec3 sin_vertices, out float cos_a, out float cos_c, out float csc_b, out float csc_c)
{    
    // The sides are denoted by lower-case letters a, b, and c. On the unit sphere their lengths are numerically equal to the radian measure of the angles that the great circle arcs subtend at the centre. The sides of proper spherical triangles are (by convention) less than PI
    const vec3 cos_sides = vec3(dot(sphericalVertices[1],sphericalVertices[2]),dot(sphericalVertices[2],sphericalVertices[0]),dot(sphericalVertices[0],sphericalVertices[1]));
    const vec3 csc_sides = inversesqrt(vec3(1.0)-cos_sides*cos_sides);

    // these variables might eventually get optimized out
    cos_a = cos_sides[0];
    cos_c = cos_sides[2];
    csc_b = csc_sides[1];
    csc_c = csc_sides[2];
    
    // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or, equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
    cos_vertices = (cos_sides-cos_sides.yzx*cos_sides.zxy)*csc_sides.yzx*csc_sides.zxy; // using Spherical Law of Cosines
    sin_vertices = sqrt(vec3(1.0)-cos_vertices*cos_vertices);
    
    // the solid angle of a triangle is the sum of its planar vertices' angles minus PI
    return irr_glsl_getArccosSumofABC_minus_PI(cos_vertices[0],cos_vertices[1],cos_vertices[2],sin_vertices[0],sin_vertices[1],sin_vertices[2]);
}
float irr_glsl_shapes_SolidAngleOfTriangle(in mat3 sphericalVertices)
{
    vec3 dummy0,dummy1;
    float dummy2,dummy3,dummy4,dummy5;
    return irr_glsl_shapes_SolidAngleOfTriangle(sphericalVertices,dummy0,dummy1,dummy2,dummy3,dummy4,dummy5);
}
// returns solid angle of a triangle given by its world-space vertices and world-space viewing position
float irr_glsl_shapes_SolidAngleOfTriangle(in mat3 vertices, in vec3 origin)
{
    return irr_glsl_shapes_SolidAngleOfTriangle(irr_glsl_shapes_getSphericalTriangle(vertices,origin));
}


// WARNING: can and will return NAN if one or three of the triangle edges are near zero length
// this function could use some more optimizing
vec3 irr_glsl_sampling_generateSphericalTriangleSample(out float rcpPdf, in mat3 sphericalVertices, in vec2 u)
{
    // for angles between view-to-vertex vectors
    float cos_a,cos_c,csc_b,csc_c;
    // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or, equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
    vec3 cos_vertices,sin_vertices;
    // get solid angle, which is also the reciprocal of the probability
    rcpPdf = irr_glsl_shapes_SolidAngleOfTriangle(sphericalVertices,cos_vertices,sin_vertices,cos_a,cos_c,csc_b,csc_c);

    // this part literally cannot be optimized further
    float negSinSubSolidAngle,negCosSubSolidAngle;
    irr_glsl_sincos(rcpPdf*u.x-irr_glsl_PI,negSinSubSolidAngle,negCosSubSolidAngle);

	const float p = negCosSubSolidAngle*sin_vertices[0]-negSinSubSolidAngle*cos_vertices[0];
	const float q = -negSinSubSolidAngle*sin_vertices[0]-negCosSubSolidAngle*cos_vertices[0];
    
    // TODO: we could optimize everything up and including to the first slerp, because precision here is just godawful
	float u_ = q - cos_vertices[0];
	float v_ = p + sin_vertices[0]*cos_c;

	const float cosAngleAlongAC = clamp(((v_*q - u_*p)*cos_vertices[0] - v_) / ((v_*p + u_*q)*sin_vertices[0]), -1.0, 1.0); // TODO: get rid of this clamp (by improving the precision here)

    // the slerps could probably be optimized by sidestepping `normalize` calls and accumulating scaling factors
	vec3 C_s = irr_glsl_slerp_impl_impl(sphericalVertices[0], sphericalVertices[2]*csc_b, cosAngleAlongAC);

    const float cosBC_s = dot(C_s,sphericalVertices[1]);
	const float cosAngleAlongBC_s = 1.0+cosBC_s*u.y-u.y;

	return irr_glsl_slerp_impl_impl(sphericalVertices[1], C_s*inversesqrt(1.0-cosBC_s*cosBC_s), cosAngleAlongBC_s);
}
vec3 irr_glsl_sampling_generateSphericalTriangleSample(out float rcpPdf, in mat3 vertices, in vec3 origin, in vec2 u)
{
    return irr_glsl_sampling_generateSphericalTriangleSample(rcpPdf,irr_glsl_shapes_getSphericalTriangle(vertices,origin),u);
}


vec4 irr_glsl_sampling_computeBilinearPatchForProjSphericalTriangle(in mat3 sphericalVertices, in vec3 receiverNormal, in bool isBSDF)
{
    // a positive would prevent us from a scenario where `irr_glsl_sampling_rcpProbBilinearSample` will return NAN
    const float minimumProjSolidAngle = 0.0;
    
    // take abs of the value if we have a BSDF, clamp to 0 otherwise
    const vec3 bxdfPdfAtVertex = irr_glsl_conditionalAbsOrMax(isBSDF,transpose(sphericalVertices)*receiverNormal,vec3(minimumProjSolidAngle));

    // the swizzle needs to match the mapping of the [0,1]^2 square to the triangle vertices
    return bxdfPdfAtVertex.yyxz;
}

// There are two different modes of sampling, one for BSDF and one for BRDF (depending if we throw away bottom hemisphere or not)
vec3 irr_glsl_sampling_generateProjectedSphericalTriangleSample(out float rcpPdf, in mat3 sphericalVertices, in vec3 receiverNormal, in bool isBSDF, vec2 u)
{
    // pre-warp according to proj solid angle approximation
    u = irr_glsl_sampling_generateBilinearSample(rcpPdf,irr_glsl_sampling_computeBilinearPatchForProjSphericalTriangle(sphericalVertices,receiverNormal,isBSDF),u);

    // now warp the points onto a spherical triangle
    float solidAngle;
    const vec3 L = irr_glsl_sampling_generateSphericalTriangleSample(solidAngle,sphericalVertices,u);
    rcpPdf *= solidAngle;

    return L;
}
vec3 irr_glsl_sampling_generateProjectedSphericalTriangleSample(out float rcpPdf, in mat3 vertices, in vec3 origin, in vec3 receiverNormal, in bool isBSDF, in vec2 u)
{
    return irr_glsl_sampling_generateProjectedSphericalTriangleSample(rcpPdf,irr_glsl_shapes_getSphericalTriangle(vertices,origin),receiverNormal,isBSDF,u);
}


//
vec2 irr_glsl_sampling_generateSphericalTriangleSampleInverse(out float pdf, in mat3 sphericalVertices, in vec3 L)
{
    // for angles between view-to-vertex vectors
    float cos_a,cos_c,csc_b,csc_c;
    // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or, equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
    vec3 cos_vertices,sin_vertices;
    // get solid angle, which is also the reciprocal of the probability
    pdf = 1.0/irr_glsl_shapes_SolidAngleOfTriangle(sphericalVertices,cos_vertices,sin_vertices,cos_a,cos_c,csc_b,csc_c);

    // get the modified B angle of the first subtriangle by getting it from the triangle formed by vertices A,B and the light sample L
    const float cosAngleAlongBC_s = dot(L,sphericalVertices[1]);
    const float csc_a_ = inversesqrt(1.0-cosAngleAlongBC_s*cosAngleAlongBC_s); // only NaN if L is close to B which implies v=0
    const float cos_b_ = dot(L,sphericalVertices[0]);

    const float cosB_ = (cos_b_-cosAngleAlongBC_s*cos_c)*csc_a_*csc_c; // only NaN if `csc_a_` (L close to B) is NaN OR if `csc_c` is NaN (which would mean zero solid angle triangle to begin with, so uv can be whatever)
    const float sinB_ = sqrt(1.0-cosB_*cosB_);

    // now all that remains is to obtain the modified C angle, which is the angle at the unknown vertex `C_s`
    const float cosC_ = sin_vertices[0]*sinB_*cos_c-cos_vertices[0]*cosB_; // if cosB_ is NaN then cosC_ doesn't matter because the subtriangle has zero Solid Angle (we could pretend its `-cos_vertices[0]`)
    const float sinC_ = sqrt(1.0-cosC_*cosC_);

    const float subTriSolidAngleRatio = irr_glsl_getArccosSumofABC_minus_PI(cos_vertices[0],cosB_,cosC_,sin_vertices[0],sinB_,sinC_)*pdf; // will only be NaN if either the original triangle has zero solid angle or the subtriangle has zero solid angle (all can be satisfied with u=0) 
    const float u = subTriSolidAngleRatio>FLT_MIN ? subTriSolidAngleRatio:0.0; // tiny overruns of u>1.0 will not affect the PDF much because a bilinear warp is used and the gradient has a bound (won't be true if LTC will get used)

    // INF if any angle is 0 degrees, which implies L lays along BA arc, if the angle at A is PI minus the angle at either B_ or C_ while the other of C_ or B_ has a zero angle, we get a NaN (which is also a zero solid angle subtriangle, implying L along AB arc)
    const float cosBC_s = (cos_vertices[0]+cosB_*cosC_)/(sinB_*sinC_);
    // if cosBC_s is really large then we have numerical issues (should be 1.0 which means the arc is really short), if its NaN then either the original or sub-triangle has zero solid angle, in both cases we can consider that the BC_s arc is actually the BA arc and substitute
    const float v = (1.0-cosAngleAlongBC_s)/(1.0-(cosBC_s<uintBitsToFloat(0x3f7fffff) ? cosBC_s:cos_c));

    return vec2(u,v);
}

//
float irr_glsl_sampling_probProjectedSphericalTriangleSample(in mat3 sphericalVertices, in vec3 receiverNormal, in bool receiverWasBSDF, in vec3 L)
{
    float pdf;
    const vec2 u = irr_glsl_sampling_generateSphericalTriangleSampleInverse(pdf,sphericalVertices,L);

    return pdf*irr_glsl_sampling_probBilinearSample(irr_glsl_sampling_computeBilinearPatchForProjSphericalTriangle(sphericalVertices,receiverNormal,receiverWasBSDF),u);
}
// End-of @Crisspl move this to `irr/builtin/glsl/sampling/triangle.glsl`


// the interaction here is the interaction at the illuminator-end of the ray, not the receiver
vec3 irr_glsl_light_deferred_eval_and_prob(
    out float pdf, in Light light, in vec3 L
#if TRIANGLE_METHOD==0
    ,in float intersectionT
#else
    ,in vec3 origin
#if TRIANGLE_METHOD==2
    ,in vec3 normalAtOrigin, in bool wasBSDFAtOrigin
#endif
#endif
)
{
    // we don't have to worry about solid angle of the light w.r.t. surface of the light because this function only ever gets called from closestHit routine, so such ray cannot be produced
    pdf = scene_getLightChoicePdf(light);

    Triangle tri = triangles[Light_getObjectID(light)];
#if TRIANGLE_METHOD==0
    pdf *= intersectionT*intersectionT/abs(dot(Triangle_getNormalTimesArea(tri),L));
#else
    const mat3 sphericalVertices = irr_glsl_shapes_getSphericalTriangle(mat3(tri.vertex0,tri.vertex1,tri.vertex2),origin);
    Triangle tmpTri = Triangle_Triangle(mat3(tri.vertex0,tri.vertex1,tri.vertex2),0u,0u);
    #if TRIANGLE_METHOD==1
        float rcpProb = irr_glsl_shapes_SolidAngleOfTriangle(sphericalVertices);
        // if `rcpProb` is NAN then the triangle's solid angle was close to 0.0 
        pdf = rcpProb>FLT_MIN ? (pdf/rcpProb):FLT_MAX;
    #elif TRIANGLE_METHOD==2
        pdf *= irr_glsl_sampling_probProjectedSphericalTriangleSample(sphericalVertices,normalAtOrigin,wasBSDFAtOrigin,L);
        // if `pdf` is NAN then the triangle's projected solid angle was close to 0.0, if its close to INF then the triangle was very small
        pdf = pdf<FLT_MAX ? pdf:0.0;
    #endif
#endif
    return Light_getRadiance(light);
}


irr_glsl_LightSample irr_glsl_light_generate_and_remainder_and_pdf(out vec3 remainder, out float pdf, out float newRayMaxT, in vec3 origin, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in bool isBSDF, in vec3 u, in int depth)
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
#else 
    float rcpPdf;

    const mat3 sphericalVertices = irr_glsl_shapes_getSphericalTriangle(mat3(tri.vertex0,tri.vertex1,tri.vertex2),origin);
#if TRIANGLE_METHOD==1
    const vec3 L = irr_glsl_sampling_generateSphericalTriangleSample(rcpPdf,sphericalVertices,u.xy);
#elif TRIANGLE_METHOD==2
    const vec3 L = irr_glsl_sampling_generateProjectedSphericalTriangleSample(rcpPdf,sphericalVertices,interaction.isotropic.N,isBSDF,u.xy);
#endif
    // if `rcpProb` is NAN or negative then the triangle's solidAngle or projectedSolidAngle was close to 0.0 
    rcpPdf = rcpPdf>FLT_MIN ? rcpPdf:0.0;

    const vec3 N = Triangle_getNormalTimesArea(tri);
    const float dist = dot(N,tri.vertex0-origin)/dot(N,L);
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
        vec3 lightVal = irr_glsl_light_deferred_eval_and_prob(
            lightPdf,lights[lightID],_immutable.direction
        #if TRIANGLE_METHOD==0
            ,mutable.intersectionT
        #else
            ,_immutable.origin
        #if TRIANGLE_METHOD==2
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
        vec3 epsilon = rand3d(depth,sampleIx,scramble_state);
    
        float rcpChoiceProb;
        const bool doNEE = irr_glsl_partitionRandVariable(bsdfGeneratorProbability,epsilon.z,rcpChoiceProb);
    

        float maxT;
        // the probability of generating a sample w.r.t. the light generator only possible and used when it was generated with it!
        float lightPdf;
        irr_glsl_LightSample _sample;
        const bool isBSDF = BSDFNode_isBSDF(bsdf);
        if (doNEE)
        {
            vec3 lightRemainder;
            _sample = irr_glsl_light_generate_and_remainder_and_pdf(
                lightRemainder,lightPdf,maxT,
                intersection,interaction,
                isBSDF,epsilon,depth
            );
            throughput *= lightRemainder;
        }
        const vec3 throughputCIE_Y = transpose(irr_glsl_sRGBtoXYZ)[1]*throughput;
        const vec3 luminosityContributionHint = throughputCIE_Y/(throughputCIE_Y.r+throughputCIE_Y.g+throughputCIE_Y.b);
        if (!doNEE)
        {
            maxT = FLT_MAX;
            _sample = irr_glsl_bsdf_cos_generate(interaction,epsilon,bsdf,luminosityContributionHint);
        }
            
        // do a cool trick and always compute the bsdf parts this way! (no divergence)
        float bsdfPdf;
        // the value of the bsdf divided by the probability of the sample being generated
        throughput *= irr_glsl_bsdf_cos_remainder_and_pdf(bsdfPdf,_sample,interaction,bsdf,luminosityContributionHint);

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
            rayStack[stackPtr]._immutable.normalAtOrigin = interaction.isotropic.N;
            rayStack[stackPtr]._immutable.wasBSDFAtOrigin = isBSDF;
            stackPtr++;
        }
    }
}