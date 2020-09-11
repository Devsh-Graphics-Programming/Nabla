#ifndef _IRR_BSDF_COMMON_SAMPLES_INCLUDED_
#define _IRR_BSDF_COMMON_SAMPLES_INCLUDED_

#include <irr/builtin/glsl/math/functions.glsl>

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_BSDFSample
{
    vec3 L;  // incoming direction, normalized
    float TdotL; 
    float BdotL;
    float NdotL;

    float TdotH;
    float BdotH;
    float NdotH;
    float VdotH;//equal to LdotH (TODO: revise, not true for BSDFs... ugh)
};

// Not optimized for divergence! Use the overload that returns `reflectvity`.
// require H and V already be normalized
// reflection from microfacet
irr_glsl_BSDFSample irr_glsl_createBSDFSample(in vec3 H, in vec3 V, in float VdotH, in mat3 m)
{
    irr_glsl_BSDFSample s;

    vec3 L = irr_glsl_reflect(V, H, VdotH);
    s.L = m * L; // m must be an orthonormal matrix
    s.TdotL = L.x;
    s.BdotL = L.y;
    s.NdotL = L.z;
    s.TdotH = H.x;
    s.BdotH = H.y;
    s.NdotH = H.z;
    s.VdotH = VdotH;

    return s;
}
// Not optimized for divergence! Use the overload that returns `reflectvity`.
// refraction from microfacet
irr_glsl_BSDFSample irr_glsl_createBSDFSample(in vec3 H, in vec3 V, in float VdotH, in mat3 m, in float eta)
{
    irr_glsl_BSDFSample s;

    vec3 L = irr_glsl_refract(V, H, VdotH, eta);
    s.L = m * L; // m must be an orthonormal matrix
    s.TdotL = L.x;
    s.BdotL = L.y;
    s.NdotL = L.z;
    s.TdotH = H.x;
    s.BdotH = H.y;
    s.NdotH = H.z;
    s.VdotH = VdotH;

    return s;
}
#include <irr/builtin/glsl/bxdf/fresnel.glsl>
/* TODO: Have to figure out what to do about VdotH
// reflection or refraction from microfacet
irr_glsl_BSDFSample irr_glsl_createBSDFSample(out vec3 reflectivity, in vec3 H, in vec3 V, in float VdotH, in mat3 m, in float eta)
{
    irr_glsl_BSDFSample s;

    // USE irr_glsl_reflect_refract
    vec3 L = irr_glsl_refract(V, H, VdotH, eta);
    s.L = m * L; // m must be an orthonormal matrix
    s.TdotL = L.x;
    s.BdotL = L.y;
    s.NdotL = L.z;
    s.TdotH = H.x;
    s.BdotH = H.y;
    s.NdotH = H.z;
    s.VdotH = VdotH;

    return s;
}
*/


#include <irr/builtin/glsl/bxdf/common.glsl>

void irr_glsl_updateBSDFParams(out irr_glsl_BSDFIsotropicParams p, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction inter)
{
    p.NdotL = s.NdotL;
    p.NdotL_squared = p.NdotL * p.NdotL;
    p.NdotH = s.NdotH;
    p.VdotH = s.VdotH;
    p.L = s.L;
    p.VdotL = dot(p.L, inter.V.dir);

    p.LplusV_rcpLen = irr_glsl_FLT_INF;
    p.invlenL2 = irr_glsl_FLT_INF;
}
void irr_glsl_updateBSDFParams(out irr_glsl_BSDFAnisotropicParams p, in irr_glsl_BSDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction inter)
{
    irr_glsl_updateBSDFParams(p.isotropic, s, inter.isotropic);

    p.TdotL = s.TdotL;
    p.TdotH = s.TdotH;
    p.BdotL = s.BdotL;
    p.BdotH = s.BdotH;
}


irr_glsl_BSDFSample irr_glsl_transmission_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BSDFSample smpl;
    smpl.L = -interaction.isotropic.V.dir;
    
    return smpl;
}

float irr_glsl_transmission_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s)
{
	pdf = 1.0/0.0;
	return 1.0;
}

irr_glsl_BSDFSample irr_glsl_reflection_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BSDFSample smpl;
    smpl.L = irr_glsl_reflect(interaction.isotropic.V.dir,interaction.isotropic.N,interaction.isotropic.NdotV);
    smpl.NdotH = 1.0; 

    return smpl;
}

float irr_glsl_reflection_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s)
{
	pdf = 1.0/0.0;
	return 1.0;
}

#endif
