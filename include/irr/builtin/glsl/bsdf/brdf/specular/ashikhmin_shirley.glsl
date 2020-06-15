#ifndef _IRR_BSDF_BRDF_SPECULAR_ASHIKHMIN_SHIRLEY_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_ASHIKHMIN_SHIRLEY_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/ndf/ashikhmin_shirley.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/fresnel/fresnel.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/geom/smith.glsl>

//n is 2 phong-like exponents for anisotropy, can be defined as vec2(1.0/at, 1.0/ab) where at is roughness along tangent direction and ab is roughness along bitangent direction
//sin_cos_phi is sin and cos of azimuth angle of half vector
vec3 irr_glsl_ashikhmin_shirley_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in vec2 n, in vec2 sin_cos_phi, in vec2 atb, in mat2x3 ior2)
{
    float ndf = irr_glsl_ashikhmin_shirley(params.isotropic.NdotL, params.isotropic.interaction.NdotV, params.isotropic.NdotH, params.isotropic.VdotH, n, sin_cos_phi);
    vec3 fr = irr_glsl_fresnel_conductor(ior2[0], ior2[1], params.isotropic.VdotH);
    //using ggx smith shadowing term here is wrong, however for now we're doing it because of lack of any other compatible one
    //Ashikhmin and Shirley came up with their own shadowing term, however implementation of it would be too complex in terms of our current design (https://www.researchgate.net/publication/220721563_A_microfacet-based_BRDF_generator)
    float g = irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(atb.x, atb.y, params.TdotL, params.TdotV, params.BdotL, params.BdotV, params.isotropic.NdotL, params.isotropic.interaction.NdotV);

    return g*ndf*fr / (4.0 * params.isotropic.interaction.NdotV);
}

#endif
