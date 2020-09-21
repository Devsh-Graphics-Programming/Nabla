#ifndef _IRR_BUILTIN_GLSL_BXDF_COMMON_SAMPLES_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_COMMON_SAMPLES_INCLUDED_

#include <irr/builtin/glsl/math/functions.glsl>
#include <irr/builtin/glsl/bxdf/common.glsl>

irr_glsl_LightSample irr_glsl_transmission_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    return irr_glsl_createLightSample(-interaction.isotropic.V.dir,-1.0,interaction.T,interaction.B,interaction.isotropic.N);
}

/** Why don't we check that the incoming and outgoing directions equal each other (or similar for other delta distributions such as reflect, or smooth [thin] dielectrics:
- The `remainder_and_pdf` functions are meant to be used with MIS
- Our own generator can never pick an improbable path, so no checking necessary
- For other generators the estimator will be `f_BSDF*f_Light*f_Visibility*clampedCos(theta)/(1+(p_BSDF^alpha+p_otherNonChosenGenerator^alpha+...)/p_ChosenGenerator^alpha)`
	when `p_BSDF` equals `FLT_INF` it will drive the overall MIS estimator for the other generators to 0 so no checking necessary
**/
float irr_glsl_transmission_cos_remainder_and_pdf(out float pdf)
{
	pdf = 1.0/0.0;
	return 1.0;
}

irr_glsl_LightSample irr_glsl_reflection_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    const vec3 L = irr_glsl_reflect(interaction.isotropic.V.dir,interaction.isotropic.N,interaction.isotropic.NdotV);
    return irr_glsl_createLightSample(L,interaction);
}

// for information why we don't check the relation between `V` and `L` or `N` and `H`, see comments for `irr_glsl_transmission_cos_remainder_and_pdf`
float irr_glsl_reflection_cos_remainder_and_pdf(out float pdf)
{
	pdf = 1.0/0.0;
	return 1.0;
}

#endif
