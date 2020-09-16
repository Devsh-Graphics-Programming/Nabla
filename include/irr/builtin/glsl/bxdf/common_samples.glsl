#ifndef _IRR_BUILTIN_GLSL_BXDF_COMMON_SAMPLES_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_COMMON_SAMPLES_INCLUDED_

#include <irr/builtin/glsl/math/functions.glsl>
#include <irr/builtin/glsl/bxdf/common.glsl>

irr_glsl_LightSample irr_glsl_transmission_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    return irr_glsl_createLightSample(-interaction.isotropic.V.dir,-1.0,interaction);
}

float irr_glsl_transmission_cos_remainder_and_pdf(out float pdf, in irr_glsl_LightSample s)
{
	pdf = 1.0/0.0;
	return 1.0;
}

irr_glsl_LightSample irr_glsl_reflection_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    const vec3 L = irr_glsl_reflect(interaction.isotropic.V.dir,interaction.isotropic.N,interaction.isotropic.NdotV);
    return irr_glsl_createLightSample(L,interaction);
}

float irr_glsl_reflection_cos_remainder_and_pdf(out float pdf, in irr_glsl_LightSample s)
{
	pdf = 1.0/0.0;
	return 1.0;
}

#endif
