#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_

#include <irr/builtin/material_compiler/glsl/common.glsl>

void instr_eval_execute(in instr_t instr, in MC_precomputed_t precomp, inout irr_glsl_LightSample s, inout irr_glsl_AnisotropicMicrofacetCache _microfacet)
{
	const uint op = instr_getOpcode(instr);
	const bool is_bsdf = !op_isBRDF(op); //note it actually tells if op is BSDF or BUMPMAP or SET_GEOM_NORMAL (divergence reasons)
	const float cosFactor = irr_glsl_conditionalAbsOrMax(is_bsdf, s.NdotL, 0.0);

	uvec3 regs = instr_decodeRegisters(instr);
	mat2x3 ior;
	mat2x3 ior2;
	params_t params;
	irr_glsl_AnisotropicMicrofacetCache microfacet;
	bsdf_data_t bsdf_data;

	bxdf_eval_t bxdf_eval = bxdf_eval_t(0.0);

	if (cosFactor > FLT_MIN)
	{
		//speculative execution
		bsdf_data = fetchBSDFDataForInstr(instr);
		ior = bsdf_data_decodeIoR(bsdf_data, op);
		ior2 = matrixCompMult(ior, ior);
		params = instr_getParameters(instr, bsdf_data);
		const float ior_scalar = dot(CIE_XYZ_Luma_Y_coeffs, ior[0]);
		float bxdf_eval_scalar_part;
		uint ndf = instr_getNDF(instr);
		float a = params_getAlpha(params);
		float a2 = a*a;
		float ay = params_getAlphaV(params);
		float ay2 = ay*ay;
		const vec3 refl = params_getReflectance(params);
		const vec3 trans = params_getTransmittance(params);

		bool is_valid = op_isBXDF(op);
		bool refraction = false;
#ifndef NO_BSDF
		//here actually using stronger check for BSDF because it's probably worth it
		if (op_isBSDF(op) && irr_glsl_isTransmissionPath(currInteraction.isotropic.NdotV, s.NdotL))
		{
			is_valid = irr_glsl_calcAnisotropicMicrofacetCache(microfacet, true, currInteraction.isotropic.V.dir, s.L, currInteraction.T, currInteraction.B, currInteraction.isotropic.N, s.NdotL, s.VdotL, ior_scalar, 1.0/ior_scalar);
			refraction = true;
		}
		else
#endif
		{
			microfacet = _microfacet;
		}

		if (is_valid)
		{
#if defined(OP_DIFFUSE) || defined(OP_DIFFTRANS)
			if (op_isDiffuse(op))
			{
				vec3 reflectance = is_bsdf ? trans : refl;
				float alpha2 = is_bsdf ? 0.0 : a2;
				bxdf_eval = reflectance * irr_glsl_oren_nayar_cos_eval(s, currInteraction.isotropic, alpha2);
			} else
#endif
#if defined(OP_CONDUCTOR) || defined(OP_DIELECTRIC)
			if (op_hasSpecular(op))
			{
			BEGIN_CASES(ndf)
#ifdef NDF_GGX
				CASE_BEGIN(ndf, NDF_GGX) {
#ifdef ALL_ISOTROPIC_BXDFS
					bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_cos_eval_DG(s, microfacet.isotropic, currInteraction.isotropic, a2);
#else
					bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_aniso_cos_eval_DG(s, microfacet, currInteraction, a, a2, ay, ay2);
#endif
				} CASE_END
#endif

#ifdef NDF_BECKMANN
				CASE_BEGIN(ndf, NDF_BECKMANN) {
#ifdef ALL_ISOTROPIC_BXDFS
					bxdf_eval_scalar_part = irr_glsl_beckmann_height_correlated_cos_eval_DG(s, microfacet.isotropic, currInteraction.isotropic, a2);
#else
					bxdf_eval_scalar_part = irr_glsl_beckmann_aniso_height_correlated_cos_eval_DG(s, microfacet, currInteraction, a, a2, ay, ay2);
#endif
				} CASE_END
#endif

#ifdef NDF_PHONG
				CASE_BEGIN(ndf, NDF_PHONG) {
					float n = irr_glsl_alpha2_to_phong_exp(a2);
#ifdef ALL_ISOTROPIC_BXDFS
					bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(s, microfacet.isotropic, currInteraction.isotropic, n, a2);
#else
					float ny = irr_glsl_alpha2_to_phong_exp(ay2);
					bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(s, microfacet, currInteraction, n, ny, a2, ay2);
#endif
				} CASE_END
#endif

				CASE_OTHERWISE
				{} //else "empty braces"
			END_CASES

				float VdotH = microfacet.isotropic.VdotH;
				float VdotH_clamp = irr_glsl_conditionalAbsOrMax(is_bsdf, VdotH, 0.0);
				vec3 fr;
#ifdef OP_CONDUCTOR
				if (op == OP_CONDUCTOR)
					fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH_clamp);
				else
#endif
					fr = irr_glsl_fresnel_dielectric_common(ior2[0], VdotH_clamp);

				const float NdotL = cosFactor;
				const float NdotV = irr_glsl_conditionalAbsOrMax(is_bsdf, currInteraction.isotropic.NdotV, 0.0);
#ifndef NO_BSDF
				if (is_bsdf)
				{
					float eta = ior_scalar;

					float LdotH = microfacet.isotropic.LdotH;
					float VdotHLdotH = VdotH * LdotH;
#ifdef NDF_GGX
					if (ndf == NDF_GGX)
						bxdf_eval_scalar_part = irr_glsl_ggx_microfacet_to_light_measure_transform(bxdf_eval_scalar_part, NdotL, refraction, VdotH, LdotH, VdotHLdotH, eta);
					else
#endif
						bxdf_eval_scalar_part = irr_glsl_microfacet_to_light_measure_transform(bxdf_eval_scalar_part, NdotV, refraction, VdotH, LdotH, VdotHLdotH, eta);
				}
#endif

				bxdf_eval = (refraction ? (vec3(1.0)-fr):fr) * bxdf_eval_scalar_part;
			} else
#endif
			{}
		}
	}

	bxdf_eval_t result = bxdf_eval;
	if (!op_isBXDF(op))
	{
		mat2x4 srcs = instr_fetchSrcRegs(instr, regs);

		BEGIN_CASES(op)
#ifdef OP_COATING
		CASE_BEGIN(op, OP_COATING) {
			float dummy;
			result = instr_execute_cos_eval_COATING(instr, srcs, params, ior[0], ior2[0], s, microfacet, bsdf_data, dummy);
		} CASE_END
#endif
#ifdef OP_BLEND
		CASE_BEGIN(op, OP_BLEND) {
			result = instr_execute_cos_eval_BLEND(instr, srcs, params, bsdf_data);
		} CASE_END
#endif
#ifdef OP_BUMPMAP
		CASE_BEGIN(op, OP_BUMPMAP) {
			instr_execute_BUMPMAP(instr, srcs, precomp);
		} CASE_END
#endif
#ifdef OP_SET_GEOM_NORMAL
		CASE_BEGIN(op, OP_SET_GEOM_NORMAL) {
			instr_execute_SET_GEOM_NORMAL(instr, precomp);
		} CASE_END
#endif
		CASE_OTHERWISE
		{} //else "empty braces"
		END_CASES
	}

	if (op_isBXDForCoatOrBlend(op))
		writeReg(REG_DST(regs), result);
}

bxdf_eval_t runEvalStream(in MC_precomputed_t precomp, in instr_stream_t stream, in vec3 L)
{
	setCurrInteraction(precomp);
	irr_glsl_LightSample s = irr_glsl_createLightSample(L, currInteraction);
	irr_glsl_AnisotropicMicrofacetCache microfacet = irr_glsl_calcAnisotropicMicrofacetCache(currInteraction, s);
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);
		uint op = instr_getOpcode(instr);

		instr_eval_execute(instr, precomp, s, microfacet);

#if defined(OP_SET_GEOM_NORMAL)||defined(OP_BUMPMAP)
		if (
#ifdef OP_SET_GEOM_NORMAL
			op==OP_SET_GEOM_NORMAL
#ifdef OP_BUMPMAP
			||
#endif
#endif
#ifdef OP_BUMPMAP
			op==OP_BUMPMAP
#endif
		) {
			s = irr_glsl_createLightSample(L, currInteraction);
			microfacet = irr_glsl_calcAnisotropicMicrofacetCache(currInteraction, s);
		}
#endif
	}
	return readReg3(0u);//result is always in regs 0,1,2
}

#endif