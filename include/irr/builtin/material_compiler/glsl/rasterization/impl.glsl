#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_

#include <irr/builtin/material_compiler/glsl/common.glsl>

void instr_eval_execute(in instr_t instr, in MC_precomputed_t precomp, inout irr_glsl_LightSample s, inout MC_microfacet_t _microfacet, in bool skip)
{
	const uint op = instr_getOpcode(instr);
	const bool is_bxdf = op_isBXDF(op);
	const bool is_bsdf = !op_isBRDF(op); //note: true for everything besides BRDF ops (combiners, SET_GEOM_NORMAL and BUMPMAP too)
	const float cosFactor = irr_glsl_conditionalAbsOrMax(is_bsdf, s.NdotL, 0.0);
	const float NdotV = irr_glsl_conditionalAbsOrMax(is_bsdf, currInteraction.inner.isotropic.NdotV, 0.0);
	const bool positiveCosFactors = (cosFactor > FLT_MIN) && (NdotV > FLT_MIN);
	const bool is_bxdf_or_combiner = op_isBXDForCoatOrBlend(op);

	uvec3 regs = instr_decodeRegisters(instr);
	mat2x3 ior;
	mat2x3 ior2;
	params_t params;
	MC_microfacet_t microfacet;
	bsdf_data_t bsdf_data;

	const bool run = !skip && positiveCosFactors;

	if (run && is_bxdf_or_combiner)
	{
		bsdf_data = fetchBSDFDataForInstr(instr);
		ior = bsdf_data_decodeIoR(bsdf_data, op);
		ior2 = matrixCompMult(ior, ior);
		params = instr_getParameters(instr, bsdf_data);
	}

	bxdf_eval_t result = bxdf_eval_t(0.0);

	if (run && is_bxdf)
	{
		const float eta = colorToScalar(ior[0]);
		const float rcp_eta = 1.0 / eta;

		bool refraction = false;
		bool is_valid = true;
#ifdef OP_DIELECTRIC
		if (op == OP_DIELECTRIC && irr_glsl_isTransmissionPath(currInteraction.inner.isotropic.NdotV, s.NdotL))
		{
			is_valid = irr_glsl_calcAnisotropicMicrofacetCache(microfacet.inner, true, currInteraction.inner.isotropic.V.dir, s.L, currInteraction.inner.T, currInteraction.inner.B, currInteraction.inner.isotropic.N, s.NdotL, s.VdotL, eta, rcp_eta);
			finalizeMicrofacet(microfacet);
			refraction = true;
		}
		else
#endif
		{
			microfacet = _microfacet;
		}

		const vec3 albedo = params_getReflectance(params);
		const float a = params_getAlpha(params);
		const float a2 = a*a;

#if defined(OP_DIFFUSE) || defined(OP_DIFFTRANS)
		if (op_isDiffuse(op))
		{
			result = albedo * (is_bsdf ? 0.5 : 1.0) * irr_glsl_oren_nayar_cos_eval_wo_clamps(a2, s.VdotL, cosFactor, NdotV);
		}
		else
#endif
#if defined(OP_CONDUCTOR) || defined(OP_DIELECTRIC)
		if (is_valid && a > ALPHA_EPSILON)
		{
			float bxdf_eval_scalar_part;
			const uint ndf = instr_getNDF(instr);

#ifndef ALL_ISOTROPIC_BXDFS
			const float ay = params_getAlphaV(params);
			const float ay2 = ay*ay;
#endif
			const float NdotL = cosFactor;
			const float NdotL2 = s.NdotL2;
#ifndef ALL_ISOTROPIC_BXDFS
			const float TdotL2 = s.TdotL * s.TdotL;
			const float BdotL2 = s.BdotL * s.BdotL;
#endif

			const float NdotV2 = currInteraction.inner.isotropic.NdotV_squared;
#ifndef ALL_ISOTROPIC_BXDFS
			const float TdotV2 = currInteraction.TdotV2;
			const float BdotV2 = currInteraction.BdotV2;
#endif

			const float NdotH = microfacet.inner.isotropic.NdotH;
			const float NdotH2 = microfacet.inner.isotropic.NdotH2;
#ifndef ALL_ISOTROPIC_BXDFS
			const float TdotH2 = microfacet.TdotH2;
			const float BdotH2 = microfacet.BdotH2;
#endif

			BEGIN_CASES(ndf)
#ifdef NDF_GGX
				CASE_BEGIN(ndf, NDF_GGX) {
#ifdef ALL_ISOTROPIC_BXDFS
					bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_cos_eval_DG_wo_clamps(NdotH2, NdotL, NdotL2, NdotV, NdotV2, a2);
#else
					bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_aniso_cos_eval_DG_wo_clamps(NdotH2, TdotH2, BdotH2, NdotL, NdotL2, TdotL2, BdotL2, NdotV, NdotV2, TdotV2, BdotV2, a, a2, ay, ay2);
#endif
				} CASE_END
#endif

#ifdef NDF_BECKMANN
				CASE_BEGIN(ndf, NDF_BECKMANN) {
#ifdef ALL_ISOTROPIC_BXDFS
					bxdf_eval_scalar_part = irr_glsl_beckmann_height_correlated_cos_eval_DG_wo_clamps(NdotH2, NdotL2, NdotV2, a2);
#else
					bxdf_eval_scalar_part = irr_glsl_beckmann_aniso_height_correlated_cos_eval_DG_wo_clamps(NdotH2, TdotH2, BdotH2, NdotL2, TdotL2, BdotL2, NdotV2, TdotV2, BdotV2, a, a2, ay, ay2);
#endif
				} CASE_END
#endif

#ifdef NDF_PHONG
				CASE_BEGIN(ndf, NDF_PHONG) {
					float n = irr_glsl_alpha2_to_phong_exp(a2);
#ifdef ALL_ISOTROPIC_BXDFS
					bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(NdotH, NdotV2, NdotL2, n, a2);
#else
					float ny = irr_glsl_alpha2_to_phong_exp(ay2);
					bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(NdotH, NdotH2, TdotH2, BdotH2, TdotL2, BdotL2, TdotV2, BdotV2, NdotV2, NdotL2, n, ny, a2, ay2);
#endif
				} CASE_END
#endif

				CASE_OTHERWISE
				{} //else "empty braces"
			END_CASES

				float VdotH = microfacet.inner.isotropic.VdotH;
				vec3 fr;
#ifdef OP_CONDUCTOR
				if (op == OP_CONDUCTOR)
					fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
				else
#endif
					fr = irr_glsl_fresnel_dielectric_common(ior2[0], VdotH);

#ifdef OP_DIELECTRIC
				if (is_bsdf)
				{
					float LdotH = microfacet.inner.isotropic.LdotH;
					float VdotHLdotH = VdotH * LdotH;
#ifdef NDF_GGX
					if (ndf == NDF_GGX)
						bxdf_eval_scalar_part = irr_glsl_ggx_microfacet_to_light_measure_transform(bxdf_eval_scalar_part, NdotL, refraction, VdotH, LdotH, VdotHLdotH, eta);
					else
#endif
						bxdf_eval_scalar_part = irr_glsl_microfacet_to_light_measure_transform(bxdf_eval_scalar_part, NdotV, refraction, VdotH, LdotH, VdotHLdotH, eta);
				}
#endif
				result = fr * bxdf_eval_scalar_part;
		} else
#endif
		{}
	}

	if (!is_bxdf)
	{
		mat2x4 srcs = instr_fetchSrcRegs(instr, regs);

		BEGIN_CASES(op)
#ifdef OP_COATING
		CASE_BEGIN(op, OP_COATING) {
			float dummy;
			result = instr_execute_cos_eval_COATING(instr, srcs, params, ior[0], ior2[0], s, bsdf_data, dummy);
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
	irr_glsl_LightSample s = irr_glsl_createLightSample(L, currInteraction.inner);
	MC_microfacet_t microfacet;
	microfacet.inner = irr_glsl_calcAnisotropicMicrofacetCache(currInteraction.inner, s);
	finalizeMicrofacet(microfacet);
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);
		const uint op = instr_getOpcode(instr);

		bool skip = false;
#ifdef OP_THINDIELECTRIC
		skip = skip || (op == OP_THINDIELECTRIC);
#endif
#ifdef OP_DELTATRANS
		skip = skip || (op == OP_DELTATRANS);
#endif

		instr_eval_execute(instr, precomp, s, microfacet, skip);

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
			updateLightSampleAfterNormalChange(s);
			updateMicrofacetCacheAfterNormalChange(s, microfacet);
		}
#endif
	}
	return readReg3(0u);//result is always in regs 0,1,2
}

#endif