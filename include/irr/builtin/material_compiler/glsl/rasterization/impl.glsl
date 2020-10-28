#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_

#include <irr/builtin/material_compiler/glsl/common.glsl>

void instr_eval_execute(in instr_t instr, inout irr_glsl_LightSample s, inout irr_glsl_AnisotropicMicrofacetCache _microfacet)
{
	uint op = instr_getOpcode(instr);

	//speculative execution
	bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
	const mat2x3 ior = bsdf_data_decodeIoR(bsdf_data,op);
	params_t params = instr_getParameters(instr, bsdf_data);
	float bxdf_eval_scalar_part;
	uint ndf = instr_getNDF(instr);
	float a = params_getAlpha(params);
	float a2 = a*a;
#ifndef ALL_ISOTROPIC_BXDFS
	float ay = params_getAlphaV(params);
	float ay2 = ay*ay;
#endif

	const bool is_bsdf = !op_isBRDF(op); //note it actually tells if op is BSDF or BUMPMAP or SET_GEOM_NORMAL (divergence reasons)

#ifndef NO_TWOSIDED
	handleTwosided(instr, s, _microfacet);
#endif

	const float cosFactor = irr_glsl_conditionalAbsOrMax(is_bsdf, s.NdotL, 0.0);

	uvec3 regs = instr_decodeRegisters(instr);

	irr_glsl_AnisotropicMicrofacetCache microfacet;
	bool is_valid = true;
#ifndef NO_BSDF
	//here actually using stronger check for BSDF because it's probably worth it
	if (op_isBSDF(op) && irr_glsl_isTransmissionPath(currInteraction.isotropic.NdotV, s.NdotL))
	{
		float orientedEta, rcpOrientedEta;
		irr_glsl_getOrientedEtas(orientedEta, rcpOrientedEta, currInteraction.isotropic.NdotV, ior[0].x);
		is_valid = irr_glsl_calcAnisotropicMicrofacetCache(microfacet, true, currInteraction.isotropic.V.dir, s.L, currInteraction.T, currInteraction.B, currInteraction.isotropic.N, s.NdotL, s.VdotL, orientedEta, rcpOrientedEta);
	}
	else
#endif
	{
		microfacet = _microfacet;
	}

	if (is_valid && cosFactor>FLT_MIN && op_hasSpecular(op))
	{
		BEGIN_CASES(ndf)
#ifdef NDF_GGX
		CASE_BEGIN(ndf, NDF_GGX) {

#if defined(OP_THINDIELECTRIC) || defined(OP_DIELECTRIC)
			if (is_bsdf) {
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_dielectric_cos_eval(s, currInteraction.isotropic, microfacet.isotropic, ior[0].x, a2);
#else
				bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_dielectric_cos_eval(s, currInteraction.isotropic, microfacet.isotropic, ior[0].x, a2);
#endif
			}
			else
#endif
			{
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_cos_eval_DG(s, microfacet.isotropic, currInteraction.isotropic, a2);
#else
				bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_aniso_cos_eval_DG(s, microfacet, currInteraction, a, a2, ay, ay2);
#endif
			}

		} CASE_END
#endif

#ifdef NDF_BECKMANN
		CASE_BEGIN(ndf, NDF_BECKMANN) {

#if defined(OP_THINDIELECTRIC) || defined(OP_DIELECTRIC)
			if (is_bsdf) {
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_beckmann_smith_height_correlated_dielectric_cos_eval_wo_cache_validation(s, currInteraction, microfacet, ior[0].x, a2);
#else
				bxdf_eval_scalar_part = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_wo_cache_validation(s, currInteraction, microfacet, ior[0].x, a, a2, ay, ay2);
#endif
			}
			else
#endif
			{
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_beckmann_height_correlated_cos_eval_DG(s, microfacet.isotropic, currInteraction.isotropic, a2);
#else
				bxdf_eval_scalar_part = irr_glsl_beckmann_aniso_height_correlated_cos_eval_DG(s, microfacet, currInteraction, a, a2, ay, ay2);
#endif
			}

		} CASE_END
#endif

#ifdef NDF_PHONG
		CASE_BEGIN(ndf, NDF_PHONG) {

#if defined(OP_THINDIELECTRIC) || defined(OP_DIELECTRIC)
			if (is_bsdf) {
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_beckmann_smith_height_correlated_dielectric_cos_eval_wo_cache_validation(s, currInteraction, microfacet, ior[0].x, a2);
#else
				bxdf_eval_scalar_part = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_wo_cache_validation(s, currInteraction, microfacet, ior[0].x, a, a2, ay, ay2);
#endif
			}
			else
#endif
			{
				float n = irr_glsl_alpha2_to_phong_exp(a2);
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(s, microfacet.isotropic, currInteraction.isotropic, n, a2);
#else
				float ny = irr_glsl_alpha2_to_phong_exp(ay2);
				bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(s, microfacet, currInteraction, n, ny, a2, ay2);
#endif
			}

		} CASE_END
#endif

		CASE_OTHERWISE
		{} //else "empty braces"
		END_CASES
	}

	bxdf_eval_t result;
	if ((cosFactor <= FLT_MIN && op_isBXDF(op)) || !is_valid)
	{
		result = bxdf_eval_t(0.0);
	}
	else
	{
		mat2x4 srcs = instr_fetchSrcRegs(instr, regs);

		BEGIN_CASES(op)
#ifdef OP_DIFFUSE
		CASE_BEGIN(op, OP_DIFFUSE) {
			result = instr_execute_cos_eval_DIFFUSE(instr, s, params, bsdf_data);
		} CASE_END
#endif
#ifdef OP_CONDUCTOR
		CASE_BEGIN(op, OP_CONDUCTOR) {
			result = instr_execute_cos_eval_CONDUCTOR(instr, ior, s, microfacet, bxdf_eval_scalar_part, params, bsdf_data);
		} CASE_END
#endif
#ifdef OP_COATING
		CASE_BEGIN(op, OP_COATING) {
			result = instr_execute_cos_eval_COATING(instr, srcs, params, ior[0], s, microfacet, bxdf_eval_scalar_part, bsdf_data);
		} CASE_END
#endif
#ifdef OP_DIFFTRANS
		CASE_BEGIN(op, OP_DIFFTRANS) {
			result = instr_execute_cos_eval_DIFFTRANS(instr, s, params, bsdf_data);
		} CASE_END
#endif
#ifdef OP_DIELECTRIC
		CASE_BEGIN(op, OP_DIELECTRIC) {
			result = instr_execute_cos_eval_DIELECTRIC(instr, s, bxdf_eval_scalar_part);
		} CASE_END
#endif
#ifdef OP_THINDIELECTRIC
		CASE_BEGIN(op, OP_THINDIELECTRIC) {
			result = instr_execute_cos_eval_THINDIELECTRIC(instr, s, params, bsdf_data);
		} CASE_END
#endif
#ifdef OP_BLEND
		CASE_BEGIN(op, OP_BLEND) {
			result = instr_execute_cos_eval_BLEND(instr, srcs, params, bsdf_data);
		} CASE_END
#endif
#ifdef OP_BUMPMAP
		CASE_BEGIN(op, OP_BUMPMAP) {
			instr_execute_BUMPMAP(instr, srcs);
		} CASE_END
#endif
#ifdef OP_SET_GEOM_NORMAL
		CASE_BEGIN(op, OP_SET_GEOM_NORMAL) {
			instr_execute_SET_GEOM_NORMAL(instr);
		} CASE_END
#endif
		CASE_OTHERWISE
		{} //else "empty braces"
		END_CASES
	}

	if (op_isBXDForBlend(op))
		writeReg(REG_DST(regs), result);
}

bxdf_eval_t runEvalStream(in instr_stream_t stream, in vec3 L)
{
	instr_execute_SET_GEOM_NORMAL();
	irr_glsl_LightSample s = irr_glsl_createLightSample(L, currInteraction);
	irr_glsl_AnisotropicMicrofacetCache microfacet = irr_glsl_calcAnisotropicMicrofacetCache(currInteraction, s);
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);
		uint op = instr_getOpcode(instr);

		instr_eval_execute(instr, s, microfacet);

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