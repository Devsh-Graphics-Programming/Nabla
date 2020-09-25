#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_

#include <irr/builtin/material_compiler/glsl/common.glsl>

void instr_eval_execute(in instr_t instr, in irr_glsl_LightSample s, in irr_glsl_AnisotropicMicrofacetCache uf)
{
	uint op = instr_getOpcode(instr);

	//speculative execution
	bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
	mat2x3 ior = bsdf_data_decodeIoR(bsdf_data,op);
	params_t params = instr_getParameters(instr, bsdf_data);
	float bxdf_eval_scalar_part;
	uint ndf = instr_getNDF(instr);
	float a = params_getAlpha(params);
	float a2 = a*a;
#ifndef ALL_ISOTROPIC_BXDFS
	float ay = params_getAlphaV(params);
	float ay2 = ay*ay;
#endif

	const bool is_bsdf = !op_isBRDF(op);
	float cosFactor = is_bsdf ? abs(s.NdotL):max(s.NdotL,0.0);

	if (cosFactor>FLT_MIN && op_hasSpecular(op))
	{
#ifdef NDF_GGX
#ifndef ONLY_ONE_NDF
		if (ndf==NDF_GGX) {
#endif

#if defined(OP_THINDIELECTRIC) || defined(OP_DIELECTRIC)
			if (is_bsdf) {
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_dielectric_cos_eval(s, currInteraction.isotropic, uf.isotropic, ior[0].x, a2);
#else
				bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_dielectric_cos_eval(s, currInteraction.isotropic, uf.isotropic, ior[0].x, a2);
#endif
			}
			else
#endif
			{
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_cos_eval_DG(s, uf.isotropic, currInteraction.isotropic, a2);
#else
				bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_aniso_cos_eval_DG(s, uf, currInteraction, a, a2, ay, ay2);
#endif
			}

#ifndef ONLY_ONE_NDF
		} else
#endif
#endif

#ifdef NDF_BECKMANN
#ifndef ONLY_ONE_NDF
		if (ndf==NDF_BECKMANN) {
#endif

#if defined(OP_THINDIELECTRIC) || defined(OP_DIELECTRIC)
			if (is_bsdf) {
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_beckmann_smith_height_correlated_dielectric_cos_eval_wo_cache_validation(s, currInteraction, uf, ior[0].x, a2);
#else
				bxdf_eval_scalar_part = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_wo_cache_validation(s, currInteraction, uf, ior[0].x, a, a2, ay, ay2);
#endif
			}
			else
#endif
			{
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_beckmann_height_correlated_cos_eval_DG(s, uf.isotropic, currInteraction.isotropic, a2);
#else
				bxdf_eval_scalar_part = irr_glsl_beckmann_aniso_height_correlated_cos_eval_DG(s, uf, currInteraction, a, a2, ay, ay2);
#endif
			}

#ifndef ONLY_ONE_NDF
		} else
#endif
#endif

#ifdef NDF_PHONG
#ifndef ONLY_ONE_NDF
		if (ndf==NDF_PHONG) {
#endif

#if defined(OP_THINDIELECTRIC) || defined(OP_DIELECTRIC)
			if (is_bsdf) {
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_beckmann_smith_height_correlated_dielectric_cos_eval_wo_cache_validation(s, currInteraction, uf, ior[0].x, a2);
#else
				bxdf_eval_scalar_part = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_wo_cache_validation(s, currInteraction, uf, ior[0].x, a, a2, ay, ay2);
#endif
			}
			else
#endif
			{
				float n = irr_glsl_alpha2_to_phong_exp(a2);
#ifdef ALL_ISOTROPIC_BXDFS
				bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(s, uf.isotropic, currInteraction.isotropic, n, a2);
#else
				float ny = irr_glsl_alpha2_to_phong_exp(ay2);
				bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(s, uf, currInteraction, n, ny, a2, ay2);
#endif
			}

#ifndef ONLY_ONE_NDF
		} else
#endif
#endif

#ifndef ONLY_ONE_NDF
		{} //else "empty braces"
#endif
	}

	uvec3 regs = instr_decodeRegisters(instr);
#ifdef OP_DIFFUSE
	if (op==OP_DIFFUSE) {
		instr_execute_cos_eval_DIFFUSE(instr, s, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_CONDUCTOR
	if (op==OP_CONDUCTOR) {
		instr_execute_cos_eval_CONDUCTOR(instr, s, uf, regs, bxdf_eval_scalar_part, params, bsdf_data);
	} else
#endif
#ifdef OP_PLASTIC
	if (op==OP_PLASTIC) {
		instr_execute_cos_eval_PLASTIC(instr, s, uf, regs, bxdf_eval_scalar_part, params, bsdf_data);
	} else
#endif
#ifdef OP_COATING
	if (op==OP_COATING) {
		instr_execute_cos_eval_COATING(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_DIFFTRANS
	if (op==OP_DIFFTRANS) {
		instr_execute_cos_eval_DIFFTRANS(instr, s, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_DIELECTRIC
	if (op==OP_DIELECTRIC) {
		instr_execute_cos_eval_DIELECTRIC(instr, s, regs, bxdf_eval_scalar_part);
	} else
#endif
#ifdef OP_THINDIELECTRIC
	if (op==OP_THINDIELECTRIC) {
		instr_execute_cos_eval_THINDIELECTRIC(instr, s, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_BLEND
	if (op==OP_BLEND) {
		instr_execute_cos_eval_BLEND(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_BUMPMAP
	if (op==OP_BUMPMAP) {
		instr_execute_BUMPMAP(instr);
	} else
#endif
#ifdef OP_SET_GEOM_NORMAL
	if (op==OP_SET_GEOM_NORMAL) {
		instr_execute_SET_GEOM_NORMAL();
	} else
#endif
	{} //else "empty braces"
}

bxdf_eval_t runEvalStream(in instr_stream_t stream, in vec3 L)
{
#ifndef NO_TWOSIDED
	bool ts = false;
#endif
	instr_execute_SET_GEOM_NORMAL();
	irr_glsl_LightSample s = irr_glsl_createLightSample(L, currInteraction);
	//Warning: here using function for reflective case only (TODO)
	irr_glsl_AnisotropicMicrofacetCache uf = irr_glsl_calcAnisotropicMicrofacetCache(currInteraction, s);
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);
		uint op = instr_getOpcode(instr);
#ifndef NO_TWOSIDED
		handleTwosided(ts, instr, s, uf);
#endif
		instr_eval_execute(instr, s, uf);

#if defined(OP_SET_GEOM_NORMAL)||defined(OP_BUMPMAP)
		if (
#ifdef OP_SET_GEOM_NORMAL
			op==OP_SET_GEOM_NORMAL
#endif
#ifdef OP_BUMPMAP
			|| op==OP_BUMPMAP
#endif
		) {
			s = irr_glsl_createLightSample(L, currInteraction);
			//TODO recompute microfacet if isBSDF(instr) && irr_glsl_isTransmissionPath()
			//Warning: here using function for reflective case only
			uf = irr_glsl_calcAnisotropicMicrofacetCache(currInteraction, s);
		}
#endif
	}
	return readReg3(0u);//result is always in regs 0,1,2
}

#endif