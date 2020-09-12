#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_

#include <irr/builtin/material_compiler/glsl/common.glsl>

void instr_eval_execute(in instr_t instr, in vec3 L)
{
	uint op = instr_getOpcode(instr);

	//speculative execution
	bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
	params_t params = instr_getParameters(instr, bsdf_data);
	float bxdf_eval_scalar_part;
	uint ndf = instr_getNDF(instr);
	float a = params_getAlpha(params);
	float a2 = a*a;
#ifndef ALL_ISOTROPIC_BXDFS
	float ay = params_getAlphaV(params);
	float ay2 = ay*ay;
#endif

	float cosFactor = op_isBSDF(op) ? abs(currBSDFParams.isotropic.NdotL):max(currBSDFParams.isotropic.NdotL,0.0);

	if (cosFactor>FLT_MIN && op_hasSpecular(op))
	{
#ifdef NDF_GGX
#ifndef ONLY_ONE_NDF
		if (ndf==NDF_GGX) {
#endif

#ifdef ALL_ISOTROPIC_BXDFS
			bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, a2);
#else
			bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_aniso_cos_eval_DG(currBSDFParams, currInteraction, a, ay);
#endif


#ifndef ONLY_ONE_NDF
		} else
#endif
#endif

#ifdef NDF_BECKMANN
#ifndef ONLY_ONE_NDF
		if (ndf==NDF_BECKMANN) {
#endif

#ifdef ALL_ISOTROPIC_BXDFS
			bxdf_eval_scalar_part = irr_glsl_beckmann_smith_height_correlated_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, a2);
#else
			bxdf_eval_scalar_part = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_DG(currBSDFParams, currInteraction, a, a2, ay, ay2);
#endif

#ifndef ONLY_ONE_NDF
		} else
#endif
#endif

#ifdef NDF_PHONG
#ifndef ONLY_ONE_NDF
		if (ndf==NDF_PHONG) {
#endif


			float n = irr_glsl_alpha2_to_phong_exp(a2);
#ifdef ALL_ISOTROPIC_BXDFS
			bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, n, a2);
#else
			float ny = irr_glsl_alpha2_to_phong_exp(ay2);
			bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(currBSDFParams, currInteraction, n, ny, a2, ay2);
#endif


#ifndef ONLY_ONE_NDF
		} else
#endif
#endif
/*
#ifdef NDF_AS
#ifndef ONLY_ONE_NDF
		if (ndf==NDF_AS) {
#endif
			float nx = irr_glsl_alpha2_to_phong_exp(a2);
			float ny = irr_glsl_alpha2_to_phong_exp(ay2);
			bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(currBSDFParams, currInteraction, nx, ny, a2, ay2);
#ifndef ONLY_ONE_NDF
		} else
#endif
#endif
*/
#ifndef ONLY_ONE_NDF
		{} //else "empty braces"
#endif
	}

	uvec3 regs = instr_decodeRegisters(instr);
#ifdef OP_DIFFUSE
	if (op==OP_DIFFUSE) {
		instr_execute_DIFFUSE(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_CONDUCTOR
	if (op==OP_CONDUCTOR) {
		instr_execute_CONDUCTOR(instr, regs, bxdf_eval_scalar_part, params, bsdf_data);
	} else
#endif
#ifdef OP_PLASTIC
	if (op==OP_PLASTIC) {
		instr_execute_PLASTIC(instr, regs, bxdf_eval_scalar_part, params, bsdf_data);
	} else
#endif
#ifdef OP_COATING
	if (op==OP_COATING) {
		instr_execute_COATING(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_DIFFTRANS
	if (op==OP_DIFFTRANS) {
		instr_execute_DIFFTRANS(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_DIELECTRIC
	if (op==OP_DIELECTRIC) {
		instr_execute_DIELECTRIC(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_BLEND
	if (op==OP_BLEND) {
		instr_execute_BLEND(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_BUMPMAP
	if (op==OP_BUMPMAP) {
		instr_execute_BUMPMAP(instr, L);
	} else
#endif
#ifdef OP_SET_GEOM_NORMAL
	if (op==OP_SET_GEOM_NORMAL) {
		instr_execute_SET_GEOM_NORMAL(L);
	} else
#endif
	{} //else "empty braces"
}

bxdf_eval_t runEvalStream(in instr_stream_t stream, in vec3 L)
{
#ifndef NO_TWOSIDED
	bool ts = false;
#endif
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);
#ifndef NO_TWOSIDED
		handleTwosided(ts, instr);
#endif
		instr_eval_execute(instr, L);
	}
	return readReg3(0u);//result is always in regs 0,1,2
}

//TODO OPTIMIZE THIS FOR MULTIPLE ITERATIONS
vec3 runGeneratorChoiceStream(in instr_stream_t stream, in vec2 rand, out vec3 out_remainder)
{
	uint rescaleChoice = 0u;
	uint ix = 0u;
	instr_t instr = irr_glsl_MC_fetchInstr(stream.offset);
	uint op = instr_getOpcode(instr);
	while (!op_isBXDF(op))
	{
#ifdef OP_BLEND
		if (op==OP_BLEND) {
			bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
			params_t params = instr_getParameters(instr, bsdf_data);
			float w = params_getBlendWeight(params);
			float u = rescaleChoice==0u ? rand.x:rand.y;
			bool choseRight = u>=w;
			u -= choseRight ? w:0.0;
			u /= choseRight ? (1.0-w):w;
			if (rescaleChoice==0u)
				rand.x = u;
			else
				rand.y = u;
			rescaleChoice ^= 0x1u;

			uint right = instr_getRightJump(instr);
			ix = choseRight ? right:(ix+1u);
		}
		else 
#endif //OP_BLEND
		{
#ifdef OP_SET_GEOM_NORMAL
			if (op==OP_SET_GEOM_NORMAL)
			{
				instr_execute_SET_GEOM_NORMAL_interactionOnly();
			} else 
#endif //OP_SET_GEOM_NORMAL
#ifdef OP_BUMPMAP
			if (op==OP_BUMPMAP)
			{
				instr_execute_BUMPMAP_interactionOnly(instr);
			} else
#endif //OP_BUMPMAP
			{}
			ix = ix + 1u;
		}

		instr = irr_glsl_MC_fetchInstr(stream.offset+ix);
		op = instr_getOpcode(instr);
	}

	bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
	params_t params = instr_getParameters(instr, bsdf_data);
	//speculatively
	float ax = params_getAlpha(params);
	float ax2 = ax*ax;
	float ay = params_getAlphaV(params);
	float ay2 = ay*ay;
	mat2x3 ior = bsdf_data_decodeIoR(bsdf_data,op);

#ifndef NO_TWOSIDED
	bool ts_flag = false;
	handleTwosided_interactionOnly(ts_flag, instr);
#endif //NO_TWOSIDED

	uint ndf = instr_getNDF(instr);
	float pdf;
	vec3 L;
#ifdef OP_DIFFUSE
	if (op==OP_DIFFUSE) {
		irr_glsl_BSDFSample s = irr_glsl_cos_weighted_cos_generate(currInteraction, rand);
		out_remainder = irr_glsl_cos_weighted_cos_remainder_and_pdf(pdf, s, currInteraction.isotropic);
		L = s.L;
	} else 
#endif //OP_DIFFUSE

#ifdef OP_DIFFTRANS
	if (op==OP_DIFFTRANS) {
		//TODO take into account full sphere
		irr_glsl_BSDFSample s = irr_glsl_cos_weighted_cos_generate(currInteraction, rand);
		out_remainder = irr_glsl_cos_weighted_cos_remainder_and_pdf(pdf, s, currInteraction.isotropic);
		L = s.L;
	} else
#endif //OP_DIFFTRANS

#ifdef NDF_GGX
	if (ndf == NDF_GGX) {
		irr_glsl_BSDFSample s = irr_glsl_ggx_cos_generate(currInteraction, rand, ax, ay);
		out_remainder = irr_glsl_ggx_aniso_cos_remainder_and_pdf(pdf, s, currInteraction, ior, ax, ay);
		L = s.L;
	} else
#endif //NDF_GGX

#ifdef NDF_BECKMANN
	if (ndf == NDF_BECKMANN) {
		irr_glsl_BSDFSample s = irr_glsl_beckmann_smith_cos_generate(currInteraction, rand, ax, ay);
		out_remainder = irr_glsl_beckmann_aniso_cos_remainder_and_pdf(pdf, s, currInteraction, ior, ax, ax2, ay, ay2);
		L = s.L;
	} else
#endif //NDF_BECKMANN

#ifdef NDF_PHONG
	if (ndf == NDF_PHONG) {
		irr_glsl_BSDFSample s = irr_glsl_beckmann_smith_cos_generate(currInteraction, rand, ax, ay);
		out_remainder = irr_glsl_beckmann_aniso_cos_remainder_and_pdf(pdf, s, currInteraction, ior, ax, ax2, ay, ay2);
		L = s.L;
	} else
#endif //NDF_PHONG
	{}

	return L;
}

#endif