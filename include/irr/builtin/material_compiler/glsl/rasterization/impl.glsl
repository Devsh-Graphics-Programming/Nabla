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
	float a = getAlpha(params);
	float a2 = a*a;
	float ay = getAlphaV(params);
	float ay2 = ay*ay;

	if (op_hasSpecular(op))
	{
		if (ndf==NDF_GGX) {
			bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, a2);
		}
		else if (ndf==NDF_BECKMANN) {
			bxdf_eval_scalar_part = irr_glsl_beckmann_smith_height_correlated_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, a2);
		}
		else if (ndf==NDF_PHONG) {
			float n = irr_glsl_alpha2_to_phong_exp(a2);
			bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, n, a2);
		}
		else if (ndf==NDF_AS) {
			float nx = irr_glsl_alpha2_to_phong_exp(a2);
			float ny = irr_glsl_alpha2_to_phong_exp(ay2);
			bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(currBSDFParams, currInteraction, nx, ny, a2, ay2);
		}
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
#ifndef OP_DIFFTRANS
	if (op==OP_DIFFTRANS) {
		instr_execute_DIFFTRANS(instr, regs, params, bsdf_data);
	} else
#endif
#ifndef OP_DIELECTRIC
	if (op==OP_DIELECTRIC) {
		instr_execute_DIELECTRIC(instr, regs, params, bsdf_data);
	} else
#endif
#ifndef OP_BLEND
	if (op==OP_BLEND) {
		instr_execute_BLEND(instr, regs, params, bsdf_data);
	} else
#endif
#ifndef OP_BUMPMAP
	if (op==OP_BUMPMAP) {
		instr_execute_BUMPMAP(instr, L);
	} else
#endif
#ifndef OP_SET_GEOM_NORMAL
	if (op==OP_SET_GEOM_NORMAL) {
		instr_execute_SET_GEOM_NORMAL(L);
	} else
#endif
	{} //else "empty braces"
}

bxdf_eval_t runEvalStream(in instr_stream_t stream, in vec3 L)
{
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);
		instr_eval_execute(instr, L);
	}
	return readReg3(0u);//result is always in regs 0,1,2
}

#endif