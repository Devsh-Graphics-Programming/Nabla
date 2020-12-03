#ifndef _NBL_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_INCLUDED_
#define _NBL_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_INCLUDED_

#include <nbl/builtin/material_compiler/glsl/common_declarations.glsl>

#ifndef _NBL_USER_PROVIDED_MATERIAL_COMPILER_GLSL_BACKEND_FUNCTIONS_
	#error "You need to define 'vec3 nbl_glsl_MC_getNormalizedWorldSpaceV()', 'vec3 nbl_glsl_MC_getNormalizedWorldSpaceN()' , 'nbl_glsl_MC_getWorldSpacePosition()', 'instr_t nbl_glsl_MC_fetchInstr(in uint)', 'prefetch_instr_t nbl_glsl_MC_fetchPrefetchInstr(in uint)', 'bsdf_data_t nbl_glsl_MC_fetchBSDFData(in uint)' functions above"
#endif

#include <nbl/builtin/glsl/math/functions.glsl>
#include <nbl/builtin/glsl/format/decode.glsl>

MC_precomputed_t precomputeData(in bool frontface)
{
	MC_precomputed_t p;
	p.N = nbl_glsl_MC_getNormalizedWorldSpaceN();
	p.V = nbl_glsl_MC_getNormalizedWorldSpaceV();
	p.frontface = frontface;
	p.pos = nbl_glsl_MC_getWorldSpacePosition();

	return p;
}

float colorToScalar(in vec3 color)
{
	return dot(color, CIE_XYZ_Luma_Y_coeffs);
}

uint instr_getOffsetIntoRnPStream(in instr_t instr)
{
	return bitfieldExtract(instr.y, int(INSTR_OFFSET_INTO_REMANDPDF_STREAM_SHIFT-32u), int(INSTR_OFFSET_INTO_REMANDPDF_STREAM_WIDTH));
}
uint instr_getOpcode(in instr_t instr)
{
	return instr.x&INSTR_OPCODE_MASK;
}
uint instr_getBSDFbufOffset(in instr_t instr)
{
	return (instr.x>>INSTR_BSDF_BUF_OFFSET_SHIFT) & INSTR_BSDF_BUF_OFFSET_MASK;
}
uint instr_getNDF(in instr_t instr)
{
	return (instr.x>>INSTR_NDF_SHIFT) & INSTR_NDF_MASK;
}
uint instr_getRightJump(in instr_t instr)
{
	return bitfieldExtract(instr.y, int(INSTR_RIGHT_JUMP_SHIFT-32u), int(INSTR_RIGHT_JUMP_WIDTH));
}

bool instr_get1stParamTexPresence(in instr_t instr)
{
#if defined(PARAM1_NEVER_TEX)
	return false;
#elif defined(PARAM1_ALWAYS_TEX)
	return true;
#else
	return (instr.x&(1u<<INSTR_1ST_PARAM_TEX_SHIFT)) != 0u;
#endif
}
bool instr_get2ndParamTexPresence(in instr_t instr)
{
#if defined(PARAM2_NEVER_TEX)
	return false;
#elif defined(PARAM2_ALWAYS_TEX)
	return true;
#else
	return (instr.x&(1u<<INSTR_2ND_PARAM_TEX_SHIFT)) != 0u;
#endif
}

bool instr_params_getAlphaUTexPresence(in instr_t instr)
{
	return instr_get1stParamTexPresence(instr);
}
bool instr_params_getAlphaVTexPresence(in instr_t instr)
{
	return instr_get2ndParamTexPresence(instr);
}
bool instr_params_getReflectanceTexPresence(in instr_t instr)
{
	return instr_get2ndParamTexPresence(instr);
}
bool instr_getSigmaATexPresence(in instr_t instr)
{
	return instr_get1stParamTexPresence(instr);
}
bool instr_getTransmittanceTexPresence(in instr_t instr)
{
	return instr_get2ndParamTexPresence(instr);
}
bool instr_getWeightTexPresence(in instr_t instr)
{
	return instr_get1stParamTexPresence(instr);
}

//returns: x=dst, y=src1, z=src2
//works with tex prefetch instructions as well (x=reg0,y=reg1,z=reg2)
uvec3 instr_decodeRegisters(in instr_t instr)
{
	uvec3 regs = instr.yyy >> (uvec3(INSTR_REG_DST_SHIFT,INSTR_REG_SRC1_SHIFT,INSTR_REG_SRC2_SHIFT)-32u);
	return regs & uvec3(INSTR_REG_MASK);
}
#define REG_DST(r)	(r).x
#define REG_SRC1(r)	(r).y
#define REG_SRC2(r)	(r).z

bsdf_data_t fetchBSDFDataForInstr(in instr_t instr)
{
	uint ix = instr_getBSDFbufOffset(instr);
	return nbl_glsl_MC_fetchBSDFData(ix);
}

uint prefetch_instr_getRegCount(in prefetch_instr_t instr)
{
	uint dword4 = instr.w;
	return bitfieldExtract(dword4, PREFETCH_INSTR_REG_CNT_SHIFT, PREFETCH_INSTR_REG_CNT_WIDTH);
}

uint prefetch_instr_getDstReg(in prefetch_instr_t instr)
{
	uint dword4 = instr.w;
	return bitfieldExtract(dword4, PREFETCH_INSTR_DST_REG_SHIFT, PREFETCH_INSTR_DST_REG_WIDTH);
}

bool op_isBRDF(in uint op)
{
	return op<=OP_MAX_BRDF;
}
bool op_isBSDF(in uint op)
{
	return !op_isBRDF(op) && op<=OP_MAX_BSDF;
}
bool op_isBXDF(in uint op)
{
	return op<=OP_MAX_BSDF;
}
bool op_isBXDForCoatOrBlend(in uint op)
{
#ifdef OP_BLEND
	return op<=OP_BLEND;
#elif defined(OP_COATING)
	return op<=OP_COATING;
#else
	return op_isBXDF(op);
#endif
}
bool op_hasSpecular(in uint op)
{
	return
#if defined(OP_DIELECTRIC)
		op == OP_DIELECTRIC
#if defined(OP_CONDUCTOR)
		||
#endif
#endif
#if defined(OP_CONDUCTOR)
		op == OP_CONDUCTOR
#endif

#if !defined(OP_CONDUCTOR) && !defined(OP_DIELECTRIC)
		false
#endif
	;
}
bool op_isDiffuse(in uint op)
{
#if !defined(OP_DIFFUSE) && !defined(OP_DIFFTRANS)
	return false;
#else
	if (
#ifdef OP_DIFFUSE
		op == OP_DIFFUSE
#ifdef OP_DIFFTRANS
		||
#endif
#endif
#ifdef OP_DIFFTRANS
		op == OP_DIFFTRANS
#endif
		)
		return true;
	else
		return false;
#endif
}

#include <nbl/builtin/glsl/bxdf/common.glsl>
#include <nbl/builtin/glsl/bxdf/fresnel.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/diffuse/fresnel_correction.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl>
#include <nbl/builtin/glsl/bxdf/ndf/ggx.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/beckmann.glsl>
#include <nbl/builtin/glsl/bxdf/bsdf/specular/beckmann.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <nbl/builtin/glsl/bxdf/bsdf/specular/ggx.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl>
#include <nbl/builtin/glsl/bxdf/bsdf/diffuse/lambert.glsl>
#include <nbl/builtin/glsl/bxdf/bsdf/specular/dielectric.glsl>
#include <nbl/builtin/glsl/bump_mapping/utils.glsl>

//nbl_glsl_BSDFAnisotropicParams currBSDFParams;
MC_interaction_t currInteraction;
reg_t registers[REG_COUNT];

void updateLightSampleAfterNormalChange(inout nbl_glsl_LightSample out_s)
{
	out_s.TdotL = dot(currInteraction.inner.T, out_s.L);
	out_s.BdotL = dot(currInteraction.inner.B, out_s.L);
	out_s.NdotL = dot(currInteraction.inner.isotropic.N, out_s.L);
	out_s.NdotL2 = out_s.NdotL*out_s.NdotL;
}
void updateMicrofacetCacheAfterNormalChange(in nbl_glsl_LightSample s, inout MC_microfacet_t out_microfacet)
{
	const float NdotL = s.NdotL;
	const float NdotV = currInteraction.inner.isotropic.NdotV;

	const float LplusV_rcplen = inversesqrt(2.0 + 2.0 * s.VdotL);

	out_microfacet.inner.isotropic.NdotH = (NdotL + NdotV) * LplusV_rcplen;
	out_microfacet.inner.isotropic.NdotH2 = out_microfacet.inner.isotropic.NdotH * out_microfacet.inner.isotropic.NdotH;

	out_microfacet.inner.TdotH = (currInteraction.inner.TdotV + s.TdotL) * LplusV_rcplen;
	out_microfacet.inner.BdotH = (currInteraction.inner.BdotV + s.BdotL) * LplusV_rcplen;

	finalizeMicrofacet(out_microfacet);
}

vec3 textureOrRGBconst(in uvec2 data, in bool texPresenceFlag)
{
	return 
#ifdef TEX_PREFETCH_STREAM
	texPresenceFlag ? 
		uintBitsToFloat(uvec3(registers[data.x],registers[data.x+1u],registers[data.x+2u])) :
#endif
		nbl_glsl_decodeRGB19E7(data);
}

vec3 bsdf_data_getParam1(in bsdf_data_t data, in bool texPresence)
{
#ifdef PARAM1_ALWAYS_SAME_VALUE
	return PARAM1_VALUE;
#else
	return textureOrRGBconst(data.data[0].xy, texPresence);
#endif
}
vec3 bsdf_data_getParam2(in bsdf_data_t data, in bool texPresence)
{
#ifdef PARAM2_ALWAYS_SAME_VALUE
	return PARAM2_VALUE;
#else
	return textureOrRGBconst(data.data[0].zw, texPresence);
#endif
}

bvec2 instr_getTexPresence(in instr_t i)
{
	return bvec2(
		instr_get1stParamTexPresence(i),
		instr_get2ndParamTexPresence(i)
	);
}
params_t instr_getParameters(in instr_t i, in bsdf_data_t data)
{
	params_t p;
	bvec2 presence = instr_getTexPresence(i);
	//speculatively always read RGB
	p[0] = bsdf_data_getParam1(data, presence.x);
	p[1] = bsdf_data_getParam2(data, presence.y);

	return p;
}

//this should thought better
mat2x3 bsdf_data_decodeIoR(in bsdf_data_t data, in uint op)
{
	mat2x3 ior = mat2x3(0.0);
	ior[0] = nbl_glsl_decodeRGB19E7(data.data[1].xy);
#ifdef OP_CONDUCTOR
	ior[1] = (op == OP_CONDUCTOR) ? nbl_glsl_decodeRGB19E7(data.data[1].zw) : vec3(0.0);
#endif

	return ior;
}

void writeReg(in uint n, in float v)
{
	registers[n] = floatBitsToUint(v);
}
void writeReg(in uint n, in vec2 v)
{
	writeReg(n   ,v.x);
	writeReg(n+1u,v.y);	
}
void writeReg(in uint n, in vec3 v)
{
	writeReg(n   ,v.xy);
	writeReg(n+2u,v.z);
}
void writeReg(in uint n, in vec4 v)
{
	writeReg(n   ,v.xyz);
	writeReg(n+3u,v.w);
}
float readReg1(in uint n)
{
	return uintBitsToFloat( registers[n] );
}
vec2 readReg2(in uint n)
{
	return vec2(
		readReg1(n), readReg1(n+1u)
	);
}
vec3 readReg3(in uint n)
{
	return vec3(
		readReg2(n), readReg1(n+2u)
	);
}
vec4 readReg4(in uint n)
{
	return vec4(
		readReg3(n), readReg1(n+3u)
	);
}

mat2x4 instr_fetchSrcRegs(in instr_t i, in uvec3 regs)
{
	uvec3 r = regs;
	mat2x4 srcs;
	srcs[0] = readReg4(REG_SRC1(r));
	srcs[1] = readReg4(REG_SRC2(r));
	return srcs;
}
mat2x4 instr_fetchSrcRegs(in instr_t i)
{
	uvec3 r = instr_decodeRegisters(i);
	return instr_fetchSrcRegs(i, r);
}

void setCurrInteraction(in vec3 N, in vec3 V, in vec3 pos)
{
	nbl_glsl_IsotropicViewSurfaceInteraction interaction = nbl_glsl_calcFragmentShaderSurfaceInteractionFromViewVector(V, pos, N);
	currInteraction.inner = nbl_glsl_calcAnisotropicInteraction(interaction);
	finalizeInteraction(currInteraction);
}
void setCurrInteraction(in MC_precomputed_t precomp)
{
	setCurrInteraction(precomp.frontface ? precomp.N : -precomp.N, precomp.V, precomp.pos);
}
void updateCurrInteraction(in MC_precomputed_t precomp, in vec3 N)
{
	// precomputed normals already have correct orientation
	setCurrInteraction(N, precomp.V, precomp.pos);
}

//clamping alpha to min MIN_ALPHA because we're using microfacet BxDFs for deltas as well (and NDFs often end up NaN when given alpha=0) because of less deivergence
//probably temporary solution
#define MIN_ALPHA 0.0001
float params_getAlpha(in params_t p)
{
	return max(p[PARAMS_ALPHA_U_IX].x,MIN_ALPHA);
}
vec3 params_getReflectance(in params_t p)
{
	return p[PARAMS_REFLECTANCE_IX];
}
float params_getAlphaV(in params_t p)
{
	return max(p[PARAMS_ALPHA_V_IX].x,MIN_ALPHA);
}
vec3 params_getSigmaA(in params_t p)
{
	return p[PARAMS_SIGMA_A_IX];
}
vec3 params_getBlendWeight(in params_t p)
{
	return p[PARAMS_WEIGHT_IX];
}
vec3 params_getTransmittance(in params_t p)
{
	return p[PARAMS_TRANSMITTANCE_IX];
}

bxdf_eval_t instr_execute_cos_eval_COATING(in instr_t instr, in mat2x4 srcs, in params_t params, in vec3 eta, in vec3 eta2, in nbl_glsl_LightSample s, in bsdf_data_t data, out float out_weight)
{
	//vec3 thickness_sigma = params_getSigmaA(params);

	// TODO include thickness_sigma in diffuse weight computation: exp(sigma_thickness * freePath)
	// freePath = ( sqrt(refract_compute_NdotT2(NdotL2, rcpOrientedEta2)) + sqrt(refract_compute_NdotT2(NdotV2, rcpOrientedEta2)) )
	vec3 fresnelNdotV = nbl_glsl_fresnel_dielectric_frontface_only(eta, max(currInteraction.inner.isotropic.NdotV, 0.0));
	vec3 wd = nbl_glsl_diffuseFresnelCorrectionFactor(eta, eta2) * (vec3(1.0) - fresnelNdotV) * (vec3(1.0) - nbl_glsl_fresnel_dielectric_frontface_only(eta, s.NdotL));

	bxdf_eval_t coat = srcs[0].xyz;
	bxdf_eval_t coated = srcs[1].xyz;

	out_weight = dot(fresnelNdotV, CIE_XYZ_Luma_Y_coeffs);

	return coat + coated*wd;
}

eval_and_pdf_t instr_execute_cos_eval_pdf_COATING(in instr_t instr, in mat2x4 srcs, in params_t params, in vec3 eta, in vec3 eta2, in nbl_glsl_LightSample s, in bsdf_data_t data)
{
	//float thickness = uintBitsToFloat(data.data[2].z);

	float weight;
	bxdf_eval_t bxdf = instr_execute_cos_eval_COATING(instr, srcs, params, eta, eta2, s, data, weight);
	float coat_pdf = srcs[0].w;
	float coated_pdf = srcs[1].w;

	float pdf = mix(coated_pdf, coat_pdf, weight);

	return eval_and_pdf_t(bxdf, pdf);
}

void instr_execute_BUMPMAP(in instr_t instr, in mat2x4 srcs, in MC_precomputed_t precomp)
{
	vec3 N = srcs[0].xyz;
	updateCurrInteraction(precomp, N);
}

void instr_execute_SET_GEOM_NORMAL(in instr_t instr, in MC_precomputed_t precomp)
{
	setCurrInteraction(precomp);
}

bxdf_eval_t instr_execute_cos_eval_BLEND(in instr_t instr, in mat2x4 srcs, in params_t params, in bsdf_data_t data)
{
	vec3 w = params_getBlendWeight(params);
	bxdf_eval_t bxdf1 = srcs[0].xyz;
	bxdf_eval_t bxdf2 = srcs[1].xyz;

	bxdf_eval_t blend = mix(bxdf1, bxdf2, w);
	return blend;
}
eval_and_pdf_t instr_execute_cos_eval_pdf_BLEND(in instr_t instr, in mat2x4 srcs, in params_t params, in bsdf_data_t data)
{
	vec3 w = params_getBlendWeight(params);
	bxdf_eval_t bxdf1 = srcs[0].xyz;
	bxdf_eval_t bxdf2 = srcs[1].xyz;
	float w_pdf = colorToScalar(w);
	float pdf1 = srcs[0].w;
	float pdf2 = srcs[1].w;

	//instead of doing this here, remainder and pdf returned from generator stream is weighted
	//so it correctly adds up at the end
	/*
	//generator is denoted as the one with negative probability
	if (pdfa<0.0) {//`a` is remainder
		pdfa = abs(pdfa);
		vec3 rem = (a*wa + b/pdfa*wb) / (wa + pdfb/pdfa*wb);
		blend = rem_and_pdf_t(rem, pdf);
	}
	else
	if (pdfb<0.0) {//`b` is remainder
		pdfb = abs(pdfb);
		vec3 rem = (b*wb + a/pdfb*wa) / (wb + pdfa/pdfb*wa);
		blend = rem_and_pdf_t(rem, pdf);
	}
	else {
		vec3 rem = mix(a,b,wa);
		blend = rem_and_pdf_t( rem,pdf );
	}
	*/
	
	bxdf_eval_t eval = mix(bxdf1, bxdf2, w);
	float pdf = mix(pdf1, pdf2, w_pdf);

	return eval_and_pdf_t(eval, pdf);
}

vec3 fetchTex(in uvec3 texid, in vec2 uv, in mat2 dUV)
{
	float scale = uintBitsToFloat(texid.z);

#if _NBL_VT_FLOAT_VIEWS_COUNT
	return nbl_glsl_vTextureGrad(texid.xy, uv, dUV).rgb*scale;
#else
	return vec3(0.0);
#endif
}

void runTexPrefetchStream(in instr_stream_t stream, in vec2 uv, in mat2 dUV)
{
	for (uint i = 0u; i < stream.count; ++i)
	{
		prefetch_instr_t instr = nbl_glsl_MC_fetchPrefetchInstr(stream.offset+i);

		uint regcnt = prefetch_instr_getRegCount(instr);
		uint reg = prefetch_instr_getDstReg(instr);

		vec3 val = fetchTex(instr.xyz, uv, dUV);

		writeReg(reg, val.x);
#if defined(PREFETCH_REG_COUNT_2) || defined(PREFETCH_REG_COUNT_3)
		if (regcnt>=2u)
			writeReg(reg+1u, val.y);
#endif
#if defined(PREFETCH_REG_COUNT_3)
		if (regcnt==3u)
			writeReg(reg+2u, val.z);
#endif
	}
}

void runNormalPrecompStream(in instr_stream_t stream, in mat2 dUV, in MC_precomputed_t precomp)
{
	setCurrInteraction(precomp);
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = nbl_glsl_MC_fetchInstr(stream.offset+i);

		bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);

		uint srcreg = bsdf_data.data[0].x;
		uint dstreg = REG_DST(instr_decodeRegisters(instr));

		vec2 dh = readReg2(srcreg);
		
		writeReg(dstreg,
			nbl_glsl_perturbNormal_derivativeMap(currInteraction.inner.isotropic.N, dh, currInteraction.inner.isotropic.V.dPosdScreen, dUV)
		);
	}
}

#ifdef GEN_CHOICE_STREAM
void instr_eval_and_pdf_execute(in instr_t instr, in MC_precomputed_t precomp, in nbl_glsl_LightSample s, in MC_microfacet_t _microfacet, in bool skip)
{
	const uint op = instr_getOpcode(instr);
	const bool is_bxdf = op_isBXDF(op);
	const bool is_bsdf = !op_isBRDF(op); //note it actually tells if op is BSDF or BUMPMAP or SET_GEOM_NORMAL (divergence reasons) [(is_bxdf && is_bsdf) actually tells that op is bsdf]
	// (is_bxdf && is_bsdf) -> BSDF
	// (is_bxdf && !is_bsdf) -> BRDF
	const bool is_bxdf_or_combiner = op_isBXDForCoatOrBlend(op);

	uvec3 regs = instr_decodeRegisters(instr);
	params_t params;
	bsdf_data_t bsdf_data;
	mat2x3 ior;
	mat2x3 ior2;
	MC_microfacet_t microfacet;

	const bool run = !skip;

	if (is_bxdf_or_combiner && run)
	{
		bsdf_data = fetchBSDFDataForInstr(instr);
		ior = bsdf_data_decodeIoR(bsdf_data, op);
		ior2 = matrixCompMult(ior, ior);
		params = instr_getParameters(instr, bsdf_data);
	}

	const float NdotV = nbl_glsl_conditionalAbsOrMax(is_bsdf, currInteraction.inner.isotropic.NdotV, 0.0);

	vec3 eval = vec3(0.0);
	float pdf = 0.0;

	if (is_bxdf && run && (NdotV > FLT_MIN))
	{
		//speculative execution
		uint ndf = instr_getNDF(instr);
		float a = params_getAlpha(params);
		float a2 = a*a;
#ifdef ALL_ISOTROPIC_BXDFS
		float one_minus_a2 = 1.0 - a2;
#else
		float ay = params_getAlphaV(params);
		float ay2 = ay*ay;
#endif

		const vec3 albedo = params_getReflectance(params);

		const float NdotL = nbl_glsl_conditionalAbsOrMax(is_bsdf, s.NdotL, 0.0);

#if defined(OP_DIFFUSE) || defined(OP_DIFFTRANS)
		if (op_isDiffuse(op))
		{
			if (NdotL > FLT_MIN)
			{
				eval = albedo * nbl_glsl_oren_nayar_cos_remainder_and_pdf_wo_clamps(pdf, a2, s.VdotL, NdotL, NdotV);
				pdf *= is_bsdf ? 0.5 : 1.0;
				eval *= pdf;
			}
		}
		else
#endif
#if defined(OP_CONDUCTOR) || defined(OP_DIELECTRIC)
		{
			const float eta = colorToScalar(ior[0]);
			const float rcp_eta = 1.0 / eta;

			bool is_valid = true;
			bool refraction = false;
#ifdef OP_DIELECTRIC
			if (op == OP_DIELECTRIC && nbl_glsl_isTransmissionPath(currInteraction.inner.isotropic.NdotV, s.NdotL))
			{
				nbl_glsl_calcAnisotropicMicrofacetCache(microfacet.inner, true, currInteraction.inner.isotropic.V.dir, s.L, currInteraction.inner.T, currInteraction.inner.B, currInteraction.inner.isotropic.N, s.NdotL, s.VdotL, eta, rcp_eta);
				finalizeMicrofacet(microfacet);
				refraction = true;
			}
			else
#endif
				microfacet = _microfacet;

#if defined(OP_DIELECTRIC) || defined(OP_CONDUCTOR)
			is_valid = nbl_glsl_isValidVNDFMicrofacet(microfacet.inner.isotropic, is_bsdf, refraction, s.VdotL, eta, rcp_eta);
#endif

			if (is_valid && a > ALPHA_EPSILON)
			{
				const float TdotV2 = currInteraction.TdotV2;
				const float BdotV2 = currInteraction.BdotV2;
				const float NdotV2 = currInteraction.inner.isotropic.NdotV_squared;
				const float NdotL2 = s.NdotL2;

				const float TdotH2 = microfacet.TdotH2;
				const float BdotH2 = microfacet.BdotH2;
				const float NdotH2 = microfacet.inner.isotropic.NdotH2;

				float G1_over_2NdotV = 0.0;
				float G2_over_G1 = 0.0;
				float ndf_val = 0.0;

				BEGIN_CASES(ndf)
#ifdef NDF_GGX
				CASE_BEGIN(ndf, NDF_GGX) {
#ifdef ALL_ISOTROPIC_BXDFS
					G1_over_2NdotV = nbl_glsl_GGXSmith_G1_wo_numerator(NdotV, NdotV2, a2, one_minus_a2);
					G2_over_G1 = nbl_glsl_ggx_smith_G2_over_G1(NdotL, NdotL2, NdotV, NdotV2, a2, one_minus_a2);
					ndf_val = nbl_glsl_ggx_trowbridge_reitz(a2, NdotH2);
#else
					G1_over_2NdotV = nbl_glsl_GGXSmith_G1_wo_numerator(NdotV, TdotV2, BdotV2, NdotV2, a2, ay2);
					G2_over_G1 = nbl_glsl_ggx_smith_G2_over_G1(NdotL, s.TdotL*s.TdotL, s.BdotL*s.BdotL, NdotL2, NdotV, TdotV2, BdotV2, NdotV2, a2, ay2);
					ndf_val = nbl_glsl_ggx_aniso(TdotH2, BdotH2, NdotH2, a, ay, a2, ay2);
#endif
				} CASE_END
#endif

#ifdef NDF_BECKMANN
				CASE_BEGIN(ndf, NDF_BECKMANN) {
#ifdef ALL_ISOTROPIC_BXDFS
					float lambdaV = nbl_glsl_smith_beckmann_Lambda(NdotV2, a2);
					G1_over_2NdotV = nbl_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
					G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, NdotL2, a2);
					ndf_val = nbl_glsl_beckmann(a2, NdotH2);
#else
					float lambdaV = nbl_glsl_smith_beckmann_Lambda(TdotV2, BdotV2, NdotV2, a2, ay2);
					G1_over_2NdotV = nbl_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
					G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, s.TdotL*s.TdotL, s.BdotL*s.BdotL, NdotL2, a2, ay2);
					ndf_val = nbl_glsl_beckmann(a, ay, a2, ay2, TdotH2, BdotH2, NdotH2);
#endif
				} CASE_END
#endif

#ifdef NDF_PHONG
				CASE_BEGIN(ndf, NDF_PHONG) {
#ifdef ALL_ISOTROPIC_BXDFS
					float lambdaV = nbl_glsl_smith_beckmann_Lambda(NdotV2, a2);
					G1_over_2NdotV = nbl_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
					G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, NdotL2, a2);
					ndf_val = nbl_glsl_beckmann(a2, NdotH2);
#else
					float lambdaV = nbl_glsl_smith_beckmann_Lambda(TdotV2, BdotV2, NdotV2, a2, ay2);
					G1_over_2NdotV = nbl_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
					G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, s.TdotL*s.TdotL, s.BdotL*s.BdotL, NdotL2, a2, ay2);
					ndf_val = nbl_glsl_beckmann(a, ay, a2, ay2, TdotH2, BdotH2, NdotH2);
#endif
				} CASE_END
#endif
				CASE_OTHERWISE
				{} //else "empty braces"
				END_CASES

				pdf = nbl_glsl_smith_VNDF_pdf_wo_clamps(ndf_val, G1_over_2NdotV);
				float remainder_scalar_part = G2_over_G1;

				const float VdotH = abs(microfacet.inner.isotropic.VdotH);
				vec3 fr;
#ifdef OP_CONDUCTOR
				if (op == OP_CONDUCTOR)
					fr = nbl_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
				else
#endif
					fr = vec3(nbl_glsl_fresnel_dielectric_common(eta*eta, VdotH));

				float eval_scalar_part = remainder_scalar_part * pdf;

#ifndef NO_BSDF
				if (is_bsdf)
				{
					float LdotH = microfacet.inner.isotropic.LdotH;
					float VdotHLdotH = microfacet.inner.isotropic.VdotH * LdotH;
					LdotH = abs(LdotH);
#ifdef NDF_GGX
					if (ndf == NDF_GGX)
						eval_scalar_part = nbl_glsl_ggx_microfacet_to_light_measure_transform(eval_scalar_part, NdotL, refraction, VdotH, LdotH, VdotHLdotH, eta);
					else
#endif
						eval_scalar_part = nbl_glsl_microfacet_to_light_measure_transform(eval_scalar_part, NdotV, refraction, VdotH, LdotH, VdotHLdotH, eta);

					float reflectance = colorToScalar(fr);
					reflectance = refraction ? (1.0 - reflectance) : reflectance;
					pdf *= reflectance;
				}
#endif 
				eval = fr * eval_scalar_part;
			} 
		}
#endif
		{} // empty else for when there are diffuse ops but arent any specular ones
	}

	eval_and_pdf_t result = eval_and_pdf_t(eval, pdf);
	if (!is_bxdf)
	{
		mat2x4 srcs = instr_fetchSrcRegs(instr, regs);
		BEGIN_CASES(op)
#ifdef OP_COATING
		CASE_BEGIN(op, OP_COATING) {
			result = instr_execute_cos_eval_pdf_COATING(instr, srcs, params, ior[0], ior2[0], s, bsdf_data);
		} CASE_END
#endif
#ifdef OP_BLEND
		CASE_BEGIN(op, OP_BLEND) {
			result = instr_execute_cos_eval_pdf_BLEND(instr, srcs, params, bsdf_data);
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

	if (is_bxdf_or_combiner)
		writeReg(REG_DST(regs), result);
}

eval_and_pdf_t nbl_bsdf_eval_and_pdf(in MC_precomputed_t precomp, in instr_stream_t stream, in uint generator_offset, inout nbl_glsl_LightSample s, inout MC_microfacet_t microfacet)
{
	setCurrInteraction(precomp);
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = nbl_glsl_MC_fetchInstr(stream.offset+i);
		uint op = instr_getOpcode(instr);

		bool skip = (i == generator_offset);
		// skip deltas
#ifdef OP_THINDIELECTRIC
		skip = skip || (op == OP_THINDIELECTRIC);
#endif
#ifdef OP_DELTATRANS
		skip = skip || (op == OP_DELTATRANS);
#endif

		instr_eval_and_pdf_execute(instr, precomp, s, microfacet, skip);

#if defined(OP_SET_GEOM_NORMAL)||defined(OP_BUMPMAP)
		if (
#ifdef OP_SET_GEOM_NORMAL
			op == OP_SET_GEOM_NORMAL
#ifdef OP_BUMPMAP
			||
#endif
#endif
#ifdef OP_BUMPMAP
			op == OP_BUMPMAP
#endif
			) {
			updateLightSampleAfterNormalChange(s);
			updateMicrofacetCacheAfterNormalChange(s, microfacet);
		}
#endif
	}

	eval_and_pdf_t eval_and_pdf = readReg4(0u);
	return eval_and_pdf;
}

nbl_glsl_LightSample nbl_bsdf_cos_generate(in MC_precomputed_t precomp, in instr_stream_t stream, in vec3 rand, out vec3 out_remainder, out float out_pdf, out MC_microfacet_t out_microfacet, out uint out_gen_rnpOffset)
{
	uint ix = 0u;
	instr_t instr = nbl_glsl_MC_fetchInstr(stream.offset);
	uint op = instr_getOpcode(instr);
	vec3 u = rand;

	out_pdf = 1.0;

	setCurrInteraction(precomp);

	bool is_coat = false;
	while (!op_isBXDF(op))
	{
#if defined(OP_COATING) || defined(OP_BLEND)
		if (op_isBXDForCoatOrBlend(op)) {
			bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
			vec3 w_;
#if defined(OP_BLEND)
			if (op == OP_BLEND)
			{
				params_t params = instr_getParameters(instr, bsdf_data);
				w_ = params_getBlendWeight(params);
			}
			else
#endif
			{
				vec3 eta = bsdf_data_decodeIoR(bsdf_data, OP_COATING)[0];
				w_ = nbl_glsl_fresnel_dielectric_frontface_only(eta, max(currInteraction.inner.isotropic.NdotV, 0.0));
			}

			float w = colorToScalar(w_);
			float rcpChoiceProb;
			// right is **coated** material in case of coating
			bool choseRight = nbl_glsl_partitionRandVariable(w, u.z, rcpChoiceProb);

			uint right_ix = instr_getRightJump(instr);
			ix = choseRight ? right_ix : (ix + 1u);
			out_pdf /= rcpChoiceProb;

			is_coat = (op==OP_COATING && !choseRight);
		} else
#endif //OP_COATING or OP_BLEND
		{
#ifdef OP_SET_GEOM_NORMAL
			if (op==OP_SET_GEOM_NORMAL)
			{
				instr_execute_SET_GEOM_NORMAL(instr, precomp);
			} else 
#endif //OP_SET_GEOM_NORMAL
#ifdef OP_BUMPMAP
			if (op==OP_BUMPMAP)
			{
				mat2x4 srcs = instr_fetchSrcRegs(instr);
				instr_execute_BUMPMAP(instr, srcs, precomp);
			} else
#endif //OP_BUMPMAP
			{}
			ix = ix + 1u;
		}

		instr = nbl_glsl_MC_fetchInstr(stream.offset+ix);
		op = instr_getOpcode(instr);
	}

	out_gen_rnpOffset = instr_getOffsetIntoRnPStream(instr);

	bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
	params_t params = instr_getParameters(instr, bsdf_data);
	//speculatively
	const float ax = params_getAlpha(params);
	const float ax2 = ax*ax;
	const float ay = params_getAlphaV(params);
	const float ay2 = ay*ay;
	const mat2x3 ior = bsdf_data_decodeIoR(bsdf_data,op);
	const mat2x3 ior2 = matrixCompMult(ior,ior);
	const bool is_bsdf = !op_isBRDF(op) && !is_coat;
	const vec3 albedo = params_getReflectance(params);

	const float NdotV = nbl_glsl_conditionalAbsOrMax(is_bsdf, currInteraction.inner.isotropic.NdotV, 0.0);
	const bool positiveNdotV = (NdotV > FLT_MIN);

	float localPdf = 0.0;
	vec3 rem = vec3(0.0);
	uint ndf = instr_getNDF(instr);
	nbl_glsl_LightSample s;

	const vec3 localV = nbl_glsl_getTangentSpaceV(currInteraction.inner);
	const mat3 tangentFrame = nbl_glsl_getTangentFrame(currInteraction.inner);


#ifdef OP_DELTATRANS
	if (op == OP_DELTATRANS)
	{
		s = nbl_glsl_createLightSample(-precomp.V, -1.0, currInteraction.inner.T, currInteraction.inner.B, currInteraction.inner.isotropic.N);
		// not computing microfacet cache since it's always transmission and it will be recomputed anyway
		rem = vec3(1.0);
		localPdf = nbl_glsl_FLT_INF;
	} else
#endif
#ifdef OP_THINDIELECTRIC
	if (op == OP_THINDIELECTRIC)
	{
		const vec3 luminosityContributionHint = CIE_XYZ_Luma_Y_coeffs;
		vec3 remMetadata;
		s = nbl_glsl_thin_smooth_dielectric_cos_generate_wo_clamps(
			currInteraction.inner.isotropic.V.dir, currInteraction.inner.T, currInteraction.inner.B, currInteraction.inner.isotropic.N, 
			currInteraction.inner.isotropic.NdotV, NdotV, u, ior2[0], luminosityContributionHint, remMetadata
		);
		out_microfacet.inner = nbl_glsl_calcAnisotropicMicrofacetCache(currInteraction.inner, s);
		finalizeMicrofacet(out_microfacet);
		rem = nbl_glsl_thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(localPdf, remMetadata);
	} else
#endif
	if (positiveNdotV)
	{
#if defined(OP_DIFFUSE) || defined(OP_DIFFTRANS)
		if (op_isDiffuse(op))
		{
			vec3 localL = nbl_glsl_projected_hemisphere_generate(u.xy);
#ifndef NO_BSDF
			if (is_bsdf)
			{
				float dummy;
				bool flip = nbl_glsl_partitionRandVariable(0.5, u.z, dummy);
				localL = flip ? -localL : localL;
			}
#endif
			s = nbl_glsl_createLightSampleTangentSpace(localV, localL, tangentFrame);
			out_microfacet.inner = nbl_glsl_calcAnisotropicMicrofacetCache(currInteraction.inner, s);
			finalizeMicrofacet(out_microfacet);

			const float NdotL = nbl_glsl_conditionalAbsOrMax(is_bsdf, s.NdotL, 0.0);
			rem = albedo*nbl_glsl_oren_nayar_cos_remainder_and_pdf_wo_clamps(localPdf, ax2, dot(currInteraction.inner.isotropic.V.dir, s.L), NdotL, NdotV);
			localPdf *= is_bsdf ? 0.5 : 1.0;
		} else
#endif
#if defined(OP_CONDUCTOR) || defined(OP_DIELECTRIC)
		if (op_hasSpecular(op))
		{
			localPdf = 1.0;

			const float TdotV2 = currInteraction.TdotV2;
			const float BdotV2 = currInteraction.BdotV2;
			const float NdotV2 = currInteraction.inner.isotropic.NdotV_squared;

			const vec3 upperHemisphereLocalV = currInteraction.inner.isotropic.NdotV < 0.0 ? -localV : localV;

			float G2_over_G1 = 0.0;
			float G1_over_2NdotV = 0.0;
			float ndf_val = 0.0;
			vec3 localH = vec3(0.0);

			BEGIN_CASES(ndf)
#ifdef NDF_GGX
			CASE_BEGIN(ndf, NDF_GGX) 
			{
				// why is it called without "wo_clamps" and beckmann sampling is?
				localH = nbl_glsl_ggx_cos_generate(upperHemisphereLocalV, u.xy, ax, ay);
			} CASE_END
#endif //NDF_GGX

#ifdef NDF_BECKMANN
			CASE_BEGIN(ndf, NDF_BECKMANN) 
			{
				localH = nbl_glsl_beckmann_cos_generate_wo_clamps(upperHemisphereLocalV, u.xy, ax, ay);
			} CASE_END
#endif //NDF_BECKMANN

#ifdef NDF_PHONG
			CASE_BEGIN(ndf, NDF_PHONG) 
			{
				localH = nbl_glsl_beckmann_cos_generate_wo_clamps(upperHemisphereLocalV, u.xy, ax, ay);
			} CASE_END
#endif //NDF_PHONG
			CASE_OTHERWISE
			{}
			END_CASES

			vec3 localL;
			const float VdotH = dot(localH, localV);
			const float VdotH_clamp = is_bsdf ? VdotH : max(VdotH, 0.0);
			vec3 fr;
			bool refraction = false;
			float eta = colorToScalar(ior[0]);
			float rcpEta = 1.0 / eta;
#ifdef OP_CONDUCTOR
			if (op == OP_CONDUCTOR)
			{
				fr = nbl_glsl_fresnel_conductor(ior[0], ior[1], VdotH_clamp);
				rem = fr;
			}
			else
#endif
			{
				fr = vec3(nbl_glsl_fresnel_dielectric_common(eta*eta, VdotH_clamp));

				const float refractionProb = colorToScalar(fr);
				float rcpChoiceProb;
				refraction = nbl_glsl_partitionRandVariable(refractionProb, u.z, rcpChoiceProb);
				localPdf /= rcpChoiceProb;
				rem = vec3(1.0);
			}

			out_microfacet.inner = nbl_glsl_calcAnisotropicMicrofacetCache(refraction, localV, localH, localL, rcpEta, rcpEta*rcpEta);
			s = nbl_glsl_createLightSampleTangentSpace(localV, localL, tangentFrame);

			finalizeMicrofacet(out_microfacet);
			const float TdotH2 = out_microfacet.TdotH2;
			const float BdotH2 = out_microfacet.BdotH2;
			const float NdotH2 = out_microfacet.inner.isotropic.NdotH2;
			const float NdotL = nbl_glsl_conditionalAbsOrMax(is_bsdf, s.NdotL, 0.0);

			BEGIN_CASES(ndf)
#ifdef NDF_GGX
			CASE_BEGIN(ndf, NDF_GGX)
			{
				G2_over_G1 = nbl_glsl_ggx_smith_G2_over_G1(NdotL, s.TdotL*s.TdotL, s.BdotL*s.BdotL, s.NdotL*s.NdotL, NdotV, TdotV2, BdotV2, NdotV2, ax2, ay2);
				G1_over_2NdotV = nbl_glsl_GGXSmith_G1_wo_numerator(NdotV, TdotV2, BdotV2, NdotV2, ax2, ay2);
				ndf_val = nbl_glsl_ggx_aniso(TdotH2, BdotH2, NdotH2, ax, ay, ax2, ay2);
			} CASE_END
#endif //NDF_GGX

#ifdef NDF_BECKMANN
			CASE_BEGIN(ndf, NDF_BECKMANN)
			{
				float lambdaV = nbl_glsl_smith_beckmann_Lambda(TdotV2, BdotV2, NdotV2, ax2, ay2);
				G1_over_2NdotV = nbl_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
				G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, s.TdotL*s.TdotL, s.BdotL*s.BdotL, s.NdotL*s.NdotL, ax2, ay2);
				ndf_val = nbl_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);
			} CASE_END
#endif //NDF_BECKMANN

#ifdef NDF_PHONG
			CASE_BEGIN(ndf, NDF_PHONG)
			{
				float lambdaV = nbl_glsl_smith_beckmann_Lambda(TdotV2, BdotV2, NdotV2, ax2, ay2);
				G1_over_2NdotV = nbl_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
				G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, s.TdotL * s.TdotL, s.BdotL * s.BdotL, s.NdotL * s.NdotL, ax2, ay2);
				ndf_val = nbl_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);
			} CASE_END
#endif //NDF_PHONG
			CASE_OTHERWISE
			{}
			END_CASES

			const float LdotH = out_microfacet.inner.isotropic.LdotH;
			const float VdotHLdotH = VdotH * LdotH;
			rem *= G2_over_G1;
			// note: at this point localPdf is already multiplied by transmission/reflection choice probability
			localPdf *= nbl_glsl_smith_FVNDF_pdf_wo_clamps(ndf_val, G1_over_2NdotV, NdotV, refraction, VdotH, LdotH, VdotHLdotH, eta);
		} else
#endif
		{} //empty braces for `else`
	}

	out_remainder = rem;
	out_pdf *= localPdf; 

	return s;
}

vec3 runGenerateAndRemainderStream(in MC_precomputed_t precomp, in instr_stream_t gcs, in instr_stream_t rnps, in vec3 rand, out float out_pdf, out nbl_glsl_LightSample out_smpl)
{
	vec3 generator_rem;
	float generator_pdf;
	MC_microfacet_t microfacet;
	uint generator_rnpOffset;
	nbl_glsl_LightSample s = nbl_bsdf_cos_generate(precomp, gcs, rand, generator_rem, generator_pdf, microfacet, generator_rnpOffset);
	eval_and_pdf_t eval_pdf = nbl_bsdf_eval_and_pdf(precomp, rnps, generator_rnpOffset, s, microfacet);
	bxdf_eval_t acc = eval_pdf.rgb;
	float restPdf = eval_pdf.a;
	float pdf = generator_pdf + restPdf;

	out_smpl = s;
	out_pdf = pdf;

	float rcp_generator_pdf = 1.0 / generator_pdf;
	vec3 rem = (generator_rem + acc*rcp_generator_pdf) / (1.0 + restPdf*rcp_generator_pdf);

	return rem;
}

#endif //GEN_CHOICE_STREAM

#endif