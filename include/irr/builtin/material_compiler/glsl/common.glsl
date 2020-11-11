#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_INCLUDED_

#include <irr/builtin/material_compiler/glsl/common_declarations.glsl>

#ifndef _IRR_USER_PROVIDED_MATERIAL_COMPILER_GLSL_BACKEND_FUNCTIONS_
	#error "You need to define 'vec3 irr_glsl_MC_getNormalizedWorldSpaceV()', 'vec3 irr_glsl_MC_getNormalizedWorldSpaceN()' , 'irr_glsl_MC_getWorldSpacePosition()', 'instr_t irr_glsl_MC_fetchInstr(in uint)', 'prefetch_instr_t irr_glsl_MC_fetchPrefetchInstr(in uint)', 'bsdf_data_t irr_glsl_MC_fetchBSDFData(in uint)' functions above"
#endif

#include <irr/builtin/glsl/math/functions.glsl>
#include <irr/builtin/glsl/format/decode.glsl>

MC_precomputed_t precomputeData()
{
	MC_precomputed_t p;
	p.N = irr_glsl_MC_getNormalizedWorldSpaceN();
	p.V = irr_glsl_MC_getNormalizedWorldSpaceV();
	p.NdotV = dot(p.N, p.V);
	p.pos = irr_glsl_MC_getWorldSpacePosition();

	return p;
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
bool instr_get3rdParamTexPresence(in instr_t instr)
{
#if defined(PARAM3_NEVER_TEX)
	return false;
#elif defined(PARAM3_ALWAYS_TEX)
	return true;
#else
	return (instr.x&(1u<<INSTR_3RD_PARAM_TEX_SHIFT)) != 0u;
#endif
}
bool instr_get4thParamTexPresence(in instr_t instr)
{
#if defined(PARAM4_NEVER_TEX)
	return false;
#elif defined(PARAM4_ALWAYS_TEX)
	return true;
#else
	return (instr.x&(1u<<INSTR_4TH_PARAM_TEX_SHIFT)) != 0u;
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
	return instr_get4thParamTexPresence(instr);
}
bool instr_getSigmaATexPresence(in instr_t instr)
{
	return instr_get4thParamTexPresence(instr);
}
bool instr_getTransmittanceTexPresence(in instr_t instr)
{
	return instr_get1stParamTexPresence(instr);
}
bool instr_getWeightTexPresence(in instr_t instr)
{
	return instr_get1stParamTexPresence(instr);
}
bool instr_getOpacityTexPresence(in instr_t instr)
{
	return instr_get3rdParamTexPresence(instr);
}

bool instr_getTwosided(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_TWOSIDED_SHIFT)) != 0u;
}
bool instr_getMaskFlag(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_MASKFLAG_SHIFT)) != 0u;
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
	return irr_glsl_MC_fetchBSDFData(ix);
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
		return false;
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

#include <irr/builtin/glsl/bxdf/common.glsl>
#include <irr/builtin/glsl/bxdf/fresnel.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/fresnel_correction.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl>
#include <irr/builtin/glsl/bxdf/ndf/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/beckmann.glsl>
#include <irr/builtin/glsl/bxdf/bsdf/specular/beckmann.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <irr/builtin/glsl/bxdf/bsdf/specular/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl>
#include <irr/builtin/glsl/bxdf/bsdf/diffuse/lambert.glsl>
#include <irr/builtin/glsl/bxdf/bsdf/specular/dielectric.glsl>
#include <irr/builtin/glsl/bump_mapping/utils.glsl>

//irr_glsl_BSDFAnisotropicParams currBSDFParams;
irr_glsl_AnisotropicViewSurfaceInteraction currInteraction;
reg_t registers[REG_COUNT];

vec3 textureOrRGBconst(in uvec2 data, in bool texPresenceFlag)
{
	return 
#ifdef TEX_PREFETCH_STREAM
	texPresenceFlag ? 
		uintBitsToFloat(uvec3(registers[data.x],registers[data.x+1u],registers[data.x+2u])) :
#endif
		irr_glsl_decodeRGB19E7(data);
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
vec3 bsdf_data_getParam3(in bsdf_data_t data, in bool texPresence)
{
#ifdef PARAM3_ALWAYS_SAME_VALUE
	return PARAM3_VALUE;
#else
	return textureOrRGBconst(data.data[1].xy, texPresence);
#endif
}
vec3 bsdf_data_getParam4(in bsdf_data_t data, in bool texPresence)
{
#ifdef PARAM4_ALWAYS_SAME_VALUE
	return PARAM4_VALUE;
#else
	return textureOrRGBconst(data.data[1].zw, texPresence);
#endif
}

bvec4 instr_getTexPresence(in instr_t i)
{
	return bvec4(
		instr_get1stParamTexPresence(i),
		instr_get2ndParamTexPresence(i),
		instr_get3rdParamTexPresence(i),
		instr_get4thParamTexPresence(i)
	);
}
params_t instr_getParameters(in instr_t i, in bsdf_data_t data)
{
	params_t p;
	bvec4 presence = instr_getTexPresence(i);
	//speculatively always read RGB
	p[0] = bsdf_data_getParam1(data, presence.x);
	p[1] = bsdf_data_getParam2(data, presence.y);
	p[2] = bsdf_data_getParam3(data, presence.z);
	p[3] = bsdf_data_getParam4(data, presence.w);

	return p;
}

//this should thought better
mat2x3 bsdf_data_decodeIoR(in bsdf_data_t data, in uint op)
{
	mat2x3 ior = mat2x3(0.0);
	ior[0] = irr_glsl_decodeRGB19E7(data.data[2].xy);
#ifdef OP_CONDUCTOR
	ior[1] = (op == OP_CONDUCTOR) ? irr_glsl_decodeRGB19E7(data.data[2].zw) : vec3(0.0);
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
	irr_glsl_IsotropicViewSurfaceInteraction interaction = irr_glsl_calcFragmentShaderSurfaceInteractionFromViewVector(V, pos, N);
	currInteraction = irr_glsl_calcAnisotropicInteraction(interaction);
}
void setCurrInteraction(in MC_precomputed_t precomp)
{
	setCurrInteraction(precomp.N, precomp.V, precomp.pos);
}
void updateCurrInteraction(in MC_precomputed_t precomp, in vec3 N, in bool ts)
{
#ifdef NO_TWOSIDED
	setCurrInteraction(precomp);
#else
	setCurrInteraction((ts && precomp.NdotV<0.0) ? -N : N, precomp.V, precomp.pos);
#endif
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
vec3 params_getOpacity(in params_t p)
{
	return p[PARAMS_OPACITY_IX];
}
float params_getAlphaV(in params_t p)
{
	return max(p[PARAMS_ALPHA_V_IX].x,MIN_ALPHA);
}
vec3 params_getSigmaA(in params_t p)
{
	return p[PARAMS_SIGMA_A_IX];
}
float params_getBlendWeight(in params_t p)
{
	return p[PARAMS_WEIGHT_IX].x;
}
vec3 params_getTransmittance(in params_t p)
{
	return p[PARAMS_TRANSMITTANCE_IX];
}


bxdf_eval_t instr_execute_cos_eval_DIFFUSE(in instr_t instr, in irr_glsl_LightSample s, in params_t params, in bsdf_data_t data)
{
	vec3 refl = params_getReflectance(params);
	float a = params_getAlpha(params);
	vec3 diffuse = irr_glsl_oren_nayar_cos_eval(s,currInteraction.isotropic,a*a) * refl;
	return diffuse;
}

bxdf_eval_t instr_execute_cos_eval_DIFFTRANS(in instr_t instr, in irr_glsl_LightSample s, in params_t params, in bsdf_data_t data)
{
	vec3 tr = params_getTransmittance(params);
	//transmittance*cos/2pi
	vec3 c = abs(s.NdotL)*irr_glsl_RECIPROCAL_PI*0.5*tr;
	return c;
}

bxdf_eval_t instr_execute_cos_eval_DIELECTRIC(in instr_t instr, in irr_glsl_LightSample s, in float eval)
{
	return bxdf_eval_t(eval);
}

bxdf_eval_t instr_execute_cos_eval_THINDIELECTRIC(in instr_t instr, in irr_glsl_LightSample s, in params_t params, in bsdf_data_t data)
{
	/*if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		//float au = params_getAlpha(params);
		//float av = params_getAlphaV(params);
		vec3 eta = vec3(1.5);
		vec3 diffuse = irr_glsl_lambertian_cos_eval(currBSDFParams.isotropic,currInteraction.isotropic) * vec3(0.89);
		diffuse *= irr_glsl_diffuseFresnelCorrectionFactor(eta,eta*eta) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currInteraction.isotropic.NdotV)) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currBSDFParams.isotropic.NdotL));
		writeReg(REG_DST(regs), diffuse);
	}
	else*/ 
	return bxdf_eval_t(0.0);
}
eval_and_pdf_t instr_execute_cos_eval_pdf_THINDIELECTRIC(in instr_t instr, in irr_glsl_LightSample s, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	bxdf_eval_t eval = instr_execute_cos_eval_THINDIELECTRIC(instr, s, params, data);
	//WARNING 1.0 instead of INF
	return eval_and_pdf_t(eval, 1.0);
}

bxdf_eval_t instr_execute_cos_eval_CONDUCTOR(in instr_t instr, in mat2x3 eta, in irr_glsl_LightSample s, in irr_glsl_AnisotropicMicrofacetCache microfacet, in float DG, in params_t params, in bsdf_data_t data)
{
	vec3 fr = irr_glsl_fresnel_conductor(eta[0],eta[1],microfacet.isotropic.VdotH);
	return DG*fr;
}

bxdf_eval_t instr_execute_cos_eval_COATING(in instr_t instr, in mat2x4 srcs, in params_t params, in vec3 eta, in vec3 eta2, in irr_glsl_LightSample s, in irr_glsl_AnisotropicMicrofacetCache microfacet, in bsdf_data_t data, out float out_weight)
{
	//vec3 thickness_sigma = params_getSigmaA(params);
	
	vec3 ws = irr_glsl_fresnel_dielectric_frontface_only(eta, microfacet.isotropic.VdotH);
	// TODO include thickness_sigma in diffuse weight computation: exp(sigma_thickness * freePath)
	// freePath = ( sqrt(refract_compute_NdotT2(NdotL2, rcpOrientedEta2)) + sqrt(refract_compute_NdotT2(NdotV2, rcpOrientedEta2)) )
	vec3 fresnelNdotV = irr_glsl_fresnel_dielectric_frontface_only(eta, currInteraction.isotropic.NdotV);
	vec3 wd = irr_glsl_diffuseFresnelCorrectionFactor(eta, eta2) * (vec3(1.0) - fresnelNdotV) * (vec3(1.0) - irr_glsl_fresnel_dielectric_frontface_only(eta, s.NdotL));

	bxdf_eval_t coat = srcs[0].xyz;
	bxdf_eval_t coated = srcs[1].xyz;

	out_weight = dot(fresnelNdotV, CIE_XYZ_Luma_Y_coeffs);

	return coat*ws + coated*wd;
}

eval_and_pdf_t instr_execute_cos_eval_pdf_COATING(in instr_t instr, in mat2x4 srcs, in params_t params, in vec3 eta, in vec3 eta2, in irr_glsl_LightSample s, in irr_glsl_AnisotropicMicrofacetCache microfacet, in bsdf_data_t data)
{
	//float thickness = uintBitsToFloat(data.data[2].z);

	float weight;
	bxdf_eval_t bxdf = instr_execute_cos_eval_COATING(instr, srcs, params, eta, eta2, s, microfacet, data, weight);
	float coat_pdf = srcs[0].w;
	float coated_pdf = srcs[1].w;

	float pdf = mix(coated_pdf, coat_pdf, weight);

	return eval_and_pdf_t(bxdf, pdf);
}

void instr_execute_BUMPMAP(in instr_t instr, in mat2x4 srcs, in MC_precomputed_t precomp)
{
	vec3 N = srcs[0].xyz;
	bool ts = instr_getTwosided(instr);
	updateCurrInteraction(precomp, N, ts);
}

void instr_execute_SET_GEOM_NORMAL(in instr_t instr, in MC_precomputed_t precomp)
{
	bool ts = instr_getTwosided(instr);
	updateCurrInteraction(precomp, precomp.N, ts);
}

bxdf_eval_t instr_execute_cos_eval_BLEND(in instr_t instr, in mat2x4 srcs, in params_t params, in bsdf_data_t data)
{
	float w = params_getBlendWeight(params);
	bxdf_eval_t bxdf1 = srcs[0].xyz;
	bxdf_eval_t bxdf2 = srcs[1].xyz;

	bxdf_eval_t blend = mix(bxdf1, bxdf2, w);
	return blend;
}
eval_and_pdf_t instr_execute_cos_eval_pdf_BLEND(in instr_t instr, in mat2x4 srcs, in params_t params, in bsdf_data_t data)
{
	float w = params_getBlendWeight(params);
	eval_and_pdf_t bxdf1 = srcs[0];
	eval_and_pdf_t bxdf2 = srcs[1];

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
	
	return mix(bxdf1, bxdf2, w);
}

vec3 fetchTex(in uvec3 texid, in vec2 uv, in mat2 dUV)
{
	float scale = uintBitsToFloat(texid.z);

	return irr_glsl_vTextureGrad(texid.xy, uv, dUV).rgb*scale;
}

void runTexPrefetchStream(in instr_stream_t stream, in vec2 uv, in mat2 dUV)
{
	for (uint i = 0u; i < stream.count; ++i)
	{
		prefetch_instr_t instr = irr_glsl_MC_fetchPrefetchInstr(stream.offset+i);

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

void runNormalPrecompStream(in instr_stream_t stream, in mat2 dUV)
{
	//either add MC_precomputed_t param and move runNormalPrecompStream() call to irr_computeLighting or leave it here
	vec3 pos = irr_glsl_MC_getWorldSpacePosition();
	setCurrInteraction(irr_glsl_MC_getNormalizedWorldSpaceN(), irr_glsl_MC_getNormalizedWorldSpaceV(), pos);
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);

		bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);

		uint srcreg = bsdf_data.data[0].x;
		uint dstreg = REG_DST(instr_decodeRegisters(instr));

		vec2 dh = readReg2(srcreg);
		
		writeReg(dstreg,
			irr_glsl_perturbNormal_derivativeMap(currInteraction.isotropic.N, dh, currInteraction.isotropic.V.dPosdScreen, dUV)
		);
	}
}

void flipCurrInteraction()
{
	currInteraction.isotropic.N = -currInteraction.isotropic.N;
	currInteraction.isotropic.NdotV = -currInteraction.isotropic.NdotV;
	//currInteraction.T = -currInteraction.T;
	currInteraction.B = -currInteraction.B;
	//currInteraction.TdotV = -currInteraction.TdotV;
	currInteraction.BdotV = -currInteraction.BdotV;
}
//call before executing an instruction/evaluating bsdf
void handleTwosided_interactionOnly(in instr_t instr)
{
#ifndef NO_TWOSIDED
	if (instr_getTwosided(instr) && currInteraction.isotropic.NdotV<0.0)
	{
		flipCurrInteraction();
	}
#endif	
}
//call before executing an instruction/evaluating bsdf
void handleTwosided(in instr_t instr, inout irr_glsl_LightSample s, inout irr_glsl_AnisotropicMicrofacetCache microfacet)
{
#ifndef NO_TWOSIDED
	if (instr_getTwosided(instr) && currInteraction.isotropic.NdotV<0.0)
	{
		flipCurrInteraction();

		s.NdotL = -s.NdotL;
		//s.TdotL = -s.TdotL;
		s.BdotL = -s.BdotL;
		microfacet.isotropic.NdotH = -microfacet.isotropic.NdotH;
		//microfacet.TdotH = -microfacet.TdotH;
		microfacet.BdotH = -microfacet.BdotH;
	}
#endif
}

//TODO move or rename those
//ggx
float irr_glsl_ggx_height_correlated_cos_eval_DG(in float NdotH, in float NdotL, in float NdotV, in float a2)
{
	float NdotH2 = NdotH*NdotH;
	float maxNdotL = max(NdotL, 0.0);
	float NdotL2 = NdotL*NdotL;
	float maxNdotV = max(NdotV,0.0);
	float NdotV2 = NdotV*NdotV;

	return irr_glsl_ggx_height_correlated_cos_eval_DG_wo_clamps(NdotH2, maxNdotL, NdotL2, maxNdotV, NdotV2, a2);
}
float irr_glsl_ggx_height_correlated_cos_eval_DG(in irr_glsl_LightSample s, in irr_glsl_IsotropicMicrofacetCache microfacet, in irr_glsl_IsotropicViewSurfaceInteraction i, in float a2)
{
	return irr_glsl_ggx_height_correlated_cos_eval_DG(
		microfacet.NdotH,
		s.NdotL,
		i.NdotV,
		a2
	);
}

float irr_glsl_ggx_height_correlated_aniso_cos_eval_DG(in float NdotH, in float TdotH, in float BdotH, in float NdotL, in float TdotL, in float BdotL, in float NdotV, in float TdotV, in float BdotV, in float ax, in float ax2, in float ay, in float ay2)
{
	float NdotH2 = NdotH * NdotH;
	float TdotH2 = TdotH * TdotH;
	float BdotH2 = BdotH * BdotH;
	float maxNdotL = max(NdotL, 0.0);
	float NdotL2 = NdotL * NdotL;
	float TdotL2 = TdotL * TdotL;
	float BdotL2 = BdotL * BdotL;
	float maxNdotV = max(NdotV, 0.0);
	float NdotV2 = NdotV * NdotV;
	float TdotV2 = TdotV * TdotV;
	float BdotV2 = BdotV * BdotV;

	return irr_glsl_ggx_height_correlated_aniso_cos_eval_DG_wo_clamps(NdotH2, TdotH2, BdotH2, maxNdotL, NdotL2, TdotL2, BdotL2, maxNdotV, NdotV2, TdotV2, BdotV2, ax, ax2, ay ,ay2);
}
float irr_glsl_ggx_height_correlated_aniso_cos_eval_DG(in irr_glsl_LightSample s, in irr_glsl_AnisotropicMicrofacetCache microfacet, in irr_glsl_AnisotropicViewSurfaceInteraction i, in float ax, in float ax2, in float ay, in float ay2)
{
	return irr_glsl_ggx_height_correlated_aniso_cos_eval_DG(
		microfacet.isotropic.NdotH,
		microfacet.TdotH,
		microfacet.BdotH,
		s.NdotL,
		s.TdotL,
		s.BdotL,
		i.isotropic.NdotV,
		i.TdotV,
		i.BdotV,
		ax, ax2, ay, ay2
	);
}

//beckmann
float irr_glsl_beckmann_height_correlated_cos_eval_DG(in float NdotH, in float NdotL, in float NdotV, in float a2)
{
	float NdotH2 = NdotH*NdotH;
	float NdotL2 = NdotL*NdotL;
	float NdotV2 = NdotV*NdotV;

	return irr_glsl_beckmann_height_correlated_cos_eval_DG_wo_clamps(NdotH2, NdotL2, NdotV2, a2);
}
float irr_glsl_beckmann_height_correlated_cos_eval_DG(in irr_glsl_LightSample s, in irr_glsl_IsotropicMicrofacetCache microfacet, in irr_glsl_IsotropicViewSurfaceInteraction i, in float a2)
{
	return irr_glsl_beckmann_height_correlated_cos_eval_DG(microfacet.NdotH, s.NdotL, i.NdotV, a2);
}

float irr_glsl_beckmann_aniso_height_correlated_cos_eval_DG(in float NdotH, in float TdotH, in float BdotH, in float NdotL, in float TdotL, in float BdotL, in float NdotV, in float TdotV, in float BdotV, in float ax, in float ax2, in float ay, in float ay2)
{
	return irr_glsl_beckmann_aniso_height_correlated_cos_eval_DG_wo_clamps(
		NdotH*NdotH,
		TdotH*TdotH,
		BdotH*BdotH,
		NdotL*NdotL,
		TdotL*TdotL,
		BdotL*BdotL,
		NdotV*NdotV,
		TdotV*TdotV,
		BdotV*BdotV,
		ax, ax2, ay, ay2
	);
}
float irr_glsl_beckmann_aniso_height_correlated_cos_eval_DG(in irr_glsl_LightSample s, in irr_glsl_AnisotropicMicrofacetCache microfacet, in irr_glsl_AnisotropicViewSurfaceInteraction i, in float ax, in float ax2, in float ay, in float ay2)
{
	return irr_glsl_beckmann_aniso_height_correlated_cos_eval_DG(
		microfacet.isotropic.NdotH,
		microfacet.TdotH,
		microfacet.BdotH,
		s.NdotL,
		s.TdotL,
		s.BdotL,
		i.isotropic.NdotV,
		i.TdotV,
		i.BdotV,
		ax, ax2, ay, ay2
	);
}

//blinn-phong
float irr_glsl_blinn_phong_cos_eval_DG(in float NdotH, in float NdotV, in float NdotL, in float n, in float a2)
{
	return irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(NdotH, NdotV*NdotV, NdotL*NdotL, n, a2);
}
float irr_glsl_blinn_phong_cos_eval_DG(in irr_glsl_LightSample s, in irr_glsl_IsotropicMicrofacetCache microfacet, in irr_glsl_IsotropicViewSurfaceInteraction i, in float n, in float a2)
{
	return irr_glsl_blinn_phong_cos_eval_DG(microfacet.NdotH, i.NdotV, s.NdotL, n, a2);
}

float irr_glsl_blinn_phong_cos_eval_DG(in float NdotH, in float TdotH, in float BdotH, in float NdotL, in float TdotL, in float BdotL, in float NdotV, in float TdotV, in float BdotV, in float nx, in float ny, in float ax2, in float ay2)
{
	return irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(NdotH, NdotH * NdotH, TdotH * TdotH, BdotH * BdotH, TdotL * TdotL, BdotL * BdotL, TdotV * TdotV, BdotV * BdotV, NdotV * NdotV, NdotL * NdotL, nx, ny, ax2, ay2);
}
float irr_glsl_blinn_phong_cos_eval_DG(in irr_glsl_LightSample s, in irr_glsl_AnisotropicMicrofacetCache microfacet, in irr_glsl_AnisotropicViewSurfaceInteraction i, in float nx, in float ny, in float ax2, in float ay2)
{
	return irr_glsl_blinn_phong_cos_eval_DG(
		microfacet.isotropic.NdotH,
		microfacet.TdotH,
		microfacet.BdotH,
		s.NdotL,
		s.TdotL,
		s.BdotL,
		i.isotropic.NdotV,
		i.TdotV,
		i.BdotV,
		nx, ny, ax2, ay2
	);
}

#ifdef GEN_CHOICE_STREAM
// sample and microfacet are inout because of handleTwosided() only
void instr_eval_and_pdf_execute(in instr_t instr, in MC_precomputed_t precomp, inout irr_glsl_LightSample s, inout irr_glsl_AnisotropicMicrofacetCache _microfacet)
{
	uint op = instr_getOpcode(instr);

	//speculative execution
	bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
	params_t params = instr_getParameters(instr, bsdf_data);
	uint ndf = instr_getNDF(instr);
	float a = params_getAlpha(params);
	float a2 = a*a;
	float ay = params_getAlphaV(params);
	float ay2 = ay*ay;

	float bxdf_eval_scalar_part = 0.0;
	vec3 bxdf_eval = vec3(0.0);
	float pdf = 1.0;

	const mat2x3 ior = bsdf_data_decodeIoR(bsdf_data, op);
	const mat2x3 ior2 = matrixCompMult(ior, ior);
	const float ior_scalar = dot(CIE_XYZ_Luma_Y_coeffs, ior[0]);
	const bool is_bsdf = !op_isBRDF(op); //note it actually tells if op is BSDF or BUMPMAP or SET_GEOM_NORMAL (divergence reasons)
	const vec3 refl = params_getReflectance(params);

#ifndef NO_TWOSIDED
	handleTwosided(instr, s, _microfacet);
#endif


	irr_glsl_AnisotropicMicrofacetCache microfacet;
	bool is_valid = true;
	bool refraction = false;
#ifndef NO_BSDF
	//here actually using stronger check for BSDF because it's probably worth it
	if (op_isBSDF(op) && irr_glsl_isTransmissionPath(currInteraction.isotropic.NdotV, s.NdotL))
	{
		float orientedEta, rcpOrientedEta;
		irr_glsl_getOrientedEtas(orientedEta, rcpOrientedEta, currInteraction.isotropic.NdotV, ior_scalar);
		is_valid = irr_glsl_calcAnisotropicMicrofacetCache(microfacet, true, currInteraction.isotropic.V.dir, s.L, currInteraction.T, currInteraction.B, currInteraction.isotropic.N, s.NdotL, s.VdotL, orientedEta, rcpOrientedEta);
		refraction = true;
	}
	else
#endif
	{
		microfacet = _microfacet;
	}

#if defined(OP_THINDIELECTRIC)
	if (op == OP_THINDIELECTRIC)
	{
		bxdf_eval = vec3(0.0);
		pdf = 1.0;
	} else
#endif
#if defined(OP_DIFFUSE) || defined(OP_DIFFTRANS)
	if (op_isDiffuse(op))
	{
		vec3 reflectance = is_bsdf ? vec3(1.0) : refl;
		float alpha2 = is_bsdf ? 0.0 : a2;
		bxdf_eval = reflectance*irr_glsl_oren_nayar_cos_remainder_and_pdf(pdf, s, currInteraction.isotropic, alpha2);
		pdf *= is_bsdf ? 0.5 : 1.0;
		bxdf_eval *= pdf;
	} else
#endif
#if defined(OP_CONDUCTOR) || defined(OP_DIELECTRIC)
	if (op_hasSpecular(op))
	{
		const float TdotV2 = currInteraction.TdotV * currInteraction.TdotV;
		const float BdotV2 = currInteraction.BdotV * currInteraction.BdotV;
		const float NdotV2 = currInteraction.isotropic.NdotV_squared;

		const float TdotH2 = microfacet.TdotH * microfacet.TdotH;
		const float BdotH2 = microfacet.BdotH * microfacet.BdotH;
		const float NdotH2 = microfacet.isotropic.NdotH2;
		const float NdotL = irr_glsl_conditionalAbsOrMax(is_bsdf, s.NdotL, 0.0);
		const float NdotV = irr_glsl_conditionalAbsOrMax(is_bsdf, currInteraction.isotropic.NdotV, 0.0);

		float G1_over_2NdotV = 0.0;
		float G2_over_G1 = 0.0;
		float ndf_val = 0.0;

		BEGIN_CASES(ndf)
#ifdef NDF_GGX
		CASE_BEGIN(ndf, NDF_GGX) {
			G1_over_2NdotV = irr_glsl_GGXSmith_G1_wo_numerator(NdotV, TdotV2, BdotV2, NdotV2, a2, ay2);
			G2_over_G1 = irr_glsl_ggx_smith_G2_over_G1(NdotL, s.TdotL*s.TdotL, s.BdotL*s.BdotL, s.NdotL*s.NdotL, NdotV, TdotV2, BdotV2, NdotV2, a2, ay2);
			ndf_val = irr_glsl_ggx_aniso(TdotH2, BdotH2, NdotH2, a, ay, a2, ay2);
		} CASE_END
#endif

#ifdef NDF_BECKMANN
		CASE_BEGIN(ndf, NDF_BECKMANN) {
			float lambdaV = irr_glsl_smith_beckmann_Lambda(TdotV2, BdotV2, NdotV2, a2, ay2);
			G1_over_2NdotV = irr_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
			G2_over_G1 = irr_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, s.TdotL*s.TdotL, s.BdotL*s.BdotL, s.NdotL*s.NdotL, a2, ay2);
			ndf_val = irr_glsl_beckmann(ax, ay, a2, ay2, TdotH2, BdotH2, NdotH2);
		} CASE_END
#endif

#ifdef NDF_PHONG
		CASE_BEGIN(ndf, NDF_PHONG) {
			float lambdaV = irr_glsl_smith_beckmann_Lambda(TdotV2, BdotV2, NdotV2, ax2, ay2);
			G1_over_2NdotV = irr_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
			G2_over_G1 = irr_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, s.TdotL*s.TdotL, s.BdotL*s.BdotL, s.NdotL*s.NdotL, ax2, ay2);
			ndf_val = irr_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);
		} CASE_END
#endif
		CASE_OTHERWISE
		{} //else "empty braces"
		END_CASES

		bxdf_eval_scalar_part = 0.5 * G2_over_G1 * G1_over_2NdotV * ndf_val; //cos*D*G2/(4*cos*cos)
		pdf = irr_glsl_smith_VNDF_pdf_wo_clamps(ndf_val, G1_over_2NdotV);

		float VdotH = microfacet.isotropic.VdotH;
		float VdotH_clamp = irr_glsl_conditionalAbsOrMax(is_bsdf, VdotH, 0.0);
		vec3 fr;
#ifdef OP_CONDUCTOR
		if (op == OP_CONDUCTOR)
			fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH_clamp);
		else
#endif
			fr = irr_glsl_fresnel_dielectric_common(ior2[0], VdotH_clamp);

		if (is_bsdf && is_valid)
		{
			float eta = ior_scalar;
			float orientedEta, rcpOrientedEta;
			irr_glsl_getOrientedEtas(orientedEta, rcpOrientedEta, currInteraction.isotropic.NdotV, eta);

			float LdotH = microfacet.isotropic.LdotH;
			float VdotHLdotH = VdotH * LdotH;
#ifdef NDF_GGX
			if (ndf == NDF_GGX)
				bxdf_eval_scalar_part = irr_glsl_ggx_microfacet_to_light_measure_transform(bxdf_eval_scalar_part, NdotL, refraction, VdotH, LdotH, VdotHLdotH, orientedEta);
			else
#endif
				bxdf_eval_scalar_part = irr_glsl_microfacet_to_light_measure_transform(bxdf_eval_scalar_part, NdotV, refraction, VdotH, LdotH, VdotHLdotH, orientedEta);
		}
			
		bxdf_eval = is_valid ? vec3(bxdf_eval_scalar_part) : vec3(0.0);

#ifdef OP_CONDUCTOR
		if (op == OP_CONDUCTOR)
			bxdf_eval *= fr;
		else
#endif
		{
			float reflectance = dot(CIE_XYZ_Luma_Y_coeffs, fr);
			reflectance = refraction ? (1.0 - reflectance) : reflectance;
			pdf *= reflectance;
			bxdf_eval *= reflectance;
		}
	} else
#endif
	{}

	uvec3 regs = instr_decodeRegisters(instr);

	eval_and_pdf_t result = eval_and_pdf_t(bxdf_eval, pdf);
	if (!op_isBXDF(op))
	{
		mat2x4 srcs = instr_fetchSrcRegs(instr, regs);
		BEGIN_CASES(op)
#ifdef OP_COATING
		CASE_BEGIN(op, OP_COATING) {
			result = instr_execute_cos_eval_pdf_COATING(instr, srcs, params, ior[0], ior2[0], s, microfacet, bsdf_data);
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

	if (op_isBXDForCoatOrBlend(op))
		writeReg(REG_DST(regs), result);
}

eval_and_pdf_t irr_bsdf_eval_and_pdf(in MC_precomputed_t precomp, in instr_stream_t stream, inout irr_glsl_LightSample s, in uint generator_offset, inout irr_glsl_AnisotropicMicrofacetCache microfacet)
{
	setCurrInteraction(precomp);
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);
		uint op = instr_getOpcode(instr);

		if (i != generator_offset)
		{
			instr_eval_and_pdf_execute(instr, precomp, s, microfacet);
		}
		else
		{
			//assert(isBXDF(instr));
			uvec3 regs = instr_decodeRegisters(instr);
			writeReg(REG_DST(regs), eval_and_pdf_t(0.0));
		}

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
			s = irr_glsl_createLightSample(s.L, currInteraction);
			microfacet = irr_glsl_calcAnisotropicMicrofacetCache(currInteraction, s);
		}
#endif
	}

	eval_and_pdf_t eval_and_pdf = readReg4(0u);
	return eval_and_pdf;
}

irr_glsl_AnisotropicMicrofacetCache getSmoothMicrofacetCache(in float NdotV, in float NdotL)
{
	irr_glsl_AnisotropicMicrofacetCache microfacet;
	const float d = 1e-4;//some delta to avoid zeroes... not a perfect solution but for now..
	microfacet.isotropic.VdotH = NdotV-d;
	microfacet.isotropic.LdotH = NdotL-d;
	microfacet.isotropic.NdotH = d;
	microfacet.isotropic.NdotH2 = d*d;
	microfacet.TdotH = 1.0-d;
	microfacet.BdotH = 1.0-d;

	return microfacet;
}

irr_glsl_LightSample irr_bsdf_cos_generate(in MC_precomputed_t precomp, in instr_stream_t stream, in vec3 rand, out vec3 out_remainder, out float out_pdf, out irr_glsl_AnisotropicMicrofacetCache out_microfacet, out uint out_gen_rnpOffset, out bool out_invalid_microfacet)
{
	uint ix = 0u;
	instr_t instr = irr_glsl_MC_fetchInstr(stream.offset);
	uint op = instr_getOpcode(instr);
	float weight = 1.0;
	vec3 u = rand;

	setCurrInteraction(precomp);
	bool is_plastic = false;
	while (!op_isBXDF(op))
	{
		handleTwosided_interactionOnly(instr);

#ifdef OP_BLEND
		if (op==OP_BLEND) {
			bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
			params_t params = instr_getParameters(instr, bsdf_data);
			float w = params_getBlendWeight(params);
			float rcpChoiceProb;
			bool choseRight = irr_glsl_partitionRandVariable(w, u.z, rcpChoiceProb);

			uint right = instr_getRightJump(instr);
			ix = choseRight ? right:(ix+1u);
			weight /= rcpChoiceProb;
		}
		else 
#endif //OP_BLEND
#ifdef OP_COATING
		if (op==OP_COATING) {
			bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
			vec3 eta = bsdf_data_decodeIoR(bsdf_data, OP_COATING)[0];
			vec3 fresnel = irr_glsl_fresnel_dielectric_frontface_only(eta, currInteraction.isotropic.NdotV);
			float w = dot(fresnel, CIE_XYZ_Luma_Y_coeffs);
			float rcpChoiceProb;
			bool choseCoated = irr_glsl_partitionRandVariable(w, u.z, rcpChoiceProb);

			uint coated_ix = instr_getRightJump(instr);
			ix = choseCoated ? coated_ix:(ix+1u);
			weight /= rcpChoiceProb;

			is_plastic = true;
		} else
#endif //OP_COATING
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

		instr = irr_glsl_MC_fetchInstr(stream.offset+ix);
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
	const bool is_bsdf = !op_isBRDF(op) && !is_plastic;
	const vec3 refl = params_getReflectance(params);

#ifndef NO_TWOSIDED
	handleTwosided_interactionOnly(instr);
#endif //NO_TWOSIDED

	float pdf = 1.0;
	vec3 rem = vec3(1.0);
	uint ndf = instr_getNDF(instr);
	irr_glsl_LightSample s;

	const vec3 localV = irr_glsl_getTangentSpaceV(currInteraction);
	const mat3 tangentFrame = irr_glsl_getTangentFrame(currInteraction);

#ifdef OP_THINDIELECTRIC
	if (op == OP_THINDIELECTRIC)
	{
		const vec3 luminosityContributionHint = CIE_XYZ_Luma_Y_coeffs;
		s = irr_glsl_thin_smooth_dielectric_cos_generate(currInteraction, u, ior2[0], luminosityContributionHint);
		out_microfacet = irr_glsl_calcAnisotropicMicrofacetCache(currInteraction, s);
		rem = irr_glsl_thin_smooth_dielectric_cos_remainder_and_pdf(pdf, s, currInteraction.isotropic, ior2[0], luminosityContributionHint);
		pdf = 1.0;
	} else
#endif
#if defined(OP_DIFFUSE) || defined(OP_DIFFTRANS)
	if (op_isDiffuse(op))
	{
		vec3 localL = irr_glsl_projected_hemisphere_generate(u.xy);
		if (is_bsdf)
		{
			float dummy;
			bool flip = irr_glsl_partitionRandVariable(0.5, u.z, dummy);
			localL = flip ? -localL : localL;
		}
		s = irr_glsl_createLightSampleTangentSpace(localV, localL, tangentFrame);
		out_microfacet = irr_glsl_calcAnisotropicMicrofacetCache(currInteraction, s);

		const float alpha2 = is_bsdf ? 0.0 : ax2;
		const vec3 reflectance = is_bsdf ? vec3(1.0) : refl;
		rem *= reflectance*irr_glsl_oren_nayar_cos_remainder_and_pdf(pdf, s, currInteraction.isotropic, alpha2);
		pdf *= is_bsdf ? 0.5 : 1.0;
	} else
#endif
#if defined(OP_CONDUCTOR) || defined(OP_COATING) || defined(OP_DIELECTRIC)
	if (op_hasSpecular(op))
	{
		const float TdotV2 = currInteraction.TdotV * currInteraction.TdotV;
		const float BdotV2 = currInteraction.BdotV * currInteraction.BdotV;
		const float NdotV2 = currInteraction.isotropic.NdotV_squared;

		const vec3 upperHemisphereLocalV = currInteraction.isotropic.NdotV < 0.0 ? -localV : localV;

		float G2_over_G1 = 0.0;
		float G1_over_2NdotV = 0.0;
		float ndf_val = 0.0;
		vec3 localH = vec3(0.0);

		BEGIN_CASES(ndf)
#ifdef NDF_GGX
		CASE_BEGIN(ndf, NDF_GGX) 
		{
			// why is it called without "wo_clamps" and beckmann sampling is?
			localH = irr_glsl_ggx_cos_generate(upperHemisphereLocalV, u.xy, ax, ay);
		} CASE_END
#endif //NDF_GGX

#ifdef NDF_BECKMANN
		CASE_BEGIN(ndf, NDF_BECKMANN) 
		{
			localH = irr_glsl_beckmann_cos_generate_wo_clamps(upperHemisphereLocalV, u.xy, ax, ay);
		} CASE_END
#endif //NDF_BECKMANN

#ifdef NDF_PHONG
		CASE_BEGIN(ndf, NDF_PHONG) 
		{
			localH = irr_glsl_beckmann_cos_generate_wo_clamps(upperHemisphereLocalV, u.xy, ax, ay);
		} CASE_END
#endif //NDF_PHONG
		CASE_OTHERWISE
		{}
		END_CASES

		vec3 localL;
		float VdotH = dot(localH, localV);
		float VdotH_clamp = irr_glsl_conditionalAbsOrMax(is_bsdf, VdotH, 0.0);
		vec3 fr;
#ifdef OP_CONDUCTOR
		if (op == OP_CONDUCTOR)
			fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH_clamp);
		else
#endif
			fr = irr_glsl_fresnel_dielectric_common(ior[0]*ior[0], VdotH_clamp);
		const float refractionProb = dot(CIE_XYZ_Luma_Y_coeffs, fr);
		float rcpChoiceProb;
		const bool refraction = is_bsdf ? irr_glsl_partitionRandVariable(refractionProb, u.z, rcpChoiceProb) : false;
		rcpChoiceProb = is_bsdf ? rcpChoiceProb : 1.0;
		rem *= rcpChoiceProb;
		rem *= refraction ? (vec3(1.0) - fr) : fr;
		pdf /= rcpChoiceProb;
		float eta = dot(CIE_XYZ_Luma_Y_coeffs, ior[0]);
		float orientedEta, rcpOrientedEta;
		irr_glsl_getOrientedEtas(orientedEta, rcpOrientedEta, currInteraction.isotropic.NdotV, eta);

		out_microfacet = irr_glsl_calcAnisotropicMicrofacetCache(refraction, localV, localH, localL, rcpOrientedEta, rcpOrientedEta*rcpOrientedEta);
		s = irr_glsl_createLightSampleTangentSpace(localV, localL, tangentFrame);

		const float TdotH2 = out_microfacet.TdotH * out_microfacet.TdotH;
		const float BdotH2 = out_microfacet.BdotH * out_microfacet.BdotH;
		const float NdotH2 = out_microfacet.isotropic.NdotH2;
		const float NdotL = irr_glsl_conditionalAbsOrMax(is_bsdf, s.NdotL, 0.0);
		const float NdotV = irr_glsl_conditionalAbsOrMax(is_bsdf, currInteraction.isotropic.NdotV, 0.0);

		BEGIN_CASES(ndf)
#ifdef NDF_GGX
		CASE_BEGIN(ndf, NDF_GGX)
		{
			G2_over_G1 = irr_glsl_ggx_smith_G2_over_G1(NdotL, s.TdotL*s.TdotL, s.BdotL*s.BdotL, s.NdotL*s.NdotL, NdotV, TdotV2, BdotV2, NdotV2, ax2, ay2);
			G1_over_2NdotV = irr_glsl_GGXSmith_G1_wo_numerator(NdotV, TdotV2, BdotV2, NdotV2, ax2, ay2);
			ndf_val = irr_glsl_ggx_aniso(TdotH2, BdotH2, NdotH2, ax, ay, ax2, ay2);
		} CASE_END
#endif //NDF_GGX

#ifdef NDF_BECKMANN
		CASE_BEGIN(ndf, NDF_BECKMANN)
		{
			float lambdaV = irr_glsl_smith_beckmann_Lambda(TdotV2, BdotV2, NdotV2, ax2, ay2);
			G1_over_2NdotV = irr_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
			G2_over_G1 = irr_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, s.TdotL*s.TdotL, s.BdotL*s.BdotL, s.NdotL*s.NdotL, ax2, ay2);
			ndf_val = irr_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);
		} CASE_END
#endif //NDF_BECKMANN

#ifdef NDF_PHONG
		CASE_BEGIN(ndf, NDF_PHONG)
		{
			float lambdaV = irr_glsl_smith_beckmann_Lambda(TdotV2, BdotV2, NdotV2, ax2, ay2);
			G1_over_2NdotV = irr_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
			G2_over_G1 = irr_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, s.TdotL * s.TdotL, s.BdotL * s.BdotL, s.NdotL * s.NdotL, ax2, ay2);
			ndf_val = irr_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);
		} CASE_END
#endif //NDF_PHONG
		CASE_OTHERWISE
		{}
		END_CASES

		rem *= G2_over_G1;
		pdf *= irr_glsl_smith_VNDF_pdf_wo_clamps(ndf_val, G1_over_2NdotV);
	} else
#endif
	{} //empty braces for `else`

	out_remainder = rem; // TODO not multiply rem by weight??
	out_pdf = weight*pdf; 

	return s;
}

vec3 runGenerateAndRemainderStream(in MC_precomputed_t precomp, in instr_stream_t gcs, in instr_stream_t rnps, in vec3 rand, out float out_pdf, out irr_glsl_LightSample out_smpl)
{
	instr_t generator;
	vec3 generator_rem;
	float generator_pdf;
	irr_glsl_AnisotropicMicrofacetCache microfacet;
	uint generator_rnpOffset;
	bool invalid_microfacet;
	irr_glsl_LightSample s = irr_bsdf_cos_generate(precomp, gcs, rand, generator_rem, generator_pdf, microfacet, generator_rnpOffset, invalid_microfacet);
	eval_and_pdf_t eval_pdf = irr_bsdf_eval_and_pdf(precomp, rnps, s, generator_rnpOffset, microfacet);
	bxdf_eval_t acc = eval_pdf.rgb;
	float restPdf = eval_pdf.a;
	float pdf = generator_pdf + restPdf;

	out_smpl = s;
	out_pdf = pdf;

	vec3 rem = (generator_rem + acc/generator_pdf) / (1.0 + restPdf/generator_pdf);

	return rem;
}

#endif //GEN_CHOICE_STREAM

#endif