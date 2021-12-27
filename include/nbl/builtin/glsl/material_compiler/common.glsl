#ifndef _NBL_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_INCLUDED_
#define _NBL_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_INCLUDED_

#include <nbl/builtin/glsl/material_compiler/common_declarations.glsl>

#ifndef _NBL_USER_PROVIDED_MATERIAL_COMPILER_GLSL_BACKEND_FUNCTIONS_
	#error "You need to define 'vec3 nbl_glsl_MC_getNormalizedWorldSpaceV()', 'vec3 nbl_glsl_MC_getNormalizedWorldSpaceN()' functions above"
	#ifdef TEX_PREFETCH_STREAM
		#error "as well as 'mat2x3 nbl_glsl_perturbNormal_dPdSomething()', and 'mat2 nbl_glsl_perturbNormal_dUVdSomething()'"
	#endif
#endif
#define _NBL_BUILTIN_GLSL_BUMP_MAPPING_DERIVATIVES_DECLARED_

#include <nbl/builtin/glsl/math/functions.glsl>
#include <nbl/builtin/glsl/format/decode.glsl>

// this all all the things we can precompute agnostic of the light sample
nbl_glsl_MC_precomputed_t nbl_glsl_MC_precomputeData(in bool frontface)
{
	nbl_glsl_MC_precomputed_t p;
	p.N = nbl_glsl_MC_getNormalizedWorldSpaceN();
	p.V = nbl_glsl_MC_getNormalizedWorldSpaceV();
	p.frontface = frontface;

	return p;
}

float nbl_glsl_MC_colorToScalar(in vec3 color)
{
	return dot(color, NBL_GLSL_MC_CIE_XYZ_Luma_Y_coeffs);
}

// Instruction Methods
// RnP = Remainder and PDF
uint nbl_glsl_MC_instr_getOffsetIntoRnPStream(in nbl_glsl_MC_instr_t instr)
{
	return bitfieldExtract(instr.y, int(INSTR_OFFSET_INTO_REMANDPDF_STREAM_SHIFT-32u), int(INSTR_OFFSET_INTO_REMANDPDF_STREAM_WIDTH));
}
uint nbl_glsl_MC_instr_getOpcode(in nbl_glsl_MC_instr_t instr)
{
	return instr.x&INSTR_OPCODE_MASK;
}
uint nbl_glsl_MC_instr_getBSDFbufOffset(in nbl_glsl_MC_instr_t instr)
{
	// if we allowed for variable size (or very padded) instructions, we wouldnt need to fetch bsdf data from offset
	// https://github.com/Devsh-Graphics-Programming/Nabla/issues/287
	return (instr.x>>INSTR_BSDF_BUF_OFFSET_SHIFT) & INSTR_BSDF_BUF_OFFSET_MASK;
}
uint nbl_glsl_MC_instr_getNDF(in nbl_glsl_MC_instr_t instr)
{
	return (instr.x>>INSTR_NDF_SHIFT) & INSTR_NDF_MASK;
}
uint nbl_glsl_MC_instr_getRightJump(in nbl_glsl_MC_instr_t instr)
{
	return bitfieldExtract(instr.y, int(INSTR_RIGHT_JUMP_SHIFT-32u), int(INSTR_RIGHT_JUMP_WIDTH));
}

// BSDFs can have at most 2 parameters come from textures
bool nbl_glsl_MC_instr_get1stParamTexPresence(in nbl_glsl_MC_instr_t instr)
{
#if defined(PARAM1_NEVER_TEX)
	return false;
#elif defined(PARAM1_ALWAYS_TEX)
	return true;
#else
	return (instr.x&(1u<<INSTR_1ST_PARAM_TEX_SHIFT)) != 0u;
#endif
}
bool nbl_glsl_MC_instr_get2ndParamTexPresence(in nbl_glsl_MC_instr_t instr)
{
#if defined(PARAM2_NEVER_TEX)
	return false;
#elif defined(PARAM2_ALWAYS_TEX)
	return true;
#else
	return (instr.x&(1u<<INSTR_2ND_PARAM_TEX_SHIFT)) != 0u;
#endif
}

// some texture parameters are mutually exclusive
bool nbl_glsl_MC_instr_params_getAlphaUTexPresence(in nbl_glsl_MC_instr_t instr)
{
	return nbl_glsl_MC_instr_get1stParamTexPresence(instr);
}
bool nbl_glsl_MC_instr_params_getAlphaVTexPresence(in nbl_glsl_MC_instr_t instr)
{
	return nbl_glsl_MC_instr_get2ndParamTexPresence(instr);
}
bool nbl_glsl_MC_instr_params_getReflectanceTexPresence(in nbl_glsl_MC_instr_t instr)
{
	return nbl_glsl_MC_instr_get2ndParamTexPresence(instr);
}
bool nbl_glsl_MC_instr_getSigmaATexPresence(in nbl_glsl_MC_instr_t instr)
{
	return nbl_glsl_MC_instr_get1stParamTexPresence(instr);
}
bool nbl_glsl_MC_instr_getTransmittanceTexPresence(in nbl_glsl_MC_instr_t instr)
{
	return nbl_glsl_MC_instr_get2ndParamTexPresence(instr);
}
bool nbl_glsl_MC_instr_getWeightTexPresence(in nbl_glsl_MC_instr_t instr)
{
	return nbl_glsl_MC_instr_get1stParamTexPresence(instr);
}

// works with tex prefetch instructions as well (x=reg0,y=reg1,z=reg2)
struct nbl_glsl_MC_RegID_t
{
	uint dst;
	uint srcA;
	uint srcB;
};
nbl_glsl_MC_RegID_t nbl_glsl_MC_instr_decodeRegisters(in nbl_glsl_MC_instr_t instr)
{
	uvec3 regs = instr.yyy >> (uvec3(INSTR_REG_DST_SHIFT,INSTR_REG_SRC1_SHIFT,INSTR_REG_SRC2_SHIFT)-32u);
	regs &= uvec3(INSTR_REG_MASK);
	nbl_glsl_MC_RegID_t retval;
	retval.dst = regs[0];
	retval.srcA = regs[1];
	retval.srcB = regs[2];
	return retval;
}

// if we allowed for variable size (or very padded) instructions, we wouldnt need to fetch bsdf data from offset
// https://github.com/Devsh-Graphics-Programming/Nabla/issues/287
nbl_glsl_MC_bsdf_data_t nbl_glsl_MC_fetchBSDFDataForInstr(in nbl_glsl_MC_instr_t instr)
{
	uint ix = nbl_glsl_MC_instr_getBSDFbufOffset(instr);
	return nbl_glsl_MC_fetchBSDFData(ix);
}

// texture prefetch instructions are a bit fatter, 128bit
uint nbl_glsl_MC_prefetch_instr_getRegCount(in nbl_glsl_MC_prefetch_instr_t instr)
{
	uint dword4 = instr.w;
	return bitfieldExtract(dword4, PREFETCH_INSTR_REG_CNT_SHIFT, PREFETCH_INSTR_REG_CNT_WIDTH);
}
uint nbl_glsl_MC_prefetch_instr_getDstReg(in nbl_glsl_MC_prefetch_instr_t instr)
{
	uint dword4 = instr.w;
	return bitfieldExtract(dword4, PREFETCH_INSTR_DST_REG_SHIFT, PREFETCH_INSTR_DST_REG_WIDTH);
}

// opcode methods
bool nbl_glsl_MC_op_isBRDF(in uint op)
{
	return op<=OP_MAX_BRDF;
}
bool nbl_glsl_MC_op_isBSDF(in uint op)
{
	return !nbl_glsl_MC_op_isBRDF(op) && op<=OP_MAX_BSDF;
}
bool nbl_glsl_MC_op_isBXDF(in uint op)
{
	return op<=OP_MAX_BSDF;
}
bool nbl_glsl_MC_op_isBXDForCoatOrBlend(in uint op)
{
#ifdef OP_BLEND
	return op<=OP_BLEND;
#elif defined(OP_COATING)
	return op<=OP_COATING;
#else
	return nbl_glsl_MC_op_isBXDF(op);
#endif
}
bool nbl_glsl_MC_op_hasSpecular(in uint op)
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
bool nbl_glsl_MC_op_isDiffuse(in uint op)
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
#ifdef TEX_PREFETCH_STREAM
#include <nbl/builtin/glsl/bump_mapping/utils.glsl>
#endif

// current interaction is a global (for now I guess)
nbl_glsl_MC_interaction_t currInteraction;
// methods to update the global
void nbl_glsl_MC_setCurrInteraction(in vec3 N, in vec3 V)
{
	nbl_glsl_IsotropicViewSurfaceInteraction interaction = nbl_glsl_calcSurfaceInteractionFromViewVector(V, N);
	currInteraction.inner = nbl_glsl_calcAnisotropicInteraction(interaction);
	nbl_glsl_MC_finalizeInteraction(currInteraction);
}
void nbl_glsl_MC_setCurrInteraction(in nbl_glsl_MC_precomputed_t precomp)
{
	nbl_glsl_MC_setCurrInteraction(precomp.frontface ? precomp.N : -precomp.N, precomp.V);
}
void nbl_glsl_MC_updateCurrInteraction(in nbl_glsl_MC_precomputed_t precomp, in vec3 N)
{
	// precomputed normals already have correct orientation
	nbl_glsl_MC_setCurrInteraction(N, precomp.V);
}


struct nbl_glsl_MC_aov_t
{
	nbl_glsl_MC_bxdf_spectrum_t albedo;
	float throughputFactor; // should we have it as a full vec3 for the duration of evaluation?
	vec3 normal;
};

// compute throughput factor from roughness
float nbl_glsl_MC_aov_t_specularThroughputFactor(float a2)
{
	// TODO: make the ramp tweakable
	return exp2(-128.f*a2);
}
float nbl_glsl_MC_aov_t_specularThroughputFactor(float ax2, float ay2)
{
	return nbl_glsl_MC_aov_t_specularThroughputFactor(ax2+ay2);
}


#define GEN_CHOICE_WITH_AOV_EXTRACTION 2
// return type depends on what we'll be doing
struct nbl_glsl_MC_eval_pdf_aov_t
{
	nbl_glsl_MC_bxdf_spectrum_t value;
#ifdef GEN_CHOICE_STREAM
	float pdf;
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	nbl_glsl_MC_aov_t aov;
#endif
#endif
};


// virtual registers
nbl_glsl_MC_reg_t registers[REG_COUNT];
// write
void nbl_glsl_MC_writeReg(in uint n, in float v)
{
	registers[n] = floatBitsToUint(v);
}
void nbl_glsl_MC_writeReg(in uint n, in vec2 v)
{
	nbl_glsl_MC_writeReg(n   ,v.x);
	nbl_glsl_MC_writeReg(n+1u,v.y);
}
void nbl_glsl_MC_writeReg(in uint n, in vec3 v)
{
	nbl_glsl_MC_writeReg(n   ,v.xy);
	nbl_glsl_MC_writeReg(n+2u,v.z);
}
void nbl_glsl_MC_writeReg(in uint n, in nbl_glsl_MC_eval_pdf_aov_t v)
{
	nbl_glsl_MC_writeReg(n   ,v.value);
#ifdef GEN_CHOICE_STREAM
	nbl_glsl_MC_writeReg(n+3u,v.pdf);
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	nbl_glsl_MC_writeReg(n+4u,v.aov.albedo);
	nbl_glsl_MC_writeReg(n+7u,v.aov.throughputFactor);
	nbl_glsl_MC_writeReg(n+8u,v.aov.normal);
#endif
#endif
}
// read
void nbl_glsl_MC_readReg(in uint n, out float v)
{
	v = uintBitsToFloat( registers[n] );
}
void nbl_glsl_MC_readReg(in uint n, out vec2 v)
{
	nbl_glsl_MC_readReg(n	,v.x);
	nbl_glsl_MC_readReg(n+1u,v.y);
}
void nbl_glsl_MC_readReg(in uint n, out vec3 v)
{
	nbl_glsl_MC_readReg(n	,v.xy);
	nbl_glsl_MC_readReg(n+2u,v.z);
}
void nbl_glsl_MC_readReg(in uint n, out nbl_glsl_MC_eval_pdf_aov_t v)
{
	nbl_glsl_MC_readReg(n	,v.value);
#ifdef GEN_CHOICE_STREAM
	nbl_glsl_MC_readReg(n+3u,v.pdf);
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	nbl_glsl_MC_readReg(n+4u,v.aov.albedo);
	nbl_glsl_MC_readReg(n+7u,v.aov.throughputFactor);
	nbl_glsl_MC_readReg(n+8u,v.aov.normal);
#endif
#endif
}

// when we finally know (or generate) our light sample we can precompute the rest of the angles
void nbl_glsl_MC_updateLightSampleAfterNormalChange(inout nbl_glsl_LightSample out_s)
{
	out_s.TdotL = dot(currInteraction.inner.T, out_s.L);
	out_s.BdotL = dot(currInteraction.inner.B, out_s.L);
	out_s.NdotL = dot(currInteraction.inner.isotropic.N, out_s.L);
	out_s.NdotL2 = out_s.NdotL*out_s.NdotL;
}
// not everything needs to be recomputed when `N` changes, only most things ;)
void nbl_glsl_MC_updateMicrofacetCacheAfterNormalChange(in nbl_glsl_LightSample s, inout nbl_glsl_MC_microfacet_t out_microfacet)
{
	const float NdotL = s.NdotL;
	const float NdotV = currInteraction.inner.isotropic.NdotV;

	const float LplusV_rcplen = inversesqrt(2.0 + 2.0 * s.VdotL);

	out_microfacet.inner.isotropic.NdotH = (NdotL + NdotV) * LplusV_rcplen;
	out_microfacet.inner.isotropic.NdotH2 = out_microfacet.inner.isotropic.NdotH * out_microfacet.inner.isotropic.NdotH;

	out_microfacet.inner.TdotH = (currInteraction.inner.TdotV + s.TdotL) * LplusV_rcplen;
	out_microfacet.inner.BdotH = (currInteraction.inner.BdotV + s.BdotL) * LplusV_rcplen;

	nbl_glsl_MC_finalizeMicrofacet(out_microfacet);
}

// most parameters can be constant or come from a texture, we have a clever system where a single bitflag tells us
// whether the 64bit value is a RGB19E7 constant or an offset to registers into which a texel was prefetched
vec3 nbl_glsl_MC_textureOrRGBconst(in uvec2 data, in bool texPresenceFlag)
{
	return
#ifdef TEX_PREFETCH_STREAM
	texPresenceFlag ?
		uintBitsToFloat(uvec3(registers[data.x],registers[data.x+1u],registers[data.x+2u])) :
#endif
		nbl_glsl_decodeRGB19E7(data);
}

vec3 nbl_glsl_MC_bsdf_data_getParam1(in nbl_glsl_MC_bsdf_data_t data, in bool texPresence)
{
#ifdef PARAM1_ALWAYS_SAME_VALUE
	return PARAM1_VALUE;
#else
	return nbl_glsl_MC_textureOrRGBconst(data.data[0].xy, texPresence);
#endif
}
vec3 nbl_glsl_MC_bsdf_data_getParam2(in nbl_glsl_MC_bsdf_data_t data, in bool texPresence)
{
#ifdef PARAM2_ALWAYS_SAME_VALUE
	return PARAM2_VALUE;
#else
	return nbl_glsl_MC_textureOrRGBconst(data.data[0].zw, texPresence);
#endif
}

// tells us if a particular parameter is fetched from a register or decoded
bvec2 nbl_glsl_MC_instr_getTexPresence(in nbl_glsl_MC_instr_t i)
{
	return bvec2(
		nbl_glsl_MC_instr_get1stParamTexPresence(i),
		nbl_glsl_MC_instr_get2ndParamTexPresence(i)
	);
}
nbl_glsl_MC_params_t nbl_glsl_MC_instr_getParameters(in nbl_glsl_MC_instr_t i, in nbl_glsl_MC_bsdf_data_t data)
{
	nbl_glsl_MC_params_t p;
	bvec2 presence = nbl_glsl_MC_instr_getTexPresence(i);
	// speculatively always read RGB
	// TODO: maybe with variable length instructions (embedded BSDF params) we could avoid reading more uvec2 than needed 
	p[0] = nbl_glsl_MC_bsdf_data_getParam1(data, presence.x);
	p[1] = nbl_glsl_MC_bsdf_data_getParam2(data, presence.y);

	return p;
}

// IoR is just a 3rd and 4th parameter
// TODO: Open question, is it possible to have just an IoR param wihout the first 2?
mat2x3 nbl_glsl_MC_bsdf_data_decodeIoR(in nbl_glsl_MC_bsdf_data_t data, in uint op)
{
	mat2x3 ior = mat2x3(0.0);
	ior[0] = nbl_glsl_decodeRGB19E7(data.data[1].xy);
#ifdef OP_CONDUCTOR
	ior[1] = (op == OP_CONDUCTOR) ? nbl_glsl_decodeRGB19E7(data.data[1].zw) : vec3(0.0);
#endif

	return ior;
}

// clamping alpha to min MIN_ALPHA because we're using microfacet BxDFs for deltas as well (and NDFs often end up NaN when given alpha=0) because of less deivergence
// TODO: NDFs have been fixed, perform rigorous numerical analysis and kill all sources of NaNs
#define MIN_ALPHA 0.0001
float nbl_glsl_MC_params_getAlpha(in nbl_glsl_MC_params_t p)
{
	return max(p[PARAMS_ALPHA_U_IX].x,MIN_ALPHA);
}
// TODO: reuse as IoR for Cook Torrance
vec3 nbl_glsl_MC_params_getReflectance(in nbl_glsl_MC_params_t p)
{
	return p[PARAMS_REFLECTANCE_IX];
}
float nbl_glsl_MC_params_getAlphaV(in nbl_glsl_MC_params_t p)
{
	return max(p[PARAMS_ALPHA_V_IX].x,MIN_ALPHA);
}
vec3 nbl_glsl_MC_params_getSigmaA(in nbl_glsl_MC_params_t p)
{
	return p[PARAMS_SIGMA_A_IX];
}
vec3 nbl_glsl_MC_params_getBlendWeight(in nbl_glsl_MC_params_t p)
{
	return p[PARAMS_WEIGHT_IX];
}
vec3 nbl_glsl_MC_params_getTransmittance(in nbl_glsl_MC_params_t p)
{
	return p[PARAMS_TRANSMITTANCE_IX];
}

//
nbl_glsl_MC_bxdf_spectrum_t nbl_glsl_MC_coatedDiffuse(
	//in vec3 thickness_sigma, TODO
	in vec3 eta, in vec3 eta2,
	in float clampedNdotV,
	in float clampedNdotL,
	out float pdf
)
{
	//vec3 thickness_sigma = params_getSigmaA(params);

	// TODO include thickness_sigma in diffuse weight computation: exp(sigma_thickness * freePath)
	// freePath = ( sqrt(refract_compute_NdotT2(NdotL2, rcpOrientedEta2)) + sqrt(refract_compute_NdotT2(NdotV2, rcpOrientedEta2)) )
	
	const vec3 transmissionNdotV = vec3(1.0)-nbl_glsl_fresnel_dielectric_frontface_only(eta,clampedNdotV);
	pdf = nbl_glsl_MC_colorToScalar(transmissionNdotV);
	return nbl_glsl_diffuseFresnelCorrectionFactor(eta,eta2)*(vec3(1.0)-nbl_glsl_fresnel_dielectric_frontface_only(eta,clampedNdotL))*transmissionNdotV;
}
// TODO: should it be additionally robustified against NdotV==0, allowing `thickness_sigma` onwards to be garbage ?
nbl_glsl_MC_eval_pdf_aov_t nbl_glsl_MC_instr_execute_COATING(
	in nbl_glsl_MC_eval_pdf_aov_t coat,
	in nbl_glsl_MC_eval_pdf_aov_t coated,
	//vec3 thickness_sigma = params_getSigmaA(params);
	in vec3 eta, in vec3 eta2,
	in float clampedNdotV,
	in float clampedNdotL
)
{
	float diffuse_pdf;
	const vec3 diffuse_weight = nbl_glsl_MC_coatedDiffuse(eta,eta2,clampedNdotV,clampedNdotL,diffuse_pdf);

	nbl_glsl_MC_eval_pdf_aov_t retval;
	retval.value = coat.value+coated.value*diffuse_weight;
#ifdef GEN_CHOICE_STREAM
	retval.pdf = mix(coat.pdf,coated.pdf,diffuse_pdf);
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	retval.aov.albedo = coat.aov.albedo+coated.aov.albedo*diffuse_weight;
	retval.aov.throughputFactor = 0.f;
	retval.aov.normal = coat.aov.normal+coated.aov.normal*diffuse_pdf;
#endif
#endif

	return retval;
}

void nbl_glsl_MC_instr_execute_BUMPMAP(in uint srcReg, in nbl_glsl_MC_precomputed_t precomp)
{
	vec3 N;
	nbl_glsl_MC_readReg(srcReg, N);
	nbl_glsl_MC_updateCurrInteraction(precomp, N);
}

void nbl_glsl_MC_instr_execute_SET_GEOM_NORMAL(in nbl_glsl_MC_precomputed_t precomp)
{
	nbl_glsl_MC_setCurrInteraction(precomp);
}

// TODO: should it be additionally robustified against NdotV==0, allowing `params` to be garbage ?
nbl_glsl_MC_bxdf_spectrum_t nbl_glsl_MC_instr_execute_BLEND(
	in nbl_glsl_MC_bxdf_spectrum_t srcA,
	in nbl_glsl_MC_bxdf_spectrum_t srcB,
	in nbl_glsl_MC_params_t params,
	out vec3 blend_weight
)
{
	blend_weight = nbl_glsl_MC_params_getBlendWeight(params);
	return mix(srcA,srcB,blend_weight);
}
nbl_glsl_MC_eval_pdf_aov_t nbl_glsl_MC_instr_execute_BLEND(
	in nbl_glsl_MC_eval_pdf_aov_t srcA,
	in nbl_glsl_MC_eval_pdf_aov_t srcB,
	in nbl_glsl_MC_params_t params
)
{
	nbl_glsl_MC_eval_pdf_aov_t retval;

	vec3 blend_weight;
	// Instead of doing this here, remainder and pdf returned from generator stream is weighted so that it correctly adds up at the end
	/*
	// generator is denoted as the one with negative probability
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
	retval.value = nbl_glsl_MC_instr_execute_BLEND(srcA.value,srcB.value,params,blend_weight);
	const float w_pdf = nbl_glsl_MC_colorToScalar(blend_weight);

#ifdef GEN_CHOICE_STREAM
	retval.pdf = mix(srcA.pdf,srcB.pdf,w_pdf);
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	retval.aov.albedo = mix(srcA.aov.albedo,srcB.aov.albedo,blend_weight);
	// lazy approach to eyeballing throughput, could theoretically come up with something better/accurate
	retval.aov.throughputFactor = mix(srcA.aov.throughputFactor,srcB.aov.throughputFactor,w_pdf);
	retval.aov.normal = mix(srcA.aov.normal,srcB.aov.normal,w_pdf);
#endif
#endif

	return retval;
}


vec3 nbl_glsl_MC_fetchTex(in uvec3 texid, in vec2 uv, in mat2 dUV)
{
	float scale = uintBitsToFloat(texid.z);

#if _NBL_VT_FLOAT_VIEWS_COUNT
	return nbl_glsl_vTextureGrad(texid.xy, uv, dUV).rgb*scale;
#else
	return vec3(0.0);
#endif
}

#ifdef TEX_PREFETCH_STREAM
void nbl_glsl_MC_runTexPrefetchStream(in nbl_glsl_MC_instr_stream_t stream, in vec2 uv, in mat2 dUV)
{
	for (uint i = 0u; i < stream.count; ++i)
	{
		nbl_glsl_MC_prefetch_instr_t instr = nbl_glsl_MC_fetchPrefetchInstr(stream.offset+i);

		// TODO: do we really need 128bits, couldn't we do something in 96 (pack scale as a half float and registers in the remaining 16 bits)
		uint regcnt = nbl_glsl_MC_prefetch_instr_getRegCount(instr);
		uint reg = nbl_glsl_MC_prefetch_instr_getDstReg(instr);

		vec3 val = nbl_glsl_MC_fetchTex(instr.xyz, uv, dUV);

		nbl_glsl_MC_writeReg(reg, val.x);
#if defined(PREFETCH_REG_COUNT_2) || defined(PREFETCH_REG_COUNT_3)
		if (regcnt>=2u)
			nbl_glsl_MC_writeReg(reg+1u, val.y);
#endif
#if defined(PREFETCH_REG_COUNT_3)
		if (regcnt==3u)
			nbl_glsl_MC_writeReg(reg+2u, val.z);
#endif
	}
}

#ifdef NORM_PRECOMP_STREAM
void nbl_glsl_MC_runNormalPrecompStream(in nbl_glsl_MC_instr_stream_t stream, in nbl_glsl_MC_precomputed_t precomp)
{
	nbl_glsl_MC_setCurrInteraction(precomp);
	for (uint i = 0u; i < stream.count; ++i)
	{
		nbl_glsl_MC_instr_t instr = nbl_glsl_MC_fetchInstr(stream.offset+i);

		// TODO: shouldn't need to read BSDF data for this instruction, SRC/DST reg should be stuffed inside the instruction itself!
		nbl_glsl_MC_bsdf_data_t bsdf_data = nbl_glsl_MC_fetchBSDFDataForInstr(instr);
		const uint srcreg = bsdf_data.data[0].x;
		const uint dstreg = nbl_glsl_MC_instr_decodeRegisters(instr).dst;

		vec2 dh;
		nbl_glsl_MC_readReg(srcreg,dh);
		nbl_glsl_MC_writeReg(dstreg,nbl_glsl_perturbNormal_derivativeMap(currInteraction.inner.isotropic.N, dh));
	}
}
#endif
#endif

//
void nbl_glsl_MC_instr_eval_and_pdf_execute(in nbl_glsl_MC_instr_t instr, in nbl_glsl_MC_precomputed_t precomp, in nbl_glsl_LightSample s, in nbl_glsl_MC_microfacet_t _microfacet, in bool skip)
{
	const uint op = nbl_glsl_MC_instr_getOpcode(instr);
	const nbl_glsl_MC_RegID_t regs = nbl_glsl_MC_instr_decodeRegisters(instr);

	const bool is_bxdf = nbl_glsl_MC_op_isBXDF(op);
	const bool is_not_brdf = !nbl_glsl_MC_op_isBRDF(op);
	const bool is_bxdf_or_combiner = nbl_glsl_MC_op_isBXDForCoatOrBlend(op);

	nbl_glsl_MC_params_t params;
	nbl_glsl_MC_bsdf_data_t bsdf_data;
	mat2x3 ior;
	mat2x3 ior2;
	nbl_glsl_MC_microfacet_t microfacet;

	// TODO: should this include the NdotV term?
	const bool run = !skip;

	if (is_bxdf_or_combiner && run)
	{
		bsdf_data = nbl_glsl_MC_fetchBSDFDataForInstr(instr);
		ior = nbl_glsl_MC_bsdf_data_decodeIoR(bsdf_data, op);
		ior2 = matrixCompMult(ior, ior);
		params = nbl_glsl_MC_instr_getParameters(instr, bsdf_data);
	}

	const float NdotV = nbl_glsl_conditionalAbsOrMax(is_not_brdf, currInteraction.inner.isotropic.NdotV, 0.0);
	const float NdotL = nbl_glsl_conditionalAbsOrMax(is_not_brdf, s.NdotL, 0.0);

	nbl_glsl_MC_eval_pdf_aov_t result;
	result.value = vec3(0.f);
	result.pdf = 0.0;
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	result.aov.albedo = vec3(0.f);
	result.aov.throughputFactor = 0.f;
	result.aov.normal = vec3(0.f);
#endif
	if (is_bxdf && run && (NdotV > nbl_glsl_FLT_MIN))
	{
		//speculative execution
		uint ndf = nbl_glsl_MC_instr_getNDF(instr);
		float a = nbl_glsl_MC_params_getAlpha(params);
		float a2 = a*a;
#ifdef ALL_ISOTROPIC_BXDFS
		float one_minus_a2 = 1.0 - a2;
#else
		float ay = nbl_glsl_MC_params_getAlphaV(params);
		float ay2 = ay*ay;
#endif
		const vec3 albedo = nbl_glsl_MC_params_getReflectance(params);

#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
		result.aov.normal = precomp.N;
#endif

#if defined(OP_DIFFUSE) || defined(OP_DIFFTRANS)
		if (nbl_glsl_MC_op_isDiffuse(op))
		{
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
			result.aov.albedo = albedo;
#endif
			if (NdotL > nbl_glsl_FLT_MIN)
			{
				result.value = albedo * nbl_glsl_oren_nayar_cos_remainder_and_pdf_wo_clamps(result.pdf, a2, s.VdotL, NdotL, NdotV);
				result.pdf *= is_not_brdf ? 0.5 : 1.0;
				result.value *= result.pdf;
			}
		}
		else
#endif
#if defined(OP_CONDUCTOR) || defined(OP_DIELECTRIC)
		{
			const float eta = nbl_glsl_MC_colorToScalar(ior[0]);
			const float rcp_eta = 1.0 / eta;

			bool is_valid = true;
			bool refraction = false;
			if (nbl_glsl_isTransmissionPath(currInteraction.inner.isotropic.NdotV, s.NdotL))
			{
				nbl_glsl_calcAnisotropicMicrofacetCache(microfacet.inner, true, currInteraction.inner.isotropic.V.dir, s.L, currInteraction.inner.T, currInteraction.inner.B, currInteraction.inner.isotropic.N, s.NdotL, s.VdotL, eta, rcp_eta);
				nbl_glsl_MC_finalizeMicrofacet(microfacet);
				refraction = true;
			}
			else
				microfacet = _microfacet;

			// TODO: remove the alpha check, implementation should be numerically stable enough to handle roughness tending to 0, also it doesnt do anything for anisotropic roughnesses right now!
			if (nbl_glsl_isValidVNDFMicrofacet(microfacet.inner.isotropic, is_not_brdf, refraction, s.VdotL, eta, rcp_eta) && a > NBL_GLSL_MC_ALPHA_EPSILON)
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

#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
#ifdef ALL_ISOTROPIC_BXDFS
					result.aov.throughputFactor = nbl_glsl_MC_aov_t_specularThroughputFactor(a2);
#else
					result.aov.throughputFactor = nbl_glsl_MC_aov_t_specularThroughputFactor(a2, ay2);
#endif
#endif

				result.pdf = nbl_glsl_smith_VNDF_pdf_wo_clamps(ndf_val, G1_over_2NdotV);
				float remainder_scalar_part = G2_over_G1;

				const float VdotH = abs(microfacet.inner.isotropic.VdotH);

				// compute fresnel for the microfacet
				vec3 fr;
				// computing it again for albedo is unfortunately quite expensive, but I have no other choice
#ifdef OP_CONDUCTOR
#ifdef OP_DIELECTRIC
				if (op == OP_CONDUCTOR)
#endif
				{
					fr = nbl_glsl_fresnel_conductor(ior[0],ior[1],VdotH);
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
					result.aov.albedo = nbl_glsl_fresnel_conductor(ior[0],ior[1],NdotV);
#endif
				}
#endif
#ifdef OP_DIELECTRIC
#ifdef OP_CONDUCTOR
				else
#endif
				{
					const float eta2 = eta*eta;
					// TODO: would be nice not to have monochrome dielectrics
					fr = vec3(nbl_glsl_fresnel_dielectric_common(eta2,VdotH));
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
					result.aov.albedo = vec3(1.0);
#endif
				}
#endif

				float eval_scalar_part = remainder_scalar_part * result.pdf;
#ifndef NO_BSDF
				if (is_not_brdf)
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

					float reflectance = nbl_glsl_MC_colorToScalar(fr);
					reflectance = refraction ? (1.0 - reflectance) : reflectance;
					result.pdf *= reflectance;
				}
#endif 
				result.value = fr * eval_scalar_part;
			} 
		}
#endif
		{} // empty else for when there are diffuse ops but arent any specular ones


#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
		const float aovContrib = 1.f-result.aov.throughputFactor;
		result.aov.albedo *= aovContrib;
		result.aov.normal *= aovContrib;
#endif
	}

	if (!is_bxdf)
	{
		nbl_glsl_MC_eval_pdf_aov_t srcA,srcB;
		nbl_glsl_MC_readReg(regs.srcA, srcA);
		nbl_glsl_MC_readReg(regs.srcB, srcB);
		BEGIN_CASES(op)
#ifdef OP_COATING
		CASE_BEGIN(op, OP_COATING) {
			//vec3 thickness_sigma = params_getSigmaA(params);
			result = nbl_glsl_MC_instr_execute_COATING(srcA, srcB, /*thickness_sigma,*/ ior[0], ior2[0], NdotV, NdotL);
		} CASE_END
#endif
#ifdef OP_BLEND
		CASE_BEGIN(op, OP_BLEND) {
			result = nbl_glsl_MC_instr_execute_BLEND(srcA, srcB, params);
		} CASE_END
#endif
#ifdef OP_BUMPMAP
		CASE_BEGIN(op, OP_BUMPMAP) {
			nbl_glsl_MC_instr_execute_BUMPMAP(regs.srcA, precomp);
		} CASE_END
#endif
#ifdef OP_SET_GEOM_NORMAL
		CASE_BEGIN(op, OP_SET_GEOM_NORMAL) {
			nbl_glsl_MC_instr_execute_SET_GEOM_NORMAL(precomp);
		} CASE_END
#endif
		CASE_OTHERWISE
		{} //else "empty braces"
		END_CASES
	}

	if (is_bxdf_or_combiner)
		nbl_glsl_MC_writeReg(regs.dst, result);
}

#ifdef GEN_CHOICE_STREAM
// function is geared toward being used in tandem with importance sampling, hence the `generator_offset` parameter
// the idea is to save computation and gain numerical stability by factoring out the generating BxDF from the quotient of sums
// this necessitates an evaluation of the sum of Value & PDF over all BxDF leafs which are not the generator
// Note: If you dont need/want this "generator skipping" behaviour, just set `generator_offset>=stream.count` (`~0u` will do as well) 
nbl_glsl_MC_eval_pdf_aov_t nbl_bsdf_eval_and_pdf(
	in nbl_glsl_MC_precomputed_t precomp,
	in nbl_glsl_MC_instr_stream_t stream,
	in uint generator_offset,
	inout nbl_glsl_LightSample s,
	inout nbl_glsl_MC_microfacet_t microfacet
)
{
	// expand thin intersection struct into full precomputation of all tangent frame angles for BxDF evaluation
	nbl_glsl_MC_setCurrInteraction(precomp);
	for (uint i=0u; i<stream.count; ++i)
	{
		const nbl_glsl_MC_instr_t instr = nbl_glsl_MC_fetchInstr(stream.offset+i);
		const uint op = nbl_glsl_MC_instr_getOpcode(instr);

		bool skip = (i == generator_offset);
		// skip deltas because they cant contribute anything if they're not the generators
		// NOTE: Material Compiler will fail if doing a BxDF blend of identical delta BxDFs (same peaks)
#ifdef OP_THINDIELECTRIC
		skip = skip || (op == OP_THINDIELECTRIC);
#endif
#ifdef OP_DELTATRANS
		skip = skip || (op == OP_DELTATRANS);
#endif

		nbl_glsl_MC_instr_eval_and_pdf_execute(instr, precomp, s, microfacet, skip);

		// microfacet cache and light sample precomputed angles need some love after the normal gets hot-swapped
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
		)
		{
			nbl_glsl_MC_updateLightSampleAfterNormalChange(s);
			nbl_glsl_MC_updateMicrofacetCacheAfterNormalChange(s, microfacet);
		}
#endif
	}

	nbl_glsl_MC_eval_pdf_aov_t retval;
	nbl_glsl_MC_readReg(0u,retval);
	return retval;
}

struct nbl_glsl_MC_quot_pdf_aov_t
{
	// TODO: Rename all `rem`/`remainder` in Nabla GLSL to `quot`/`quotient`, bad taxonomy
	nbl_glsl_MC_bxdf_spectrum_t quotient;
	float pdf;
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	nbl_glsl_MC_aov_t aov;
#endif
};
// we're able to "perfectly" importance sample sums of BRDFs, thanks to noting that:
// - our BxDF tree is transformed into a binary tree during IR generation
// - choosing left vs right branch with probability proportional to the weight keeps the ratio of evaluated value to pdf of the final leaf BxDF constant
// - this means that the overall pdf (from all generators) is a sum of individual Importance Sampling PDFs of BxDFs with the same LUMA weights as the BxDF itself
// - rgb weighting and fresnel complicates this, but this only makes the weights slightly different, their ratio is still in the valid (0,INF) range
nbl_glsl_LightSample nbl_bsdf_cos_generate(
	in nbl_glsl_MC_precomputed_t precomp,
	in nbl_glsl_MC_instr_stream_t stream,
	in vec3 rand,
	out nbl_glsl_MC_quot_pdf_aov_t out_values,
	out nbl_glsl_MC_microfacet_t out_microfacet,
	out uint out_gen_rnpOffset
)
{
	// position in the flattenned tree (i.e. the stream)
	uint ix = 0u;
	nbl_glsl_MC_instr_t instr = nbl_glsl_MC_fetchInstr(stream.offset);
	// get root op
	uint op = nbl_glsl_MC_instr_getOpcode(instr);
	vec3 u = rand;

	// precompute some stuff from a lightweight intersectionstruct
	nbl_glsl_MC_setCurrInteraction(precomp);

#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	// set up the default as full throughput, which will set normal and albedo to 0 despite being uninitialized or initialized speculatively
	out_values.aov.throughputFactor = 1.f;
#endif

	vec3 branchWeight = vec3(1.0);
	// PDFs will be multiplied in (as choices are independent), at the end of the while loop it will be the PDF of choosing a particular BxDF leaf
	float rcpBranchPdf = 1.0;
	// keep track if we chose the diffuse coatee
	bool no_coat_parent = true;
	// stochastic descent
	while (!nbl_glsl_MC_op_isBXDF(op))
	{
		#if defined(OP_COATING) || defined(OP_BLEND)
		if (nbl_glsl_MC_op_isBXDForCoatOrBlend(op))
		{
			#if defined(OP_BLEND)
			const bool isBlend = op==OP_BLEND;
			#else
			const bool isBlend = false;
			#endif
			// two node children, one must be picked according to some PDF
			vec3 blendWeight_OR_fresnelTransmission;
			{
				const nbl_glsl_MC_bsdf_data_t bsdf_data = nbl_glsl_MC_fetchBSDFDataForInstr(instr);
				if (isBlend)
				{
					nbl_glsl_MC_params_t params = nbl_glsl_MC_instr_getParameters(instr,bsdf_data);
					blendWeight_OR_fresnelTransmission = nbl_glsl_MC_params_getBlendWeight(params);
				}
				else
				{
					const vec3 eta = nbl_glsl_MC_bsdf_data_decodeIoR(bsdf_data,OP_COATING)[0];
					// fresnel gets tricky, we kind-of assume fresnel against the surface macro-normal is somewhat proportional to integrated fresnel over the distribution of visible normals
					blendWeight_OR_fresnelTransmission = vec3(1.f)-nbl_glsl_fresnel_dielectric_frontface_only(eta,max(currInteraction.inner.isotropic.NdotV,0.f));

					no_coat_parent = false;
				}
			}
			// choice is binary, want PDF proportional to the blend weight which is an RGB triplet, so convert it to a scalar
			const float rightChildProb = nbl_glsl_MC_colorToScalar(blendWeight_OR_fresnelTransmission);
			// this will be 1/pdf of the choice made
			float rcpChoiceProb;
			const bool choseLeft = nbl_glsl_partitionRandVariable(rightChildProb,u.z,rcpChoiceProb);
			// keep track of the total PDF of choosing the branch so far
			rcpBranchPdf *= rcpChoiceProb;
			if (choseLeft)
			{
				// TODO: if we turn the instruction set variable, we'll need both a left jump as well.
				// Good news is that both jumps are short jumps (very few dwords) and left (being closer) will be very short.
				ix++;
				// if the op was a coating and we chose the left child, we're dealing with the specular BRDF which has no extra branch weight
				if (isBlend)
					blendWeight_OR_fresnelTransmission = vec3(1.f)-blendWeight_OR_fresnelTransmission;
			}
			else// right child BxDF node is **coated** material in the case of a coating
			{
				ix = nbl_glsl_MC_instr_getRightJump(instr);
			}
			// keep track of the total Weight chosen the branch so far
			// NOTE: The coat doesn't have an extra weight, and we don't weigh the diffuse coatee because we'll get its value from the regular eval stream
			if (isBlend)
				branchWeight *= blendWeight_OR_fresnelTransmission;
		}
		else
		#endif //OP_COATING or OP_BLEND
		{
			// these only modify the shading normal which will be used to importance sample the leaf BxDF
			#ifdef OP_SET_GEOM_NORMAL
			if (op==OP_SET_GEOM_NORMAL)
			{
				nbl_glsl_MC_instr_execute_SET_GEOM_NORMAL(instr,precomp);
			} else 
			#endif //OP_SET_GEOM_NORMAL
			#ifdef OP_BUMPMAP
			if (op==OP_BUMPMAP)
			{
				nbl_glsl_MC_instr_execute_BUMPMAP(nbl_glsl_MC_instr_decodeRegisters(instr).srcA,precomp);
			} else
			#endif //OP_BUMPMAP
			{}
			// left child is always after its parent (useful for single child nodes like bump modifiers)
			ix++;
		}
		// TODO: compute jumps as relative, then we dont have to do this addition
		instr = nbl_glsl_MC_fetchInstr(stream.offset+ix);
		op = nbl_glsl_MC_instr_getOpcode(instr);
	}
	// Its important to keep track of the chosen BxDF leaf for importance sampling generation, to skip its contribution the weighted sum.
	// We leverage the fact that the instruction stream is the same for both Quotient&PDF and Evaluation functions.
	out_gen_rnpOffset = nbl_glsl_MC_instr_getOffsetIntoRnPStream(instr);

	// if PDF is 0, none of the other values will be used for any arithmetic at all
	out_values.pdf = 0.f;

	nbl_glsl_LightSample s;
	#ifdef OP_DELTATRANS
	if (op == OP_DELTATRANS)
	{
		s = nbl_glsl_createLightSample(-precomp.V, -1.0, currInteraction.inner.T, currInteraction.inner.B, currInteraction.inner.isotropic.N);
		// not computing microfacet cache since it's always transmission and it will be recomputed anyway
		out_values.quotient = vec3(1.0);
		out_values.pdf = nbl_glsl_FLT_INF;
		// do nothing to AoVs because the default is full throughput
	}
	else
	#endif
	{
		// preload common data
		const nbl_glsl_MC_bsdf_data_t bsdf_data = nbl_glsl_MC_fetchBSDFDataForInstr(instr);
		const mat2x3 ior = nbl_glsl_MC_bsdf_data_decodeIoR(bsdf_data,op);

		// precompute common parameters
		const bool is_bsdf = !nbl_glsl_MC_op_isBRDF(op);
		const float NdotV = nbl_glsl_conditionalAbsOrMax(is_bsdf,currInteraction.inner.isotropic.NdotV,0.f);
		const mat2x3 ior2 = matrixCompMult(ior,ior);

		//
		#ifdef OP_THINDIELECTRIC
		if (op == OP_THINDIELECTRIC)
		{
			const vec3 luminosityContributionHint = NBL_GLSL_MC_CIE_XYZ_Luma_Y_coeffs;
			vec3 remMetadata;
			s = nbl_glsl_thin_smooth_dielectric_cos_generate_wo_clamps(
				currInteraction.inner.isotropic.V.dir, currInteraction.inner.T, currInteraction.inner.B, currInteraction.inner.isotropic.N, 
				currInteraction.inner.isotropic.NdotV, NdotV, u, ior2[0], luminosityContributionHint, remMetadata
			);
			out_values.quotient = nbl_glsl_thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(out_values.pdf, remMetadata);
			// do nothing to AoVs because the default is full throughput

			// TODO: factor it out!
			out_microfacet.inner = nbl_glsl_calcAnisotropicMicrofacetCache(currInteraction.inner,s);
			nbl_glsl_MC_finalizeMicrofacet(out_microfacet);
		} else
		#endif
		{
			if (NdotV>nbl_glsl_FLT_MIN)
			{
				const nbl_glsl_MC_params_t params = nbl_glsl_MC_instr_getParameters(instr, bsdf_data);

				const float ax = nbl_glsl_MC_params_getAlpha(params);
				const float ax2 = ax * ax;
				// TODO: specialize for all isotropic NDFs
				const float ay = nbl_glsl_MC_params_getAlphaV(params);
				const float ay2 = ay * ay;
				#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
				out_values.aov.normal = precomp.N;
				#endif

				// TODO: refactor
				const vec3 localV = nbl_glsl_getTangentSpaceV(currInteraction.inner);
				const mat3 tangentFrame = nbl_glsl_getTangentFrame(currInteraction.inner);
				#if defined(OP_DIFFUSE) || defined(OP_DIFFTRANS)
				if (nbl_glsl_MC_op_isDiffuse(op))
				{
					vec3 localL = nbl_glsl_projected_hemisphere_generate(u.xy);
					#ifndef NO_BSDF
					if (is_bsdf)
					{
						float dummy; // we dont bother using this value because its constant
						bool flip = nbl_glsl_partitionRandVariable(0.5,u.z,dummy);
						localL = flip ? -localL : localL;
						out_values.pdf = 0.5;
					}
					else
					#endif
						out_values.pdf = 1.f;

					s = nbl_glsl_createLightSampleTangentSpace(localV,localL,tangentFrame);

					// TODO: factor it out!
					out_microfacet.inner = nbl_glsl_calcAnisotropicMicrofacetCache(currInteraction.inner,s);
					nbl_glsl_MC_finalizeMicrofacet(out_microfacet);

					// NOTE: If the chosen generator is coated diffuse, we let the evaluation function compute the full weighted sum.
					// This is because the diffuse coatee needs a special fresnel factor which can only be computed while `L` is known.
					// We can allow for this because the coating must have a diffuse coatee AND a diffuse BRDF has a pretty wide PDF,
					// so numerical stability is not affected.
					if (no_coat_parent)
					{
						const vec3 albedo = nbl_glsl_MC_params_getReflectance(params);

						float pdf;
						const float NdotL = nbl_glsl_conditionalAbsOrMax(is_bsdf,s.NdotL,0.f);
						out_values.quotient = albedo*nbl_glsl_oren_nayar_cos_remainder_and_pdf_wo_clamps(pdf,ax2,s.VdotL,NdotL,NdotV);
						out_values.pdf *= pdf;
						#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
						out_values.aov.albedo = albedo;
						#endif
					}
					else
					{
						out_gen_rnpOffset = 0xffffffffu;
						out_values.pdf = 0.f;
					}
					#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
					out_values.aov.throughputFactor = 0.f;
					#endif
				}
				else
				#endif
				#if defined(OP_CONDUCTOR) || defined(OP_DIELECTRIC)
				if (nbl_glsl_MC_op_hasSpecular(op))
				{
					const uint ndf = nbl_glsl_MC_instr_getNDF(instr);
					const vec3 upperHemisphereLocalV = currInteraction.inner.isotropic.NdotV<0.f ? -localV:localV;

					vec3 localH = vec3(0.0);
					BEGIN_CASES(ndf)
		#ifdef NDF_GGX
					CASE_BEGIN(ndf, NDF_GGX) 
					{
						// NOTE: why is it called without "wo_clamps" and beckmann sampling is?
						// Cause GGX sampling needs no clamping of `localV` to upper hemisphere to not generate garbage
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

					const float eta = nbl_glsl_MC_colorToScalar(ior[0]);
					const float VdotH = dot(localH,localV);
			
					bool refraction = false;
					out_values.pdf = 1.f;
					{
						const float rcpEta = 1.0/eta; // TODO: move computation into the reflective case?

						#ifdef OP_CONDUCTOR
						const bool isConductor = op==OP_CONDUCTOR;
						if (isConductor)
						{
							const float VdotH_clamp = max(VdotH,0.f);
							out_values.quotient = nbl_glsl_fresnel_conductor(ior[0], ior[1], VdotH_clamp);
							#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
							// computing fresnel again for albedo is unfortunately quite expensive, but I have no other choice
							out_values.aov.albedo = nbl_glsl_fresnel_conductor(ior[0],ior[1],NdotV);
							if (!no_coat_parent)
								out_values.aov.normal *= nbl_glsl_MC_colorToScalar(out_values.aov.albedo);
							#endif
						}
						else
						#endif
						{
							// TODO: it would be nice to make dielectrics not monochrome somehow
							out_values.quotient = vec3(1.0);
					
							const float eta2 = eta*eta;

							float rcpChoiceProb;
							{
								const float refractionProb = nbl_glsl_fresnel_dielectric_common(eta2,VdotH);
								refraction = nbl_glsl_partitionRandVariable(refractionProb,u.z,rcpChoiceProb);
							}
							out_values.pdf /= rcpChoiceProb;
							
							#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
							out_values.aov.albedo = vec3(1.f);
							#endif
						}
						// scope localL out
						// TODO: factor the microfacet update stuff out!
						{
							vec3 localL;
							out_microfacet.inner = nbl_glsl_calcAnisotropicMicrofacetCache(refraction, localV, localH, localL, rcpEta, rcpEta*rcpEta);
							s = nbl_glsl_createLightSampleTangentSpace(localV, localL, tangentFrame);
						}
					}

					nbl_glsl_MC_finalizeMicrofacet(out_microfacet);
					const float TdotH2 = out_microfacet.TdotH2;
					const float BdotH2 = out_microfacet.BdotH2;
					const float NdotH2 = out_microfacet.inner.isotropic.NdotH2;
					const float NdotL = nbl_glsl_conditionalAbsOrMax(is_bsdf, s.NdotL, 0.0);

					const float TdotV2 = currInteraction.TdotV2;
					const float BdotV2 = currInteraction.BdotV2;
					const float NdotV2 = currInteraction.inner.isotropic.NdotV_squared;

					float G2_over_G1 = 0.0;
					float G1_over_2NdotV = 0.0;
					float ndf_val = 0.0;
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

					#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
					if (no_coat_parent)
					{
						#ifdef ALL_ISOTROPIC_BXDFS
						out_values.aov.throughputFactor = nbl_glsl_MC_aov_t_specularThroughputFactor(ax2);
						#else
						out_values.aov.throughputFactor = nbl_glsl_MC_aov_t_specularThroughputFactor(ax2,ay2);
						#endif
					}
					else
						out_values.aov.throughputFactor = 0.f;
					#endif

					const float LdotH = out_microfacet.inner.isotropic.LdotH;
					const float VdotHLdotH = VdotH * LdotH;
					out_values.quotient *= G2_over_G1;
					// note: at this point the pdf is already multiplied by transmission/reflection choice probability if applicable
					out_values.pdf *= nbl_glsl_smith_FVNDF_pdf_wo_clamps(ndf_val, G1_over_2NdotV, NdotV, refraction, VdotH, LdotH, VdotHLdotH, eta);
				} else
				#endif
				{} //empty braces for `else`
			}
		}
	}

	out_values.quotient *= branchWeight*rcpBranchPdf;
	out_values.pdf /= rcpBranchPdf;
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	// now also weight the branch by whatever was not the AoV throughput
	branchWeight *= 1.f-out_values.aov.throughputFactor;
	out_values.aov.albedo *= branchWeight;
	out_values.aov.normal *= nbl_glsl_MC_colorToScalar(branchWeight);
#endif

	return s;
}

nbl_glsl_MC_quot_pdf_aov_t nbl_glsl_MC_runGenerateAndRemainderStream(
	in nbl_glsl_MC_precomputed_t precomp,
	in nbl_glsl_MC_instr_stream_t gcs,
	in nbl_glsl_MC_instr_stream_t rnps,
	in vec3 rand,
	out nbl_glsl_LightSample out_smpl
)
{
	nbl_glsl_MC_quot_pdf_aov_t generator_qpa;
	nbl_glsl_MC_microfacet_t microfacet;
	uint generator_rnpOffset;
	out_smpl = nbl_bsdf_cos_generate(precomp, gcs, rand, generator_qpa, microfacet, generator_rnpOffset);

	// we need regular evaluation of the rest, without quotients, we'll divide later
	const nbl_glsl_MC_eval_pdf_aov_t rest_epa = nbl_bsdf_eval_and_pdf(precomp, rnps, generator_rnpOffset, out_smpl, microfacet);

	nbl_glsl_MC_quot_pdf_aov_t retval;
	retval.quotient = rest_epa.value;
	retval.pdf = rest_epa.pdf;
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	retval.aov = rest_epa.aov;
#endif
	{
		float den = rest_epa.pdf;
		// Conditional is needed because sometimes generated microfacet is geometrically impossible to reach (Bump Mapping or Total Internal Reflection)
		// We use a PDF of 0 to denote that (because generators are guaranteed to not produce samples with an actual value of 0).
		if (generator_qpa.pdf>nbl_glsl_FLT_MIN)
		{
			retval.pdf += generator_qpa.pdf;
			
			// guaranteed less than INF, because now we know its a valid sample
			const float rcp_generator_pdf = 1.0/generator_qpa.pdf;
			// this seems like a really roundabout way of doing things but its numerically stable
			// Instead of (gen_v+rest_v)/(gen_p+rest_p)
			// Have (quot_v+rest_v/gen_p)/(1.0+rest_p/gen_p)
			// which is resilient to NaNs when mixing smooth BxDFs
			retval.quotient *= rcp_generator_pdf;
			retval.quotient += generator_qpa.quotient;

#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
			retval.aov.albedo += generator_qpa.aov.albedo;
			// NOTE: real throughput factor would have been
			// `(generator_qpa.quotient*generator_qpa.aov.throughput+rest_epa.value*rest_epa.aov.throughput*rcp_generator_pdf)/retval.quotient`
			retval.aov.throughputFactor += generator_qpa.aov.throughputFactor;
			retval.aov.normal += generator_qpa.aov.normal;
#endif

			den = den*rcp_generator_pdf+1.0;
		}
		// However just because the generator's sample is invalid for its own shading model, doesn't mean that its not valid for others
		if (retval.pdf>nbl_glsl_FLT_MIN)
			retval.quotient /= den;
	}
	return retval;
}
#endif //GEN_CHOICE_STREAM

#endif