#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_INCLUDED_

#include <irr/builtin/material_compiler/glsl/common_declarations.glsl>

#ifndef _IRR_USER_PROVIDED_MATERIAL_COMPILER_GLSL_BACKEND_FUNCTIONS_
	#error "You need to define 'vec3 irr_glsl_MC_getCamPos()', 'instr_t irr_glsl_MC_fetchInstr(in uint)', 'bsdf_data_t irr_glsl_MC_fetchBSDFData(in uint)' functions above"
#endif

#include <irr/builtin/glsl/math/functions.glsl>
#include <irr/builtin/glsl/format/decode.glsl>

uint instr_getNormalId(in instr_t instr)
{
	return (instr.y>>(INSTR_NORMAL_ID_SHIFT-32u)) & INSTR_NORMAL_ID_MASK;
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
	return (instr.x&(1u<<INSTR_1ST_PARAM_TEX_SHIFT)) != 0u;
}
bool instr_get2ndParamTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_2ND_PARAM_TEX_SHIFT)) != 0u;
}
bool instr_get3rdParamTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_3RD_PARAM_TEX_SHIFT)) != 0u;
}
bool instr_get4thParamTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_4TH_PARAM_TEX_SHIFT)) != 0u;
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
#define REG_PREFETCH0(r) (r).x
#define REG_PREFETCH1(r) (r).y
#define REG_PREFETCH2(r) (r).z
#define REG_PREFETCH3(r) (r).w

bsdf_data_t fetchBSDFDataForInstr(in instr_t instr)
{
	uint ix = instr_getBSDFbufOffset(instr);
	return irr_glsl_MC_fetchBSDFData(ix);
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
bool op_isBXDForBlend(in uint op)
{
#ifdef OP_BLEND
	return op<=OP_BLEND;
#else
	return op_isBXDF(op);
#endif
}
bool op_hasSpecular(in uint op)
{
	return op<=OP_MAX_BSDF
#ifdef OP_DIFFUSE
	&& op!=OP_DIFFUSE
#endif
#ifdef OP_DIFFTRANS
	&& op!=OP_DIFFTRANS
#endif
	;
}

#include <irr/builtin/glsl/bxdf/common.glsl>
#include <irr/builtin/glsl/bxdf/fresnel.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/fresnel_correction.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ndf/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/beckmann_smith.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl>
#include <irr/builtin/glsl/bxdf/brdf/cos_weighted_sample.glsl>
#include <irr/builtin/glsl/bump_mapping/utils.glsl>

irr_glsl_BSDFAnisotropicParams currBSDFParams;
irr_glsl_AnisotropicViewSurfaceInteraction currInteraction;
reg_t registers[REG_COUNT];

vec3 textureOrRGBconst(in uvec3 data, in bool texPresenceFlag)
{
	return 
#ifdef TEX_PREFETCH_STREAM
	texPresenceFlag ? 
		uintBitsToFloat(uvec3(registers[data.z],registers[data.z+1u],registers[data.z+2u])) :
#endif
		uintBitsToFloat(data);
}
float textureOrRconst(in uvec3 data, in bool texPresenceFlag)
{
	return 
#ifdef TEX_PREFETCH_STREAM
	texPresenceFlag ?
		uintBitsToFloat(registers[data.z]) :
#endif
		uintBitsToFloat(data.x);
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
	p[0] = textureOrRGBconst(data.data[0].xyz, presence.x);
	p[1] = textureOrRGBconst(uvec3(data.data[0].w,data.data[1].xy), presence.y);
	p[2] = textureOrRGBconst(uvec3(data.data[1].zw,data.data[2].x), presence.z);
	p[3] = textureOrRGBconst(data.data[2].yzw, presence.w);

	return p;
}

//this should thought better
mat2x3 bsdf_data_decodeIoR(in bsdf_data_t data, in uint op)
{
	mat2x3 ior = mat2x3(0.0);
#ifdef OP_CONDUCTOR
	if (op==OP_CONDUCTOR) {
		ior[0] = irr_glsl_decodeRGB19E7(data.data[2].yz);
		ior[1] = irr_glsl_decodeRGB19E7(uvec2(data.data[2].w,data.data[3].x));
	} else
#endif
#ifdef OP_PLASTIC
	if (op==OP_PLASTIC) {
		ior[0] = vec3(uintBitsToFloat(data.data[3].x));
	} else
#endif
#ifdef OP_DIELECTRIC
	if (op==OP_DIELECTRIC) {
		ior[0] = vec3(uintBitsToFloat(data.data[3].x));
	} else
#endif
#ifdef OP_COATING
	if (op==OP_COATING) {
		ior[0] = vec3(unpackHalf2x16(data.data[3].x).y);
	} else
#endif
	{}

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
	writeReg(n   ,v.x);
	writeReg(n+1u,v.y);
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
		readReg1(n), readReg1(n+1u), readReg1(n+2u)
	);
}
vec4 readReg4(in uint n)
{
	return vec4(
		readReg3(n), readReg1(n+3u)
	);
}

void setCurrInteraction(in vec3 N)
{
	vec3 campos = irr_glsl_MC_getCamPos();
	irr_glsl_IsotropicViewSurfaceInteraction interaction = irr_glsl_calcFragmentShaderSurfaceInteraction(campos, WorldPos, N);
	currInteraction = irr_glsl_calcAnisotropicInteraction(interaction);
}
void setCurrBSDFParams(in vec3 N, in vec3 L)
{
	setCurrInteraction(N);
	irr_glsl_BSDFIsotropicParams isoparams = irr_glsl_calcBSDFIsotropicParams(currInteraction.isotropic, L);
	currBSDFParams = irr_glsl_calcBSDFAnisotropicParams(isoparams, currInteraction);
}

float params_getAlpha(in params_t p)
{
	return p[PARAMS_ALPHA_U_IX].x;
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
	return p[PARAMS_ALPHA_V_IX].x;
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


#ifdef OP_DIFFUSE
void instr_execute_cos_eval_DIFFUSE(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		vec3 refl = params_getReflectance(params);
		float a = params_getAlpha(params);
		vec3 diffuse = irr_glsl_oren_nayar_cos_eval(currBSDFParams.isotropic,currInteraction.isotropic,a*a) * refl;
		writeReg(REG_DST(regs), diffuse);
	}
	else writeReg(REG_DST(regs), bxdf_eval_t(0.0));
}
#endif

#ifdef OP_DIFFTRANS
void instr_execute_cos_eval_DIFFTRANS(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	vec3 tr = params_getTransmittance(params);
	//transmittance*cos/2pi
	vec3 c = currBSDFParams.isotropic.NdotL*irr_glsl_RECIPROCAL_PI*0.5*tr;
	writeReg(REG_DST(regs), c);
}
//TODO instr_execute_cos_eval_pdf
#endif

#ifdef OP_DIELECTRIC
void instr_execute_cos_eval_DIELECTRIC(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		//float au = params_getAlpha(params);
		//float av = params_getAlphaV(params);
		vec3 eta = vec3(1.5);
		vec3 diffuse = irr_glsl_lambertian_cos_eval(currBSDFParams.isotropic,currInteraction.isotropic) * vec3(0.89);
		diffuse *= irr_glsl_diffuseFresnelCorrectionFactor(eta,eta*eta) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currInteraction.isotropic.NdotV)) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currBSDFParams.isotropic.NdotL));
		writeReg(REG_DST(regs), diffuse);
	}
	else writeReg(REG_DST(regs), bxdf_eval_t(0.0));
}
#endif

#ifdef OP_THINDIELECTRIC
void instr_execute_cos_eval_THINDIELECTRIC(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		//float au = params_getAlpha(params);
		//float av = params_getAlphaV(params);
		vec3 eta = vec3(1.5);
		vec3 diffuse = irr_glsl_lambertian_cos_eval(currBSDFParams.isotropic,currInteraction.isotropic) * vec3(0.89);
		diffuse *= irr_glsl_diffuseFresnelCorrectionFactor(eta,eta*eta) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currInteraction.isotropic.NdotV)) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currBSDFParams.isotropic.NdotL));
		writeReg(REG_DST(regs), diffuse);
	}
	else writeReg(REG_DST(regs), bxdf_eval_t(0.0));
}
#endif

#ifdef OP_CONDUCTOR
void instr_execute_cos_eval_CONDUCTOR(in instr_t instr, in uvec3 regs, in float DG, in params_t params, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		mat2x3 eta;
		eta[0] = irr_glsl_decodeRGB19E7(data.data[2].yz);
		eta[1] = irr_glsl_decodeRGB19E7(uvec2(data.data[2].w,data.data[3].x));
		vec3 fr = irr_glsl_fresnel_conductor(eta[0],eta[1],currBSDFParams.isotropic.VdotH);
		writeReg(REG_DST(regs), DG*fr);
	}
	else writeReg(REG_DST(regs), bxdf_eval_t(0.0));
}
#endif

#ifdef OP_PLASTIC
void instr_execute_cos_eval_PLASTIC(in instr_t instr, in uvec3 regs, in float DG, in params_t params, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		vec3 eta = vec3(uintBitsToFloat(data.data[3].x));
		vec3 eta2 = eta*eta;
		vec3 refl = params_getReflectance(params);
		float a2 = params_getAlpha(params);
		a2*=a2;

		vec3 diffuse = irr_glsl_oren_nayar_cos_eval(currBSDFParams.isotropic, currInteraction.isotropic, a2) * refl;
		diffuse *= irr_glsl_diffuseFresnelCorrectionFactor(eta,eta2) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currInteraction.isotropic.NdotV)) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currBSDFParams.isotropic.NdotL));
		vec3 fr = vec3(1.0);//irr_glsl_fresnel_dielectric(eta,currBSDFParams.isotropic.VdotH);
		vec3 specular = DG*fr;

		writeReg(REG_DST(regs), specular+diffuse);
	}
	else writeReg(REG_DST(regs), bxdf_eval_t(0.0));
}
float instr_execute_cos_eval_pdf_PLASTIC(in instr_t instr, in uvec3 regs, in float DG, in params_t params, in bsdf_data_t data, in irr_glsl_BSDFSample s, in float specular_pdf)
{
	float a2 = params_getAlpha(params);
	a2 *= a2;
	instr_execute_cos_eval_PLASTIC(instr, regs, DG, params, data);

	//TODO those weights should be different
	//return 0.5*specular_pdf + 0.5*irr_glsl_oren_nayar_pdf(s, currInteraction.isotropic, a2);
	return specular_pdf;
}
#endif

#ifdef OP_COATING
void instr_execute_cos_eval_COATING(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	//vec2 thickness_eta = unpackHalf2x16(data.data[3].x);
	//vec3 sigmaA = params_getSigmaA(params);
	//float a = params_getAlpha(params);
	writeReg(REG_DST(regs), bxdf_eval_t(1.0,0.0,0.0));
}
#endif

#ifdef OP_BUMPMAP
void instr_execute_BUMPMAP_interactionOnly(in instr_t instr)
{
	vec3 N = readReg3( REG_SRC1(instr_decodeRegisters(instr)) );
	setCurrInteraction(N);
}
void instr_execute_BUMPMAP(in instr_t instr, in vec3 L)
{
	vec3 N = readReg3( REG_SRC1(instr_decodeRegisters(instr)) );
	setCurrBSDFParams(N, L);
}
#endif

#ifdef OP_SET_GEOM_NORMAL
//executed at most once
void instr_execute_SET_GEOM_NORMAL_interactionOnly()
{
	setCurrInteraction(normalize(Normal));
}
void instr_execute_SET_GEOM_NORMAL(in vec3 L)
{
	setCurrBSDFParams(normalize(Normal), L);
}
#endif

#ifdef OP_BLEND
void instr_execute_cos_eval_BLEND(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	float w = params_getBlendWeight(params);
	bxdf_eval_t bxdf1 = readReg3(REG_SRC1(regs));
	bxdf_eval_t bxdf2 = readReg3(REG_SRC2(regs));

	bxdf_eval_t blend = mix(bxdf1, bxdf2, w);
	writeReg(REG_DST(regs), blend);
}
void instr_execute_cos_eval_pdf_BLEND(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	float w = params_getBlendWeight(params);
	eval_and_pdf_t bxdf1 = readReg4(REG_SRC1(regs));
	eval_and_pdf_t bxdf2 = readReg4(REG_SRC2(regs));

	float wa = w;
	float wb = 1.0-wa;
	bxdf_eval_t a = bxdf1.rgb;
	float pdfa = bxdf1.a;
	bxdf_eval_t b = bxdf2.rgb;
	float pdfb = bxdf2.a;
	float pdf = mix(pdfa,pdfb,wa);

	bxdf_eval_t blend;
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
	blend = mix(a,b,wa);
	writeReg(REG_DST(regs), blend);

	return mix(pdfa,pdfb,wa);
}
#endif

vec3 fetchTex(in uvec3 texid, in vec2 uv, in mat2 dUV)
{
	float scale = uintBitsToFloat(texid.z);

	return irr_glsl_vTextureGrad(texid.xy, uv, dUV).rgb*scale;
}

uint instr_getTexFetchFlags(in instr_t i)
{
	uint flags = bitfieldExtract(i.x,INSTR_FETCH_FLAG_TEX_0_SHIFT,1);
	flags |= bitfieldExtract(i.x,INSTR_FETCH_FLAG_TEX_1_SHIFT,1)<<1;
	flags |= bitfieldExtract(i.x,INSTR_FETCH_FLAG_TEX_2_SHIFT,1)<<2;
	flags |= bitfieldExtract(i.x,INSTR_FETCH_FLAG_TEX_3_SHIFT,1)<<3;

	return flags;
}

uvec4 instr_decodePrefetchRegs(in instr_t i)
{
	return 
	(i.yyyy >> (uvec4(INSTR_PREFETCH_REG_0_SHIFT,INSTR_PREFETCH_REG_1_SHIFT,INSTR_PREFETCH_REG_2_SHIFT,INSTR_PREFETCH_REG_3_SHIFT)-32u)) & uvec4(INSTR_PREFETCH_REG_MASK);
}

void runTexPrefetchStream(in instr_stream_t stream, in mat2 dUV)
{
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);

		uvec4 regcnt = 
		( instr.yyyy >> (uvec4(INSTR_FETCH_TEX_0_REG_CNT_SHIFT,INSTR_FETCH_TEX_1_REG_CNT_SHIFT,INSTR_FETCH_TEX_2_REG_CNT_SHIFT,INSTR_FETCH_TEX_3_REG_CNT_SHIFT)-uvec4(32)) ) & uvec4(INSTR_FETCH_TEX_REG_CNT_MASK);
		bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);

		uint fetchFlags = instr_getTexFetchFlags(instr);
		uvec4 regs = instr_decodePrefetchRegs(instr);
#ifdef PREFETCH_TEX_0
		if ((fetchFlags & 0x01u) == 0x01u)
		{
			vec3 val = fetchTex(bsdf_data.data[0].xyz, UV, dUV);
			uint reg = REG_PREFETCH0(regs);
#ifdef PREFETCH_REG_COUNT_1
			if (regcnt.x==1u)
				writeReg(reg, val.x);
			else
#endif
#ifdef PREFETCH_REG_COUNT_2
			if (regcnt.x==2u)
				writeReg(reg, val.xy);
			else
#endif
#ifdef PREFETCH_REG_COUNT_3
			if (regcnt.x==3u)
				writeReg(reg, val);
			else
#endif
			{} //else "empty braces"
		}
#endif
#ifdef PREFETCH_TEX_1
		if ((fetchFlags & 0x02u) == 0x02u)
		{
			uvec3 texid = uvec3(bsdf_data.data[0].w,bsdf_data.data[1].xy);
			vec3 val = fetchTex(texid,UV,dUV);
			uint reg = REG_PREFETCH1(regs);
#ifdef PREFETCH_REG_COUNT_1
			if (regcnt.y==1u)
				writeReg(reg, val.x);
			else
#endif
#ifdef PREFETCH_REG_COUNT_2
			if (regcnt.y==2u)
				writeReg(reg, val.xy);
			else
#endif
#ifdef PREFETCH_REG_COUNT_3
			if (regcnt.y==3u)
				writeReg(reg, val);
			else
#endif
			{} //else "empty braces"
		}
#endif
#ifdef PREFETCH_TEX_2
		if ((fetchFlags & 0x04u) == 0x04u)
		{
			uvec3 texid = uvec3(bsdf_data.data[1].zw, bsdf_data.data[2].x);
			vec3 val = fetchTex(texid,UV,dUV);
			uint reg = REG_PREFETCH2(regs);
#ifdef PREFETCH_REG_COUNT_1
			if (regcnt.z==1u)
				writeReg(reg, val.x);
			else
#endif
#ifdef PREFETCH_REG_COUNT_2
			if (regcnt.z==2u)
				writeReg(reg, val.xy);
			else
#endif
#ifdef PREFETCH_REG_COUNT_3
			if (regcnt.z==3u)
				writeReg(reg, val);
			else
#endif
			{} //else "empty braces"
		}
#endif
#ifdef PREFETCH_TEX_3
		if ((fetchFlags & 0x08u) == 0x08u)
		{
			vec3 val = fetchTex(bsdf_data.data[2].yzw, UV, dUV);
			uint reg = REG_PREFETCH3(regs);
#ifdef PREFETCH_REG_COUNT_1
			if (regcnt.w==1u)
				writeReg(reg, val.x);
			else
#endif
#ifdef PREFETCH_REG_COUNT_2
			if (regcnt.w==2u)
				writeReg(reg, val.xy);
			else
#endif
#ifdef PREFETCH_REG_COUNT_3
			if (regcnt.w==3u)
				writeReg(reg, val);
			else
#endif
			{} //else "empty braces"
		}
#endif
	}
}

void runNormalPrecompStream(in instr_stream_t stream, in mat2 dUV)
{
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);

		bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);

		uint srcreg = bsdf_data.data[0].z;
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
	currInteraction.T = -currInteraction.T;
	currInteraction.B = -currInteraction.B;
	currInteraction.TdotV = -currInteraction.TdotV;
	currInteraction.BdotV = -currInteraction.BdotV;
}
//call before executing an instruction/evaluating bsdf
void handleTwosided_interactionOnly(inout bool ts_flag, in instr_t instr)
{
#ifndef NO_TWOSIDED
	if (!ts_flag && instr_getTwosided(instr))
	{
		ts_flag = true;
		if (currInteraction.isotropic.NdotV<0.0)
		{
			flipCurrInteraction();
		}
	}
#ifdef OP_BUMPMAP
	ts_flag = instr_getOpcode(instr)==OP_BUMPMAP ? false:ts_flag;
#endif
#endif	
}
//call before executing an instruction/evaluating bsdf
void handleTwosided(inout bool ts_flag, in instr_t instr)
{
#ifndef NO_TWOSIDED
	if (!ts_flag && instr_getTwosided(instr))
	{
		ts_flag = true;
		if (currInteraction.isotropic.NdotV<0.0)
		{
			flipCurrInteraction();

			currBSDFParams.isotropic.NdotL = -currBSDFParams.isotropic.NdotL;
			currBSDFParams.isotropic.NdotH = -currBSDFParams.isotropic.NdotH;
			currBSDFParams.TdotL = -currBSDFParams.TdotL;
			currBSDFParams.BdotL = -currBSDFParams.BdotL;
			currBSDFParams.TdotH = -currBSDFParams.TdotH;
			currBSDFParams.BdotH = -currBSDFParams.BdotH;
		}
	}
#ifdef OP_BUMPMAP
	ts_flag = instr_getOpcode(instr)==OP_BUMPMAP ? false:ts_flag;
#endif
#endif
}

#ifdef GEN_CHOICE_STREAM
void instr_eval_and_pdf_execute(in instr_t instr, in irr_glsl_BSDFSample s)
{
	uint op = instr_getOpcode(instr);

	//speculative execution
	bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
	params_t params = instr_getParameters(instr, bsdf_data);
	uint ndf = instr_getNDF(instr);
	float a = params_getAlpha(params);
	float a2 = a*a;
	float n = irr_glsl_alpha2_to_phong_exp(a2);
#ifndef ALL_ISOTROPIC_BXDFS
	float ay = params_getAlphaV(params);
	float ay2 = ay*ay;
	float ny = irr_glsl_alpha2_to_phong_exp(ay2);
#endif
	float bxdf_eval_scalar_part = 0.0;
	float pdf = 0.0;

	float cosFactor = op_isBSDF(op) ? abs(currBSDFParams.isotropic.NdotL):max(currBSDFParams.isotropic.NdotL,0.0);

	if (cosFactor>FLT_MIN && op_hasSpecular(op))//does cos>0 check even has any point here? L comes from a sample..
	{
#ifdef OP_DIFFUSE
		if (op==OP_DIFFUSE) {
			pdf = irr_glsl_oren_nayar_pdf(s, currInteraction.isotropic, a2);
		} else
#endif

#ifdef OP_DIFFTRANS
		if (op==OP_DIFFTRANS) {
			//TODO take into account full sphere
			pdf = irr_glsl_lambertian_pdf(s, currInteraction.isotropic);
		}
		else
#endif //OP_DIFFTRANS

#ifdef NDF_GGX
		if (ndf==NDF_GGX) {

#ifdef ALL_ISOTROPIC_BXDFS
			bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, a2);
			pdf = irr_glsl_ggx_pdf(s, currInteraction.isotropic, a2);
#else
			bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_aniso_cos_eval_DG(currBSDFParams, currInteraction, a, ay);
			pdf = irr_glsl_ggx_pdf(s, currInteraction, a, ay, a2, ay2);
#endif

		} else
#endif

#ifdef NDF_BECKMANN
		if (ndf==NDF_BECKMANN) {

#ifdef ALL_ISOTROPIC_BXDFS
			bxdf_eval_scalar_part = irr_glsl_beckmann_smith_height_correlated_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, a2);
			pdf = irr_glsl_beckmann_pdf(s, currInteraction.isotropic, a2);
#else
			bxdf_eval_scalar_part = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_DG(currBSDFParams, currInteraction, a, a2, ay, ay2);
			pdf = irr_glsl_beckmann_pdf(s, currInteraction, a, a2, ay, ay2);
#endif

		} else
#endif

#ifdef NDF_PHONG
		if (ndf==NDF_PHONG) {

#ifdef ALL_ISOTROPIC_BXDFS
			bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, n, a2);
			pdf = irr_glsl_beckmann_pdf(s, currInteraction.isotropic, a2);
#else
			bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(currBSDFParams, currInteraction, n, ny, a2, ay2);
			pdf = irr_glsl_beckmann_pdf(s, currInteraction, a, a2, ay, ay2);
#endif

		} else
#endif
		{} //else "empty braces"
	}

	uvec3 regs = instr_decodeRegisters(instr);
#ifdef OP_DIFFUSE
	if (op==OP_DIFFUSE) {
		instr_execute_cos_eval_DIFFUSE(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_DIFFTRANS
	if (op==OP_DIFFTRANS) {
		instr_execute_cos_eval_DIFFTRANS(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_CONDUCTOR
	if (op==OP_CONDUCTOR) {
		instr_execute_cos_eval_CONDUCTOR(instr, regs, bxdf_eval_scalar_part, params, bsdf_data);
	} else
#endif
#ifdef OP_PLASTIC
	if (op==OP_PLASTIC) {
		pdf = instr_execute_cos_eval_pdf_PLASTIC(instr, regs, bxdf_eval_scalar_part, params, bsdf_data, s, pdf);
	} else
#endif
#ifdef OP_COATING
	if (op==OP_COATING) {
		instr_execute_cos_eval_COATING(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_DIELECTRIC
	if (op==OP_DIELECTRIC) {
		instr_execute_cos_eval_DIELECTRIC(instr, regs, params, bsdf_data);//TODO!!!
	} else
#endif
#ifdef OP_THINDIELECTRIC
	if (op==OP_THINDIELECTRIC) {
		instr_execute_cos_eval_THINDIELECTRIC(instr, regs, params, bsdf_data);//TODO!!!
	} else
#endif
#ifdef OP_BLEND
	if (op==OP_BLEND) {
		pdf = instr_execute_cos_eval_pdf_BLEND(instr, regs, params, bsdf_data);
	} else
#endif
#ifdef OP_BUMPMAP
	if (op==OP_BUMPMAP) {
		instr_execute_BUMPMAP(instr, s.L);
	} else
#endif
#ifdef OP_SET_GEOM_NORMAL
	if (op==OP_SET_GEOM_NORMAL) {
		instr_execute_SET_GEOM_NORMAL(s.L);
	} else
#endif
	{} //else "empty braces"

	if (op_isBXDForBlend(op))
		writeReg(REG_DST(regs)+3u, pdf);
}

eval_and_pdf_t irr_bsdf_eval_and_pdf(in instr_stream_t stream, in irr_glsl_BSDFSample s, in instr_t generator)
{
#ifndef NO_TWOSIDED
	bool ts = false;
#endif
	for (uint i = 0u; i < stream.count; ++i)
	{
		instr_t instr = irr_glsl_MC_fetchInstr(stream.offset+i);

		if (INSTR_1ST_DWORD(instr) != INSTR_1ST_DWORD(generator))
		{
#ifndef NO_TWOSIDED
			handleTwosided(ts, instr);
#endif
			instr_eval_and_pdf_execute(instr, s);
		}
		else
		{
			//assert(isBXDF(instr));
			uvec3 regs = instr_decodeRegisters(instr);
			writeReg(REG_DST(regs), eval_and_pdf_t(0.0));
		}
	}

	eval_and_pdf_t eval_and_pdf = readReg4(0u);
	return eval_and_pdf;
}

irr_glsl_BSDFSample irr_bsdf_cos_generate(in instr_stream_t stream, in vec3 rand, out vec3 out_remainder, out float out_pdf, out instr_t out_generatorInstr)
{
	uint ix = 0u;
	instr_t instr = irr_glsl_MC_fetchInstr(stream.offset);
	uint op = instr_getOpcode(instr);
	float weight = 1.0;
	vec3 u = rand;
	while (!op_isBXDF(op))
	{
#ifdef OP_BLEND
		if (op==OP_BLEND) {
			bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
			params_t params = instr_getParameters(instr, bsdf_data);
			float w = params_getBlendWeight(params);
			float rcpChoiceProb;
			bool choseRight = irr_glsl_partitionRandVariable(w, u.z, rcpChoiceProb);

			uint right = instr_getRightJump(instr);
			ix = choseRight ? right:(ix+1u);
			weight *= choseRight ? (1.0-w):w;
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
	const float ax = params_getAlpha(params);
	const float ax2 = ax*ax;
	const float ay = params_getAlphaV(params);
	const float ay2 = ay*ay;
	const mat2x3 ior = bsdf_data_decodeIoR(bsdf_data,op);
	const bool is_bsdf = !op_isBRDF(op);
	const vec3 refl = params_getReflectance(params);

#ifndef NO_TWOSIDED
	bool ts_flag = false;
	handleTwosided_interactionOnly(ts_flag, instr);
#endif //NO_TWOSIDED

	const bool is_plastic = op==OP_PLASTIC;

#ifdef OP_DIFFUSE

#define OP_DIFFUSE_ALIAS OP_DIFFUSE

#else

#define OP_DIFFUSE_ALIAS 0xdeadbeefu

#endif

	float pdf;
	vec3 rem;
	uint ndf = instr_getNDF(instr);
	irr_glsl_BSDFSample s;

#ifdef OP_PLASTIC
	float plastic_spec_weight;
	if (is_plastic) {
		vec3 fresnel = irr_glsl_fresnel_dielectric(ior[0],currBSDFParams.isotropic.VdotH);
		float ws = max(fresnel.x, max(fresnel.y, fresnel.z));
		bool choseDiffuse = u.z>=ws;
		op = choseDiffuse ? OP_DIFFUSE_ALIAS : op;
		plastic_spec_weight = ws;
	}
#endif

#if defined(OP_DIFFUSE) || defined(OP_PLASTIC)
	if (op==OP_DIFFUSE_ALIAS) {
		s = irr_glsl_oren_nayar_cos_generate(currInteraction, u.xy, ax2);
		rem = refl*irr_glsl_oren_nayar_cos_remainder_and_pdf(pdf, s, currInteraction.isotropic, ax2);
	} else 
#endif //OP_DIFFUSE

#ifdef OP_DIFFTRANS
	if (op==OP_DIFFTRANS) {
		//TODO take into account full sphere
		s = irr_glsl_cos_weighted_cos_generate(currInteraction, u.xy);
		rem = irr_glsl_cos_weighted_cos_remainder_and_pdf(pdf, s, currInteraction.isotropic);
	} else
#endif //OP_DIFFTRANS

//TODO need to distinguish BSDFs and BRDFs
#ifdef NDF_GGX
	if (ndf == NDF_GGX) {
		s = irr_glsl_ggx_cos_generate(currInteraction, u.xy, ax, ay);
		rem = irr_glsl_ggx_aniso_cos_remainder_and_pdf(pdf, s, currInteraction, ior, ax, ay);
	} else
#endif //NDF_GGX

#ifdef NDF_BECKMANN
	if (ndf == NDF_BECKMANN) {
		s = irr_glsl_beckmann_smith_cos_generate(currInteraction, u.xy, ax, ay);
		rem = irr_glsl_beckmann_aniso_cos_remainder_and_pdf(pdf, s, currInteraction, ior, ax, ax2, ay, ay2);
	} else
#endif //NDF_BECKMANN

#ifdef NDF_PHONG
	if (ndf == NDF_PHONG) {
		s = irr_glsl_beckmann_smith_cos_generate(currInteraction, u.xy, ax, ay);
		rem = irr_glsl_beckmann_aniso_cos_remainder_and_pdf(pdf, s, currInteraction, ior, ax, ax2, ay, ay2);
	} else
#endif //NDF_PHONG
	{}

#ifdef OP_PLASTIC
	if (is_plastic)
	{
		irr_glsl_updateBSDFParams(currBSDFParams, s, currInteraction);//TODO i could avoid this if using wo_clamps version of cos_eval()s

		//proposed weights: len(fresnel), len(reflectance)
		vec3 eval;
		float pdf_b;
		float wa;
		float wb;
		if (u.z>=plastic_spec_weight) { //chose diffuse as generator
			wb = plastic_spec_weight;
			wa = 1.0-wb;
#ifdef NDF_GGX
			if (ndf == NDF_GGX) {
				eval = irr_glsl_ggx_height_correlated_aniso_cos_eval(currBSDFParams, currInteraction, ior, ax, ay);
				pdf_b = irr_glsl_ggx_pdf(s, currInteraction, ax, ay, ax2, ay2);
			} else
#endif //NDF_GGX

#ifdef NDF_BECKMANN
			if (ndf == NDF_BECKMANN) {
				eval = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval(currBSDFParams, currInteraction, ior, ax, ax2, ay, ay2);
				pdf_b = irr_glsl_beckmann_pdf(s, currInteraction, ax, ax2, ay, ay2);
			} else
#endif //NDF_BECKMANN

#ifdef NDF_PHONG
			if (ndf == NDF_PHONG) {
				eval = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval(currBSDFParams, currInteraction, ior, ax, ax2, ay, ay2);
				pdf_b = irr_glsl_beckmann_pdf(s, currInteraction, ax, ax2, ay, ay2);
			} else
#endif //NDF_PHONG
			{}
		}
		else { //specular was generator
			wa = plastic_spec_weight;
			wb = 1.0-wa;
			eval = refl*irr_glsl_oren_nayar_cos_eval(currBSDFParams.isotropic, currInteraction.isotropic, ax2);//TODO fresnels and correction
			pdf_b = irr_glsl_oren_nayar_pdf(s, currInteraction.isotropic, ax2);
		}

		rem = (rem*wa + eval/pdf*wb)/(wa + pdf_b/pdf*wb);
		pdf = pdf*wa + pdf_b*wb;
	}
#endif

	out_remainder = weight*rem;
	out_pdf = weight*pdf;
	out_generatorInstr = instr;

	return s;
}

vec3 runGenerateAndRemainderStream(in instr_stream_t gcs, in instr_stream_t rnps, in vec3 rand, out float out_pdf, out irr_glsl_BSDFSample out_smpl)
{
	instr_t generator;
	vec3 generator_rem;
	float generator_pdf;
	irr_glsl_BSDFSample s = irr_bsdf_cos_generate(gcs, rand, generator_rem, generator_pdf, generator);
	irr_glsl_updateBSDFParams(currBSDFParams, s, currInteraction);
	eval_and_pdf_t eval_pdf = irr_bsdf_eval_and_pdf(rnps, s, generator);
	bxdf_eval_t acc = eval_pdf.rgb;
	float restPdf = eval_pdf.a;
	float pdf = generator_pdf + restPdf;

	out_smpl = s;

	vec3 rem = generator_rem/(1.0 + restPdf/generator_pdf) + acc/pdf;

	return rem;
}

#endif //GEN_CHOICE_STREAM

#endif