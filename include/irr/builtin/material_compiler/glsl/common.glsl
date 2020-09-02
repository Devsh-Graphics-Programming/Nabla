#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_INCLUDED_

#include <irr/builtin/material_compiler/glsl/common_declarations.glsl>

#ifndef _IRR_USER_PROVIDED_MATERIAL_COMPILER_GLSL_BACKEND_FUNCTIONS_
	#error "You need to define 'vec3 irr_glsl_MC_getCamPos()', 'instr_t irr_glsl_MC_fetchInstr(in uint)', 'bsdf_data_t irr_glsl_MC_fetchBSDFData(in uint)' functions above"
#endif

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
#define REG_PREFETCH0(r) REG_DST(r)
#define REG_PREFETCH1(r) REG_SRC1(r)
#define REG_PREFETCH2(r) REG_SRC2(r)
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

void writeReg(uint n, float v)
{
	registers[n] = floatBitsToUint(v);
}
void writeReg(uint n, vec2 v)
{
	writeReg(n   ,v.x);
	writeReg(n+1u,v.y);	
}
void writeReg(uint n, vec3 v)
{
	writeReg(n   ,v.x);
	writeReg(n+1u,v.y);
	writeReg(n+2u,v.z);
}
float readReg1(uint n)
{
	return uintBitsToFloat( registers[n] );
}
vec2 readReg2(uint n)
{
	return vec2(
		readReg1(n), readReg1(n+1u)
	);
}
vec3 readReg3(uint n)
{
	return vec3(
		readReg1(n), readReg1(n+1u), readReg1(n+2u)
	);
}

void setCurrBSDFParams(in vec3 n, in vec3 L)
{
	vec3 campos = irr_glsl_MC_getCamPos();
	irr_glsl_IsotropicViewSurfaceInteraction interaction = irr_glsl_calcFragmentShaderSurfaceInteraction(campos, WorldPos, n);
	currInteraction = irr_glsl_calcAnisotropicInteraction(interaction);
	irr_glsl_BSDFIsotropicParams isoparams = irr_glsl_calcBSDFIsotropicParams(interaction, L);
	currBSDFParams = irr_glsl_calcBSDFAnisotropicParams(isoparams, currInteraction);
}

float params_getAlpha(in params_t p)
{
	return p[0].x;
}
vec3 params_getReflectance(in params_t p)
{
	return p[3];
}
vec3 params_getOpacity(in params_t p)
{
	return p[2];
}
float params_getAlphaV(in params_t p)
{
	return p[1].x;
}
vec3 params_getSigmaA(in params_t p)
{
	return p[3];
}
float params_getBlendWeight(in params_t p)
{
	return p[0].x;
}
vec3 params_getTransmittance(in params_t p)
{
	return p[0];
}


#ifdef OP_DIFFUSE
void instr_execute_DIFFUSE(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
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
void instr_execute_DIFFTRANS(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	vec3 tr = params_getTransmittance(params);
	//transmittance*cos/2pi
	vec3 c = currBSDFParams.isotropic.NdotL*irr_glsl_RECIPROCAL_PI*0.5*tr;
	writeReg(REG_DST(regs), c);
}
#endif
#ifdef OP_DIELECTRIC
void instr_execute_DIELECTRIC(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
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
void instr_execute_CONDUCTOR(in instr_t instr, in uvec3 regs, in float DG, in params_t params, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		//float au = params_getAlpha(params);
		//float av = params_getAlphaV(params);
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
void instr_execute_PLASTIC(in instr_t instr, in uvec3 regs, in float DG, in params_t params, in bsdf_data_t data)
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
#endif
#ifdef OP_COATING
void instr_execute_COATING(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	//vec2 thickness_eta = unpackHalf2x16(data.data[3].x);
	//vec3 sigmaA = params_getSigmaA(params);
	//float a = params_getAlpha(params);
	writeReg(REG_DST(regs), bxdf_eval_t(1.0,0.0,0.0));
}
#endif
#ifdef OP_BUMPMAP
void instr_execute_BUMPMAP(in instr_t instr, in vec3 L)
{
	vec3 n = readReg3( REG_SRC1(instr_decodeRegisters(instr)) );
	setCurrBSDFParams(n, L);
}
#endif
#ifdef OP_SET_GEOM_NORMAL
//executed at most once
void instr_execute_SET_GEOM_NORMAL(in vec3 L)
{
	setCurrBSDFParams(normalize(Normal), L);
}
#endif
#ifdef OP_BLEND
void instr_execute_BLEND(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	float w = params_getBlendWeight(params);
	bxdf_eval_t bxdf1 = readReg3(REG_SRC1(regs));
	bxdf_eval_t bxdf2 = readReg3(REG_SRC2(regs));

	bxdf_eval_t blend = mix(bxdf1, bxdf2, w);
	writeReg(REG_DST(regs), blend);
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

#endif